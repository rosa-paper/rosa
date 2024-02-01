from typing import Union
import logging

import torch
import torch.nn as nn
import numpy as np


class PeftLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False,
            use_scale: bool = False,
            alpha: float = 32.0,
            adapt_method: str = 'ab',  # 'ab', 'a', 'b'
            sample_method: str = 'random',
            factorize_method: str = 'svd_equal',  # 'svd_equal', 'svd_add', 'random_proj', 'random_proj_orthogonal'
            init_method: str = 'zero',  # 'zero', 'random'
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False
    ):
        """ PEFT linear layer with trainable and fixed parameters in parallel.

        Args:
            in_features: number of input features
            out_features: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias
            use_scale: whether to use scale factor
            alpha: scale factor
            adapt_method: which parameters to adapt [`ab`, `a`, `b`] (default: `ab`)
            sample_method: sample method [`random`, `top`, `bottom`]
            factorize_method: factorize method `w` \gets usv_1 + usv_2  (equal) or `w` \gets w + usv_2 (add)
            init_method: initialization method for `a` [`zero`, `random`]
            debug: whether to use debug mode

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()

        # Input validation
        assert isinstance(in_features, int) and isinstance(out_features, int), \
            "in_features and out_features must be integers"
        assert isinstance(rank, (int, float)), "rank must be an integer or a float"
        assert isinstance(bias, bool), "bias must be a boolean"
        assert isinstance(use_scale, bool), "use_scale must be a boolean"
        assert isinstance(alpha, (int, float)), "alpha must be an integer or a float"
        assert factorize_method in ['svd_equal', 'svd_add', 'random_add', 'random_proj', 'random_proj_orthogonal'], \
            "factorize_method must be one of ['svd_equal', 'svd_add', 'random', 'random_orthogonal']"
        assert sample_method in ['random', 'top', 'bottom'], \
            "sample_method must be one of ['random', 'top', 'bottom']"
        assert init_method in ['zero', 'random'], "init_method must be one of ['zero', 'random']"
        assert adapt_method in ['ab', 'a', 'b'], "adapt_method must be one of ['ab', 'a', 'b']"

        self.in_features = in_features
        self.out_features = out_features
        self.rank = self._integer_rank(rank, full_rank=min(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias_requires_grad) if bias else None
        self.use_scale = use_scale
        self.alpha = alpha
        self.adapt_method = adapt_method
        self.sample_method = sample_method
        self.factorize_method = factorize_method
        self.init_method = init_method
        self.debug = debug
        self.fast_mode = fast_mode

        # Set requires_grad for a and b
        self.requires_grad_a = True if self.adapt_method in ['ab', 'a'] else False
        self.requires_grad_b = True if self.adapt_method in ['ab', 'b'] else False

        a_init_func = torch.zeros if self.init_method == 'zero' else torch.randn
        self.a = nn.Parameter(a_init_func(in_features, self.rank), requires_grad=self.requires_grad_a)
        self.b = nn.Parameter(torch.randn(self.rank, out_features), requires_grad=self.requires_grad_b)
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.register_buffer('merged', torch.tensor([False]))

    def initialize_weights(self, a_init: torch.Tensor = None, b_init: torch.Tensor = None, w_init: torch.Tensor = None,
                           bias_init: torch.Tensor = None):
        """Initialize weights and biases with given tensors."""
        self.a.data = a_init if a_init is not None else self.a.data
        self.b.data = b_init if b_init is not None else self.b.data
        self.w.data = w_init if w_init is not None else self.w.data

        if bias_init is not None:
            assert self.bias.data.shape == bias_init.shape, "Bias shape mismatch"
            self.bias.data = bias_init

    @classmethod
    def from_module(cls, linear_layer: nn.Module, rank=1.0, fan_in_fan_out=True, init_method: str = 'zero',
                    *args, **kwargs) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module

        Args:
            linear_layer: linear layer to initialize from
            rank: rank of factorized matrices
            fan_in_fan_out: if true assumes weight is in [fan_in, fan_out] format
            init_method:
            *args:
            **kwargs:

        Returns:
            obj: initialized PEFTLinear object

        Note:
            - Huggingface Conv1D is in [in_f, out_f] format. Set `fan_in_fan_out`=False
            - PyTorch Linear is in [out_f, in_f] format. Set `fan_in_fan_out`=False

        """

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T  # [in_f, out_f]
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), rank=rank, bias=bias is not None, *args, **kwargs
        )
        a = torch.zeros(obj.in_features, obj.rank, device=w.device) if init_method == 'zero' else \
            torch.randn(obj.in_features, obj.rank, device=w.device)
        b = torch.randn(obj.rank, obj.out_features, device=w.device)
        obj.initialize_weights(w_init=w, a_init=a, b_init=b, bias_init=bias)
        return obj

    def merge(self):
        """Merge `a` and `b` with `w` and make `w` trainable"""
        if not self.merged.item():
            # Merge w and ab
            self.w.data = (self.alpha/self.rank) * self.a.data @ self.b.data + self.w.data if self.use_scale \
                else self.a.data @ self.b.data + self.w.data

            # todo: empty a and b tensors to save memory
            # Make a, b fixed and w trainable
            self.a.requires_grad = False
            self.b.requires_grad = False
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = torch.tensor([True])
        return self

    def factorize(self):
        """Factorize `w` into `a` and `b` and make a portion of `a` and `b` trainable"""
        with torch.no_grad():
            if not self.merged:
                self.merge()

            rank_upper_bound = min(self.w.shape)
            if self.rank >= rank_upper_bound:
                # If rank is larger than the rank upper bound, train the whole layer
                return self

            if self.factorize_method == 'random_add':
                init_a_trainable = torch.zeros(self.in_features, self.rank, device=self.w.device)
                init_b_trainable = torch.randn(self.rank, self.out_features, device=self.w.device)
                init_w = self.w.data

            elif self.factorize_method == 'random_proj':
                # Generate random ortho matrix and take first r columns
                random_matrix = torch.randn(self.w.shape[1], self.w.shape[1], device=self.w.device)
                q = random_matrix[:, :self.rank]

                init_a_trainable = q
                init_b_trainable = q.T @ self.w  # [r, d] [d, p] => [r, p]
                init_w = self.w - q @ (q.T @ self.w)

            elif self.factorize_method == 'random_proj_orthogonal':
                # Generate random ortho matrix and take first r columns
                random_matrix = torch.randn(self.w.shape[1], self.w.shape[1], device=self.w.device)
                q, _ = torch.linalg.qr(random_matrix)
                q = q[:, :self.rank]

                init_a_trainable = q
                init_b_trainable = q.T @ self.w  # [r, d] [d, p] => [r, p]
                init_w = self.w - q @ (q.T @ self.w)

            else:
                # Factorize
                u, s, vt = torch.linalg.svd(self.w.data, full_matrices=False)  # [in_f, r],[r,],[r, out_f]
                a = s.reshape(1, -1) * u
                b = vt
                w_hat = a @ b

                # Check reconstruction error
                if not self.fast_mode:
                    assert torch.allclose(self.w.data, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"
                trainable_indices, fixed_indices = self._select_k_from_n(self.rank, rank_upper_bound, mode=self.sample_method)

                # Set trainable and fixed parameters
                init_a_trainable = a[:, trainable_indices]  # [in_f, r']
                init_b_trainable = b[trainable_indices, :]  # [r', out_f]
                init_a_fixed = a[:, fixed_indices]
                init_b_fixed = b[fixed_indices, :]

                if self.factorize_method == 'svd_equal':  # (regular) init + ortho
                    init_w = init_a_fixed @ init_b_fixed
                elif self.factorize_method == 'svd_add':  # init
                    init_w = self.w.data
                else:
                    raise AttributeError(
                        f"Unknown factorize method: {self.factorize_method}. Method must be one of ['svd_equal', 'svd_add']"
                    )

            # Initialize
            self.a.data = init_a_trainable
            self.b.data = init_b_trainable
            self.w.data = init_w

            # Make contiguous
            self.a.data = self.a.data.contiguous()
            self.b.data = self.b.data.contiguous()
            self.w.data = self.w.data.contiguous()

            # Make `a`, `b` trainable and `w` fixed
            self.a.requires_grad = self.requires_grad_a
            self.b.requires_grad = self.requires_grad_b
            self.w.requires_grad = False

            # Toggle merged flag
            self.merged = torch.tensor([False])
            return self

    @staticmethod
    def _generate_random_orthogonal_matrix(n, r):
        """Generate a random orthogonal matrix Q of size nxr, where r < n and Q^TQ = I.

        Args:
            n (int): Number of rows of the matrix.
            r (int): Number of columns of the matrix, where r < n.

        Returns:
            numpy.ndarray: An nxr orthogonal matrix Q.
        """
        if r >= n:
            raise ValueError("r must be less than n.")

        # Generate a random nxn matrix
        random_matrix = np.random.randn(n, n)

        # Perform QR decomposition on the random matrix
        Q, _ = np.linalg.qr(random_matrix)

        # Return the first r columns of Q
        return Q[:, :r]

    def __repr__(self):
        cls = self.__class__.__name__
        return (f'{cls}('
                f'rank={self.rank}, '
                f'a={self.a.shape} grad={self.a.requires_grad} scale={self.use_scale}, alpha={self.alpha}, '
                f'b={self.b.shape} grad={self.b.requires_grad}, '
                f'w={self.w.shape} grad={self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
                f')')

    @staticmethod
    def _integer_rank(rank, full_rank):
        """Convert a ratio to an integer"""
        return rank if isinstance(rank, int) else max(int(rank * full_rank), 1)

    @staticmethod
    def _select_k_from_n(k: int, n: int, mode: str = 'random'):
        """Choose `k` indices from `n` indices"""

        assert 0 < k < n, f"k must be an integer between 0 and n, received k={k}, n={n}"
        assert isinstance(k, int) and isinstance(n, int), "k and n must be integers"

        if mode.lower() == 'random':
            # Select k random indices from n indices
            start_idx = torch.randint(0, n - k, (1,)).item()
            end_idx = start_idx + k
            chosen_ids = torch.arange(start_idx, end_idx)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'top':
            # Select top k indices from n indices
            chosen_ids = torch.arange(0, min(k, n - k))
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'bottom':
            # Select bottom k indices from n indices
            chosen_ids = torch.arange(max(0, n - k), n)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        else:
            raise AttributeError(f"Unknown mode: {mode}. Mode must be one of ['random', 'top', 'bottom']")

        return chosen_ids, remaining_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. Assumes `w` is fan_in x fan_out.

        Args:
            x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        # if self.merged.item() and self.training:
        #     raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged.item():
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias.reshape(-1) if self.bias is not None else x @ self.w
        elif self.debug:  # retain intermediate gradient (for plotting purposes)
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            a = (self.alpha / self.rank) * self.a if self.use_scale else self.a
            self.ab = a @ self.b
            self.ab.retain_grad()
            return (x @ self.ab) + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else (x @ self.ab) + x @ self.w
        else:
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            a = (self.alpha/self.rank) * self.a if self.use_scale else self.a
            return (x @ a) @ self.b + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else (x @ a) @ self.b + x @ self.w


class PeftLinearDebug(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False,
            use_scale: bool = False,
            alpha: float = 32.0,
            sample_method: str = 'random',
            adapt_method: str = 'ab',  # 'ab', 'a', 'b'
            factorize_method: str = 'equal',  # 'equal', 'add', 'random'
            init_method: str = 'zero',  # 'zero', 'random'
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
    ):
        """ PEFT linear layer with trainable and fixed parameters in parallel.

        Args:
            in_features: number of input features
            out_features: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias
            use_scale: whether to use scale factor
            alpha: scale factor
            adapt_method: which parameters to adapt [`ab`, `a`, `b`] (default: `ab`)
            sample_method: factorize mode [`random`, `top`, `bottom`]
            factorize_method: factorize method `w` \gets usv_1 + usv_2  (equal) or `w` \gets w + usv_2 (add)
            init_method: initialization method for `a` [`zero`, `random`]
            debug: whether to use debug mode

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()

        # Input validation
        assert isinstance(in_features, int) and isinstance(out_features, int), \
            "in_features and out_features must be integers"
        assert isinstance(rank, (int, float)), "rank must be an integer or a float"
        assert isinstance(bias, bool), "bias must be a boolean"
        assert isinstance(use_scale, bool), "use_scale must be a boolean"
        assert isinstance(alpha, (int, float)), "alpha must be an integer or a float"
        assert factorize_method in ['equal', 'add'], "factorize_method must be one of ['equal', 'add']"
        assert sample_method in ['random', 'top', 'bottom'], \
            "sample_method must be one of ['random', 'top', 'bottom']"
        assert init_method in ['zero', 'random'], "init_method must be one of ['zero', 'random']"
        assert adapt_method in ['ab', 'a', 'b'], "adapt_method must be one of ['ab', 'a', 'b']"

        self.in_features = in_features
        self.out_features = out_features
        self.rank = self._integer_rank(rank, full_rank=min(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias_requires_grad) if bias else None
        self.use_scale = use_scale
        self.alpha = alpha
        self.adapt_method = adapt_method
        self.sample_method = sample_method
        self.factorize_method = factorize_method
        self.init_method = init_method
        self.debug = debug
        self.fast_mode = fast_mode

        # Set requires_grad for a and b
        requires_grad_a = True if self.adapt_method in ['ab', 'a'] else False
        requires_grad_b = True if self.adapt_method in ['ab', 'b'] else False

        a_init_func = torch.zeros if self.init_method == 'zero' else torch.randn
        self.a = nn.Parameter(a_init_func(in_features, self.rank), requires_grad=requires_grad_a)
        self.b = nn.Parameter(torch.randn(self.rank, out_features), requires_grad=requires_grad_b)
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.register_buffer('merged', torch.tensor([False]))


    def initialize_weights(self, a_init: torch.Tensor = None, b_init: torch.Tensor = None, w_init: torch.Tensor = None,
                           bias_init: torch.Tensor = None):
        """Initialize weights and biases with given tensors."""
        self.a.data = a_init if a_init is not None else self.a.data
        self.b.data = b_init if b_init is not None else self.b.data
        self.w.data = w_init if w_init is not None else self.w.data

        if bias_init is not None:
            assert self.bias.data.shape == bias_init.shape, "Bias shape mismatch"
            self.bias.data = bias_init

    @classmethod
    def from_module(cls, linear_layer: nn.Module, rank=1.0, fan_in_fan_out=True, init_method: str = 'zero',
                    *args, **kwargs) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module

        Args:
            linear_layer: linear layer to initialize from
            rank: rank of factorized matrices
            fan_in_fan_out: if true assumes weight is in [fan_in, fan_out] format
            init_method:
            *args:
            **kwargs:

        Returns:
            obj: initialized PEFTLinear object

        Note:
            - Huggingface Conv1D is in [in_f, out_f] format. Set `fan_in_fan_out`=False
            - PyTorch Linear is in [out_f, in_f] format. Set `fan_in_fan_out`=False

        """

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T  # [in_f, out_f]
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), rank=rank, bias=bias is not None, *args, **kwargs
        )
        a = torch.zeros(obj.in_features, obj.rank, device=w.device) if init_method == 'zero' else \
            torch.randn(obj.in_features, obj.rank, device=w.device)
        b = torch.randn(obj.rank, obj.out_features, device=w.device)
        obj.initialize_weights(w_init=w, a_init=a, b_init=b, bias_init=bias)
        return obj

    def merge(self):
        """Merge `a` and `b` with `w` and make `w` trainable"""
        if not self.merged.item():
            # Merge w and ab
            self.w.data = (self.alpha/self.rank) * self.a.data @ self.b.data + self.w.data if self.use_scale \
                else self.a.data @ self.b.data + self.w.data

            # todo: empty a and b tensors to save memory
            # Make a, b fixed and w trainable
            self.a.requires_grad = False
            self.b.requires_grad = False
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = torch.tensor([True])
        return self

    def factorize(self):
        """Factorize `w` into `a` and `b` and make a portion of `a` and `b` trainable"""
        with torch.no_grad():
            if not self.merged:
                self.merge()

            rank_upper_bound = min(self.w.shape)
            if self.rank >= rank_upper_bound:
                # If rank is larger than the rank upper bound, train the whole layer
                return self

            # Factorize
            # u, s, vt = torch.linalg.svd(self.w.data, full_matrices=False)  # [in_f, r],[r,],[r, out_f]
            # a = s.reshape(1, -1) * u
            # b = vt
            # w_hat = a @ b

            # Check reconstruction error
            # if not self.fast_mode:
                # assert torch.allclose(self.w.data, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"
            # trainable_indices, fixed_indices = self._select_k_from_n(self.rank, rank_upper_bound, mode=self.sample_method)

            # Set trainable and fixed parameters
            # init_a_trainable = a[:, trainable_indices]  # [in_f, r']
            # init_b_trainable = b[trainable_indices, :]  # [r', out_f]
            # init_a_fixed = a[:, fixed_indices]
            # init_b_fixed = b[fixed_indices, :]

            init_a_trainable = torch.randn(self.in_features, self.rank, device=self.w.device)
            init_b_trainable = torch.randn(self.rank, self.out_features, device=self.w.device)
            # init_a_fixed = torch.randn(self.in_features, self.rank, device=self.w.device)
            # init_b_fixed = torch.randn(self.rank, self.out_features, device=self.w.device)

            if self.factorize_method == 'equal':
                # init_w = init_a_fixed @ init_b_fixed
                init_w = torch.randn(self.in_features, self.out_features, device=self.w.device)
            elif self.factorize_method == 'add':
                init_w = self.w.data
            else:
                raise AttributeError(
                    f"Unknown factorize method: {self.factorize_method}. Method must be one of ['equal', 'add']"
                )

            # Initialize
            self.a.data = init_a_trainable
            self.b.data = init_b_trainable
            self.w.data = init_w

            # Make contiguous
            self.a.data = self.a.data.contiguous()
            self.b.data = self.b.data.contiguous()
            self.w.data = self.w.data.contiguous()

            # Make `a`, `b` trainable and `w` fixed
            self.a.requires_grad = True
            self.b.requires_grad = True
            self.w.requires_grad = False

            # Toggle merged flag
            self.merged = torch.tensor([False])
            return self

    def __repr__(self):
        cls = self.__class__.__name__
        return (f'{cls}('
                f'rank={self.rank}, '
                f'a={self.a.shape} grad={self.a.requires_grad} scale={self.use_scale}, alpha={self.alpha}, '
                f'b={self.b.shape} grad={self.b.requires_grad}, '
                f'w={self.w.shape} grad={self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
                f')')

    @staticmethod
    def _integer_rank(rank, full_rank):
        """Convert a ratio to an integer"""
        return rank if isinstance(rank, int) else max(int(rank * full_rank), 1)

    @staticmethod
    def _select_k_from_n(k: int, n: int, mode: str = 'random'):
        """Choose `k` indices from `n` indices"""

        assert 0 < k < n, f"k must be an integer between 0 and n, received k={k}, n={n}"
        assert isinstance(k, int) and isinstance(n, int), "k and n must be integers"

        if mode.lower() == 'random':
            # Select k random indices from n indices
            start_idx = torch.randint(0, n - k, (1,)).item()
            end_idx = start_idx + k
            chosen_ids = torch.arange(start_idx, end_idx)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'top':
            # Select top k indices from n indices
            chosen_ids = torch.arange(0, min(k, n - k))
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'bottom':
            # Select bottom k indices from n indices
            chosen_ids = torch.arange(max(0, n - k), n)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        else:
            raise AttributeError(f"Unknown mode: {mode}. Mode must be one of ['random', 'top', 'bottom']")

        return chosen_ids, remaining_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. Assumes `w` is fan_in x fan_out.

        Args:
            x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        # if self.merged.item() and self.training:
        #     raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged.item():
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias.reshape(-1) if self.bias is not None else x @ self.w
        elif self.debug:  # retain intermediate gradient (for plotting purposes)
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            a = (self.alpha / self.rank) * self.a if self.use_scale else self.a
            self.ab = a @ self.b
            self.ab.retain_grad()
            return (x @ self.ab) + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else (x @ self.ab) + x @ self.w
        else:
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            a = (self.alpha/self.rank) * self.a if self.use_scale else self.a
            return (x @ a) @ self.b + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else (x @ a) @ self.b + x @ self.w

# [2023-09-27 19:00:42,795][root][INFO] - [Epoch    2 Step  100/ 268] | loss  0.34 | trainable: 294,912 | lr: 0.001396
# [2023-09-27 19:00:42,796][root][INFO] - Latency Report:  | forward  241 ms | loss.mean()  242 ms | loss.backward()  526 ms | optimizer.step()  529 ms
# [2023-09-27 19:00:42,796][root][INFO] - Memory Report: [train_epoch] Initial  607 MB | [train_epoch] After model to device  607 MB | [train_epoch] After batch to device  610 MB | [train_epoch] After forward 17617 MB | [train_epoch] After loss.backward()  611 MB | [train_epoch] After optimizer.step()  611 MB | [train_epoch] After optimizer.zero_grad()  610 MB


#
# [Epoch    4 Step  400/ 535] | loss  0.03 | trainable: 124,647,170 | lr: 0.000001
# Latency Report:  | forward   80 ms | loss.mean()   81 ms | loss.backward()  207 ms | optimizer.step()  217 ms
# Memory Report: [train_epoch] Initial 1510 MB | [train_epoch] After model to device 1510 MB | [train_epoch] After batch to device 1510 MB | [train_epoch] After forward 6494 MB | [train_epoch] After loss.backward() 1988 MB | [train_epoch] After optimizer.step() 1988 MB | [train_epoch] After optimizer.zero_grad() 1510 MB
