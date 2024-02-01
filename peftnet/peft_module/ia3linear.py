from typing import Union

import torch
import torch.nn as nn


class IA3Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            mode: str = 'in',
            bias: bool = False
    ):
        """ IA3 linear layer with multiplicative trainable parameters.

        Args:
            in_features: number of input features
            out_features: number of output features
            mode: which side to apply multiplicative parameters [in, out, in_out]
            bias: whether to include bias

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.mode = mode
        if self.mode == 'in':
            self.d_in = nn.Parameter(torch.zeros(in_features))
            self.d_out = None
        elif self.mode == 'out':
            self.d_in = None
            self.d_out = nn.Parameter(torch.zeros(out_features))
        else:  # todo: try in_out
            self.d_in = nn.Parameter(torch.zeros(in_features))
            self.d_out = nn.Parameter(torch.zeros(out_features))

        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.register_buffer('merged', torch.tensor([False]))

    def initialize_weights(self, d_in_init: torch.Tensor = None, d_out_init: torch.Tensor = None,
                           w_init: torch.Tensor = None, bias_init: torch.Tensor = None):
        """Initialize weights and biases with given tensors."""
        if d_in_init is not None and d_out_init is not None:
            self.d_in.data = d_in_init
            self.d_out.data = d_out_init
        elif self.d_in is not None:
            self.d_in.data = d_in_init
        elif self.d_out is not None:
            self.d_out.data = d_out_init

        self.w.data = w_init if w_init is not None else self.w.data

        if bias_init is not None:
            assert self.bias.data.shape == bias_init.shape, "Bias shape mismatch"
            self.bias.data = bias_init

    @classmethod
    def from_module(cls, linear_layer: nn.Module, mode='in', fan_in_fan_out=True) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module"""

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), bias=bias is not None, mode=mode
        )

        if mode == 'in':
            d_in = torch.ones(obj.in_features, device=w.device)
            obj.initialize_weights(w_init=w, d_in_init=d_in, bias_init=bias)
        elif mode == 'out':
            d_out = torch.ones(obj.out_features, device=w.device)
            obj.initialize_weights(w_init=w, d_out_init=d_out, bias_init=bias)
        elif mode == 'in_out':
            d_in = torch.ones(obj.in_features, device=w.device)
            d_out = torch.ones(obj.out_features, device=w.device)
            obj.initialize_weights(w_init=w, d_in_init=d_in, d_out_init=d_out, bias_init=bias)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return obj

    def merge(self):
        """Merge `d` with `w` and make `w` trainable"""
        if not self.merged.item():
            # Merge w [in_f, out_f] with d [in_f] or [out_f]

            if self.mode == 'in':
                self.w.data = self.w.data * self.d_in.data.reshape(-1, 1)
                self.d_in.requires_grad = False
            elif self.mode == 'out':
                self.w.data = self.w.data * self.d_out.data.reshape(1, -1)
                self.d_out.requires_grad = False
            elif self.mode == 'in_out':
                self.w.data = (self.w.data * self.d_in.data.reshape(-1, 1)) * self.d_out.data.reshape(1, -1)
                self.d_in.requires_grad = False
                self.d_out.requires_grad = False

            # Make `d` fixed and `w` trainable
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = torch.tensor([True])
        return self

    def factorize(self, **kwargs):
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        # import pdb; pdb.set_trace()

        dstring = f'd_in={self.d_in.shape, self.d_in.requires_grad} ' if self.d_in is not None else f''
        dstring += f'd_out={self.d_out.shape, self.d_out.requires_grad} ' if self.d_out is not None else f''

        return (f'{cls}('
                f'{dstring}'                                            
                f'w={self.w.shape, self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None} '
                f'merged={self.merged.item()} '
                f'mode={self.mode}'
                f')')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        if self.merged.item() and self.training:
            raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged.item():
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias.reshape(-1) if self.bias is not None else x @ self.w
        else:
            if self.mode == 'in':
                # [in_f, out_f] * [in_f, 1] = [in_f, out_f]
                w_scaled = self.w * self.d_in.reshape(-1, 1)
            elif self.mode == 'out':
                # [in_f, out_f] * [1, out_f] = [in_f, out_f]
                w_scaled = self.w * self.d_out.reshape(1, -1)
            elif self.mode == 'in_out':
                # [in_f, out_f] * [in_f, 1] * [1, out_f] = [in_f, out_f]
                w_scaled = (self.w * self.d_in.reshape(-1, 1)) * self.d_out.reshape(1, -1)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            # [*, in_f] @ [in_f, out_f] + [out_f] = [*, out_features]
            return x @ w_scaled + self.bias.reshape(-1) if self.bias is not None else x @ w_scaled
