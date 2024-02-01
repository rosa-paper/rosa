from typing import Union
from ._peftlinear import PeftLinear


class LoraLinear(PeftLinear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False,
            *args, **kwargs
    ):
        """ LORA linear layer with trainable and fixed parameters in parallel.

        Args:
            in_features: number of input features
            out_features: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias,
            *args, **kwargs
        )
