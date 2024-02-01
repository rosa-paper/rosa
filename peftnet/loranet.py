from typing import Union
import logging

import torch.nn as nn

from peftnet.peft_module.loralinear import LoraLinear
from peftnet._peftnet import PEFTNet, PEFTNetDebug


class LoraNet(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            adapt_method: str = 'ab',  # 'a', 'b', 'ab'
            init_method: str = "zero",
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ LoRa PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            adapt_method: adaptation method [`a`, `b`, `ab`]
            debug: whether to use debug mode

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """
        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=LoraLinear,
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                adapt_method=adapt_method,
                init_method=init_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode
            ),
        )

        logging.info(f'Initialized LoRA model with params:')
        logging.info(
            f'rank: {rank}, '
             f'bias_requires_grad: {bias_requires_grad} '
             f'debug: {debug}, '
             f'fast_mode: {fast_mode}'
        )


class LoraNetDebug(PEFTNetDebug):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            init_method: str = "zero",
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ LoRa PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            debug: whether to use debug mode

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """
        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=LoraLinear,
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                init_method=init_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode
            ),
        )

        logging.info(f'Initialized LoRA model with params:')
        logging.info(
            f'rank: {rank}, '
             f'bias_requires_grad: {bias_requires_grad} '
             f'debug: {debug}, '
             f'fast_mode: {fast_mode}'
        )
