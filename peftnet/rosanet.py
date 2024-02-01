from typing import Union
import logging

import torch.nn as nn

from peftnet.peft_module.rosalinear import RosaLinear, RosaLinearDebug
from peftnet._peftnet import PEFTNet, PEFTNetDebug


class RosaNet(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            adapt_method: str = 'ab',  # 'a', 'b', 'ab'
            sample_method: str = 'random',
            factorize_method: str = 'equal',  # 'equal', 'add'
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ ROSA PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            adapt_method: adaptation method [`a`, `b`, `ab`]
            sample_method: factorize mode [`random`, `top`, `bottom`]
            factorize_method: factorize method `w` \gets usv_1 + usv_2  (equal) or `w` \gets w + usv_2 (add)
            bias_requires_grad: whether to make bias trainable
            debug: whether to use debug mode
            fast_mode: whether to use fast mode (no checks)

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """
        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=RosaLinear,
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                adapt_method=adapt_method,
                sample_method=sample_method,
                factorize_method=factorize_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode
            ),
        )

        logging.info(f'Initialized ROSA model with params:')
        logging.info(
            f'rank: {rank}, '
            f'adapt_method: {adapt_method}, '
            f'sample_method: {sample_method}, '
            f'factorize_method: {factorize_method}, '
            f'bias_requires_grad: {bias_requires_grad} '
            f'debug: {debug}, '
            f'fast_mode: {fast_mode}'
        )

        # ROSA Model initializes low rank matrices with values obtained from SVD
        self.factorize()


class RosaNetDebug(PEFTNetDebug):
    def __init__(
            self,
            model: nn.Module,
            rank: Union[int, float],
            use_scale: bool = False,
            ignore_list: list = None,
            factorize_list: list = None,
            sample_method: str = 'random',
            factorize_method: str = 'equal',  # 'equal', 'add'
            bias_requires_grad: bool = True,
            debug: bool = False,
            fast_mode: bool = False,
            *args, **kwargs
    ):
        """ ROSA PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            sample_method: factorize mode [`random`, `top`, `bottom`]
            factorize_method: factorize method `w` \gets usv_1 + usv_2  (equal) or `w` \gets w + usv_2 (add)
            bias_requires_grad: whether to make bias trainable
            debug: whether to use debug mode
            fast_mode: whether to use fast mode (no checks)

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously

        """
        super().__init__(
            model,
            ignore_list=ignore_list,
            factorize_list=factorize_list,
            replacement_module=RosaLinearDebug,
            replacement_kwargs=dict(
                rank=rank,
                use_scale=use_scale,
                sample_method=sample_method,
                factorize_method=factorize_method,
                bias_requires_grad=bias_requires_grad,
                debug=debug,
                fast_mode=fast_mode
            ),
        )

        logging.info(f'Initialized ROSA model with params:')
        logging.info(
            f'rank: {rank}, '
             f'sample_method: {sample_method}, '
             f'factorize_method: {factorize_method}, '
             f'bias_requires_grad: {bias_requires_grad} '
             f'debug: {debug}, '
             f'fast_mode: {fast_mode}'
        )

        # ROSA Model initializes low rank matrices with values obtained from SVD
        self.factorize()
