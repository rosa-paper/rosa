from typing import Union

import torch.nn as nn

from peftnet.peft_module.ia3linear import IA3Linear
from peftnet._peftnet import PEFTNet


class IA3Net(PEFTNet):
    def __init__(
            self,
            model: nn.Module,
            ignore_list: list = None,
            peft_list: list = None,
            ia3_mode: str = 'in',
            *args, **kwargs
    ):
        """ IA3 PEFT model for efficient adaptation of linear layers

        Args:
            model: model to be factorized
            rank: rank of factorized matrices
            ignore_list: names of layers to ignore
            peft_list: names of modules types to replace with peft module

        Notes:
            - only modules types in `peft_list` will be replaced with peft module

        """
        super().__init__(
            model,
            ignore_list,
            peft_list,
            replacement_module=IA3Linear,
            replacement_kwargs=dict(mode=ia3_mode),
        )
