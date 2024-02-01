import torch.nn as nn
import pandas as pd


class PEFTNet(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            ignore_list: list = None,
            factorize_list: list = None,
            fan_in_fan_out_map: dict = None,
            replacement_module: nn.Module = None,
            replacement_kwargs: dict = None,
            *args, **kwargs
    ):
        """ Abstract class for PEFT models. PEFT models are models that can be factorized and merged.

        Args:
            model: model to be factorized
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            fan_in_fan_out_map: map from module type to fan_in_fan_out flag
            replacement_module: replacement module
            replacement_kwargs: kwargs for replacement module constructor

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously


        """
        super().__init__(*args, **kwargs)

        # Warning
        assert (factorize_list is None and fan_in_fan_out_map is None) or \
               (factorize_list is not None and fan_in_fan_out_map is not None), \
            "`factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously"

        self.factorize_list = ['Linear', 'Conv1D'] if factorize_list is None else factorize_list
        self.fan_in_fan_out_map = {
            "Conv1D": True, "Linear": False
        } if fan_in_fan_out_map is None else fan_in_fan_out_map
        self.ignore_list = list() if ignore_list is None else ignore_list
        self.replacement_module = replacement_module
        self.factorize_map = {f: replacement_module for f in self.factorize_list}
        self.replacement_kwargs = replacement_kwargs if replacement_kwargs is not None else dict()

        # Make modules non-trainable
        for param in model.parameters():
            param.requires_grad = False

        self.peft_model = model
        self.apply_peft()

    def apply_peft(self) -> 'PEFTNet':
        """Replace linear modules with peft modules"""
        condition = lambda lyr, name: type(lyr).__name__ in self.factorize_map.keys() and \
                                      name not in self.ignore_list
        replacement_function = lambda lyr: self.factorize_map[type(lyr).__name__].from_module(
            lyr, fan_in_fan_out=self.fan_in_fan_out_map[type(lyr).__name__], **self.replacement_kwargs
        )
        return self._replace(condition, replacement_function)

    def merge(self) -> 'PEFTNet':
        """Apply `merge` on peft modules"""
        condition = lambda lyr, name: isinstance(lyr, self.replacement_module)
        replacement_function = lambda lyr: lyr.merge()
        return self._replace(condition, replacement_function)

    def factorize(self) -> 'PEFTNet':
        """Apply `factorize` on peft modules. If a module is already factorized, it will be merged and re-factorized"""
        condition = lambda lyr, name: isinstance(lyr, self.replacement_module)
        replacement_function = lambda lyr: lyr.factorize()
        return self._replace(condition, replacement_function)

    def _replace(self, condition: callable, replacement_function: callable) -> 'PEFTNet':
        """Replace modules that satisfy a condition using a replacement function"""
        for name, layer in self.peft_model.named_modules():
            if condition(layer, name):
                replacement_module = replacement_function(layer)
                replacement_address = self._parse_model_addr(name)
                self._set_module(self.peft_model, replacement_address, replacement_module)

        return self

    def get_report(self) -> str:
        """Get report on factorized model"""

        safe_div = lambda x, y: x / y if y != 0 else 0

        # Trainable params table
        df = pd.DataFrame()
        for name, layer in self.peft_model.named_modules():
            params_dict = self.get_num_params(layer)

            df.at[name, 'name'] = name
            df.at[name, 'type'] = type(layer).__name__
            df.at[name, '# train'] = params_dict['trainable']
            df.at[name, '# fixed'] = params_dict['fixed']
            df.at[name, 'total'] = params_dict['total']
            df.at[name, '% train'] = round(safe_div(params_dict['trainable'], params_dict['total']) * 100, 2)

        # Set the 'name' column as the index
        df.set_index('name', inplace=True)

        # Return string
        return df.to_string()

    @staticmethod
    def get_num_params(model: nn.Module) -> dict:
        params_dict = {k: 0 for k in ["trainable", "fixed", "total"]}
        for p in model.parameters():
            params_dict['total'] += p.numel()
            if p.requires_grad:
                params_dict['trainable'] += p.numel()
            else:
                params_dict['fixed'] += p.numel()

        params_dict = {k: v / 1e6 for k, v in params_dict.items()}
        return params_dict

    def _get_module(self, parent: nn.Module, replacement_addr_list: list) -> nn.Module:
        """ Recursive function used to access child modules from a parent nn.Module object

        Args:
            replacement_addr_list: specifies how to access target object from ancestor object.
                ['layer1', 0, 'conv2']
        Returns:
            target object/layer to be replaced.
        """

        if len(replacement_addr_list) == 0:
            return parent
        else:
            attr = replacement_addr_list.pop(0)

            # attr can be accessible in two ways
            child = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
            return self._get_module(child, replacement_addr_list)

    def _set_module(self, model: nn.Module, replacement_addr_list: list, replacement_layer: nn.Module) -> None:
        """ Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer` """
        if isinstance(replacement_addr_list[-1], int):
            self._get_module(model, replacement_addr_list[:-1])[replacement_addr_list[-1]] = replacement_layer
        else:
            setattr(self._get_module(model, replacement_addr_list[:-1]), replacement_addr_list[-1], replacement_layer)

    @staticmethod
    def _parse_model_addr(access_str: str) -> list:
        """ Parses path to child from a parent. E.g., layer1.0.conv2 ==> ['layer1', 0, 'conv2'] """
        parsed = access_str.split('.')
        for i in range(len(parsed)):
            try:
                parsed[i] = int(parsed[i])
            except ValueError:
                pass
        return parsed

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)


class PEFTNetDebug(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            ignore_list: list = None,
            factorize_list: list = None,
            fan_in_fan_out_map: dict = None,
            replacement_module: nn.Module = None,
            replacement_kwargs: dict = None,
            *args, **kwargs
    ):
        """ Abstract class for PEFT models. PEFT models are models that can be factorized and merged.

        Args:
            model: model to be factorized
            ignore_list: names of layers to ignore
            factorize_list: names of modules types to replace
            fan_in_fan_out_map: map from module type to fan_in_fan_out flag
            replacement_module: replacement module
            replacement_kwargs: kwargs for replacement module constructor

        Notes:
            - only modules types in `factorize_list` will be factorized
            - `factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously


        """
        super().__init__(*args, **kwargs)

        # Warning
        assert (factorize_list is None and fan_in_fan_out_map is None) or \
               (factorize_list is not None and fan_in_fan_out_map is not None), \
            "`factorize_list` and `fan_in_fan_out_map` must be specified/unspecified simultaneously"

        self.factorize_list = ['Linear', 'Conv1D'] if factorize_list is None else factorize_list
        self.fan_in_fan_out_map = {
            "Conv1D": True, "Linear": False
        } if fan_in_fan_out_map is None else fan_in_fan_out_map
        self.ignore_list = list() if ignore_list is None else ignore_list
        self.replacement_module = replacement_module
        self.factorize_map = {f: replacement_module for f in self.factorize_list}
        self.replacement_kwargs = replacement_kwargs if replacement_kwargs is not None else dict()

        # Make modules non-trainable
        for param in model.parameters():
            param.requires_grad = False

        self.peft_model = model
        self.apply_peft()

    def apply_peft(self) -> 'PEFTNet':
        """Replace linear modules with peft modules"""
        condition = lambda lyr, name: type(lyr).__name__ in self.factorize_map.keys() and \
                                      name not in self.ignore_list
        replacement_function = lambda lyr: self.factorize_map[type(lyr).__name__].from_module(
            lyr, fan_in_fan_out=self.fan_in_fan_out_map[type(lyr).__name__], **self.replacement_kwargs
        )
        return self._replace(condition, replacement_function)

    def merge(self) -> 'PEFTNet':
        """Apply `merge` on peft modules"""
        condition = lambda lyr, name: isinstance(lyr, self.replacement_module)
        replacement_function = lambda lyr: lyr.merge()
        return self._replace(condition, replacement_function)

    def factorize(self) -> 'PEFTNet':
        """Apply `factorize` on peft modules. If a module is already factorized, it will be merged and re-factorized"""
        condition = lambda lyr, name: isinstance(lyr, self.replacement_module)
        replacement_function = lambda lyr: lyr.factorize()
        return self._replace(condition, replacement_function)

    def _replace(self, condition: callable, replacement_function: callable) -> 'PEFTNet':
        """Replace modules that satisfy a condition using a replacement function"""
        for name, layer in self.peft_model.named_modules():
            if condition(layer, name):
                replacement_module = replacement_function(layer)
                replacement_address = self._parse_model_addr(name)
                self._set_module(self.peft_model, replacement_address, replacement_module)

        return self

    def get_report(self) -> str:
        """Get report on factorized model"""

        safe_div = lambda x, y: x / y if y != 0 else 0

        # Trainable params table
        df = pd.DataFrame()
        for name, layer in self.peft_model.named_modules():
            params_dict = self.get_num_params(layer)

            df.at[name, 'name'] = name
            df.at[name, 'type'] = type(layer).__name__
            df.at[name, '# train'] = params_dict['trainable']
            df.at[name, '# fixed'] = params_dict['fixed']
            df.at[name, 'total'] = params_dict['total']
            df.at[name, '% train'] = round(safe_div(params_dict['trainable'], params_dict['total']) * 100, 2)

        # Set the 'name' column as the index
        df.set_index('name', inplace=True)

        # Return string
        return df.to_string()

    @staticmethod
    def get_num_params(model: nn.Module) -> dict:
        params_dict = {k: 0 for k in ["trainable", "fixed", "total"]}
        for p in model.parameters():
            params_dict['total'] += p.numel()
            if p.requires_grad:
                params_dict['trainable'] += p.numel()
            else:
                params_dict['fixed'] += p.numel()

        params_dict = {k: v / 1e6 for k, v in params_dict.items()}
        return params_dict

    def _get_module(self, parent: nn.Module, replacement_addr_list: list) -> nn.Module:
        """ Recursive function used to access child modules from a parent nn.Module object

        Args:
            replacement_addr_list: specifies how to access target object from ancestor object.
                ['layer1', 0, 'conv2']
        Returns:
            target object/layer to be replaced.
        """

        if len(replacement_addr_list) == 0:
            return parent
        else:
            attr = replacement_addr_list.pop(0)

            # attr can be accessible in two ways
            child = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
            return self._get_module(child, replacement_addr_list)

    def _set_module(self, model: nn.Module, replacement_addr_list: list, replacement_layer: nn.Module) -> None:
        """ Sets attribute of `model` accessed via `replacement_addr_list` to `replacement_layer` """
        if isinstance(replacement_addr_list[-1], int):
            self._get_module(model, replacement_addr_list[:-1])[replacement_addr_list[-1]] = replacement_layer
        else:
            setattr(self._get_module(model, replacement_addr_list[:-1]), replacement_addr_list[-1], replacement_layer)

    @staticmethod
    def _parse_model_addr(access_str: str) -> list:
        """ Parses path to child from a parent. E.g., layer1.0.conv2 ==> ['layer1', 0, 'conv2'] """
        parsed = access_str.split('.')
        for i in range(len(parsed)):
            try:
                parsed[i] = int(parsed[i])
            except ValueError:
                pass
        return parsed

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)
