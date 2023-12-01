import torch
from torch import nn


class Medium(nn.Module):
    """Medium modules are used to compute any parameters that evolve with time."""
    def __init__(self, config, Av=None, den_gas=None, T_gas=None, T_dust=None):
        super().__init__()
        self._constants = {}
        self._module_dict = nn.ModuleDict()
        key = ["Av", "den_gas", "T_gas", "T_dust"]
        values = [Av, den_gas, T_gas, T_dust]
        for key, val in zip(key, values):
            if val is None:
                self._constants[key] = getattr(config, key)
            else:
                self.add_medium_parameter(key, val)

    def add_medium_parameter(self, key, val):
        try:
            fval = float(val)
        except:
            fval = None
        if fval is not None:
            self._constants[key] = val
            return

        if isinstance(val, nn.Module):
            self._module_dict[key] = val
        else:
            raise TypeError("'val' can only be float or nn.Module.")

    def is_static(self):
        """Check if any medium parameters evolve with time."""
        return len(self._module_dict) == 0

    def forward(self, t_in):
        """
        Args:
            t_in (tensor): Time. (B, 1)

        Returns:
            dict: Medium parameters. (B, X) for each element.
        """
        params_med = {}
        for key, val in self._constants.items():
            params_med[key] = torch.full(
                (t_in.shape[0], 1), val, dtype=t_in.dtype, device=t_in.device
            )
        for key, module in self._module_dict.items():
            params_med[key] = module(t_in, params_med)
        return params_med