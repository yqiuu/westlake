"""Medium modules are used to compute any parameters that evolve with time.

Args:
    t_in (tensor): Time. (B, 1)
    params_med (dict): Medium parameters. (B, X) for each element.

Returns:
    dict: Medium parameters. This should include input parameters.
"""
import torch
from torch import nn

from .utils import LinearInterpolation


class SequentialMedium(nn.ModuleList):
    def __init__(self, *args):
        super(SequentialMedium, self).__init__(args)

    def forward(self, t_in, params_med=None):
        for module in self:
            params_med = module(t_in, params_med)
        return params_med


class StaticMedium(nn.Module):
    def __init__(self, params_dict):
        super(StaticMedium, self).__init__()
        params_med = torch.tensor(
            list(params_dict.values()), dtype=torch.get_default_dtype()).reshape(-1, 1, 1)
        self.register_buffer("params_med", params_med)
        self.columns = tuple(params_dict.keys())

    def forward(self, t_in=None, params_med=None):
        params = self.params_med
        return {col: params[i_col] for i_col, col in enumerate(self.columns)}


class InterpolationMedium(nn.Module):
    def __init__(self, tau, params, columns, meta_params):
        super(InterpolationMedium, self).__init__()
        self.interp = LinearInterpolation(
            torch.tensor(tau*meta_params.to_second, dtype=torch.get_default_dtype()),
            torch.tensor(params, dtype=torch.get_default_dtype()),
        )
        self.columns = columns

    def forward(self, t_in, params_med=None):
        params_p = self.interp(t_in)
        return {col: params_p[:, i_col, None] for i_col, col in enumerate(self.columns)}


class CoevolutionMedium(nn.Module):
    def __init__(self, name_new, name_co):
        super(CoevolutionMedium, self).__init__()
        self.name_new = name_new
        self.name_co = name_co

    def forward(self, t_in, params_med):
        params_med[self.name_new] = params_med[self.name_co]
        return params_med


class ThermalHoppingRate(nn.Module):
    def __init__(self, E_barr, freq_vib, meta_params):
        super().__init__()
        self.register_buffer("E_barr", torch.tensor(E_barr, dtype=torch.get_default_dtype()))
        self.register_buffer("freq_vib", torch.tensor(freq_vib, dtype=torch.get_default_dtype()))
        self.register_buffer("inv_num_sites_per_grain",
            torch.tensor(1./meta_params.num_sites_per_grain))

    def forward(self, t_in, params_med):
        params_med["rate_hopping"] = self.freq_vib \
            * torch.exp(-self.E_barr/params_med["T_dust"]) \
            * self.inv_num_sites_per_grain
        return params_med