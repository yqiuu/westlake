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


class InterpolationMedium(nn.Module):
    def __init__(self, tau, params, columns):
        super(InterpolationMedium, self).__init__()
        self.interp = LinearInterpolation(
            torch.tensor(tau, dtype=torch.get_default_dtype()),
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