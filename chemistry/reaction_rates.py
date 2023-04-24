import torch
from torch import nn
from torch.nn import functional as F

from .utils import TensorDict


class FixedReactionRate(nn.Module):
    def __init__(self, rmat, rate):
        super(FixedReactionRate, self).__init__()
        rate = rmat.rate_sign*rate[rmat.inds]
        self.register_buffer("rate", torch.tensor(rate, dtype=torch.float32))

    def forward(self, t_in):
        return self.rate


class FormulaDictReactionRate(nn.Module):
    def __init__(self, formula_dict, formula, rmat, module_env, params_reac):
        super(FormulaDictReactionRate, self).__init__()

        lookup = {key: idx for idx, key in enumerate(formula_dict.keys())}
        mask = F.one_hot(torch.tensor([lookup[name] for name in formula]), len(lookup))
        mask *= rmat.rate_sign[:, None]
        self.register_buffer("mask", mask.type(torch.float32)) # (R, F)
        self.formula_list = nn.ModuleList(formula_dict.values())
        if isinstance(module_env, TensorDict) or isinstance(module_env, nn.Module):
            self.module_env = module_env
        else:
            raise ValueError("Unknown 'module_env'.")
        self.params_reac = params_reac

    def forward(self, t_in):
        params_env = self.module_env(t_in)
        params_reac = self.params_reac()
        rate = [formula(params_env, params_reac) for formula in self.formula_list]
        rate = torch.vstack(rate).T
        rate = torch.sum(rate*self.mask, dim=-1)
        return rate