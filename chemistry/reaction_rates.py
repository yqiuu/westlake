import torch
from torch import nn
from torch.nn import functional as F

from .utils import KeyTensor, Constant


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
        if isinstance(module_env, KeyTensor):
            self.module_env = Constant(module_env)
        elif isinstance(module_env, nn.Module):
            self.module_env = module_env
        else:
            raise ValueError("Unknown 'module_env'.")
        params_reac.register_buffer(self, "params_reac")

    def forward(self, t_in):
        # TODO: check if we need more efficient masks.
        rate = torch.zeros(
            (t_in.shape[0], *self.mask.shape), dtype=self.mask.dtype, device=self.mask.device)
        for i_f, formula in enumerate(self.formula_list):
            rate[..., i_f] = formula(self.module_env(t_in), self.params_reac)
        rate = torch.sum(rate*self.mask, dim=-1)
        return rate