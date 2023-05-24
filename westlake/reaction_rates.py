from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from .utils import data_frame_to_tensor_dict


class ConstantReactionRate(nn.Module):
    def __init__(self, rmat, rate):
        super(ConstantReactionRate, self).__init__()
        rate = torch.tensor(rmat.rate_sign*rate[rmat.inds_k], dtype=torch.get_default_dtype())
        self.register_buffer("rate", rate)

    def forward(self, t_in):
        return self.rate


class FormulaDictReactionModule(nn.Module):
    def __init__(self, rmat, formula_dict, inds_fm_dict, inds_reac, inds_k, params_reac):
        super(FormulaDictReactionModule, self).__init__()
        self.order = rmat.order
        for i_fm, inds in enumerate(inds_fm_dict.values()):
            setattr(self, f"_params_reac_{i_fm}", params_reac.indexing(inds))
        self.formula_list = nn.ModuleList([formula_dict[key] for key in inds_fm_dict])
        self.register_buffer(
            "rate_sign", torch.tensor(rmat.rate_sign, dtype=torch.get_default_dtype()))
        self.register_buffer("inds_reac", torch.tensor(inds_reac))
        self.register_buffer("inds_k", torch.tensor(inds_k))

    def forward(self, t_in, params_med):
        rates = self.compute_rates(params_med)
        if self.order == 2:
            rates = rates*params_med["den_gas"]
        return rates[:, self.inds_k]*self.rate_sign

    def compute_rates_reac(self, t_in, params_med):
        # out: (B, R)
        return self.compute_rates(params_med)[:, self.inds_reac]

    def compute_rates(self, params_med):
        batch_size = next(iter(params_med.values())).shape[0]
        def compute_rates_sub(i_fm):
            params_reac_sub = getattr(self, f"_params_reac_{i_fm}")()
            rates = self.formula_list[i_fm](params_med, params_reac_sub)
            if rates.dim() == 1:
                rates = rates.repeat(batch_size, 1)
            return rates

        return torch.concat(
            [compute_rates_sub(i_fm) for i_fm in range(len(self.formula_list))], dim=-1)


def create_formula_dict_reaction_module(df_reac, rmat, formula_dict, param_names):
    df_sub = df_reac.iloc[rmat.inds_id]

    # The code below construct the following variables.
    #   1. inds_reac, index in the reaction dataframe for the outputs of the reaction module.
    #   2. inds_k, index of the rates in the equation for the outputs of the reaction module. This
    #   should align with rmat.rate_sign

    # Link the indices in the df_sub to those in the reaction dataframe.
    lookup_sub = pd.DataFrame(np.arange(len(df_sub)), index=df_sub.index, columns=["index_sub"])
    # Link the indices in the outputs of the reaction module to those in the df_sub.
    inds_fm_dict = defaultdict(list)
    for i_fm, fm in enumerate(df_sub["formula"]):
        inds_fm_dict[fm].append(i_fm)
    inds_fm = np.asarray(sum(inds_fm_dict.values(), start=[]))
    lookup_fm = pd.DataFrame(np.arange(len(inds_fm)), index=inds_fm, columns=["index_fm"])
    # Link the indices in the outputs of the reaction module to those in the reaction dataframe.
    lookup_sub["index_fm"] = lookup_fm.loc[lookup_sub["index_sub"], "index_fm"].values
    #
    inds_reac = lookup_sub.loc[rmat.inds_id, "index_fm"].values
    inds_k = lookup_sub.loc[rmat.inds_k, "index_fm"].values

    return FormulaDictReactionModule(
        rmat, formula_dict, inds_fm_dict, inds_reac, inds_k,
        data_frame_to_tensor_dict(df_sub[param_names])
    )
