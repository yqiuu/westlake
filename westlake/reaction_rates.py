from collections import defaultdict
from dataclasses import replace

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

    def forward(self):
        return self.rate


class FormulaDictReactionModule(nn.Module):
    def __init__(self, formula_dict, inds_fm_dict, params_reac, inds_reac):
        super(FormulaDictReactionModule, self).__init__()
        for i_fm, inds in enumerate(inds_fm_dict.values()):
            setattr(self, f"_params_reac_{i_fm}", params_reac.indexing(inds))
        self.formula_list = nn.ModuleList([formula_dict[key] for key in inds_fm_dict])
        self.register_buffer("inds_reac", inds_reac)

    def forward(self, t_in, params_med):
        batch_size = next(iter(params_med.values())).shape[0]
        def compute_rates_sub(i_fm):
            params_reac_sub = getattr(self, f"_params_reac_{i_fm}")()
            rates = self.formula_list[i_fm](params_med, params_reac_sub)
            if rates.dim() == 1:
                rates = rates.repeat(batch_size, 1)
            return rates

        return torch.concat(
            [compute_rates_sub(i_fm) for i_fm in range(len(self.formula_list))], dim=-1)

    def assign_rate_coeffs(self, coeffs, t_in, params_med):
        # out: (B, R)
        coeffs[:, self.inds_reac] = self(t_in, params_med)


class SurfaceMantleTransition(nn.Module):
    def __init__(self, inds_fm_dict, inds_reac, params_reac, meta_params):
        super().__init__()
        self._params_reac_m2s = params_reac.indexing(inds_fm_dict["mantle to surface"])
        self._params_reac_s2m = params_reac.indexing(inds_fm_dict["surface to mantle"])
        self.register_buffer("inds_reac", inds_reac)
        self.register_buffer("layer_factor",
            torch.tensor(1./(meta_params.dtg_num_ratio_0*meta_params.num_sites_per_grain)))
        self.register_buffer("alpha_gain", self.layer_factor/meta_params.num_active_layers)

    def forward(self, params_med, y_in, inds_mant, y_surf, y_mant, dy_surf_gain, dy_surf_loss):
        n_layer_mant = y_mant*self.layer_factor
        k_swap_mant = params_med["rate_hopping"]/n_layer_mant.clamp_min(1.)
        rates_m2s = k_swap_mant[:, self._params_reac_m2s()["inds_r"][:, 0]] \
            + dy_surf_loss/torch.maximum(y_surf, y_mant)

        # inds_mant must be bool indices
        k_swap_surf = k_swap_mant*y_in*inds_mant.type(y_in.dtype)/y_surf
        k_swap_surf = torch.sum(k_swap_surf, dim=-1, keepdim=True)
        rates_s2m = dy_surf_gain*self.alpha_gain + k_swap_surf
        rates_s2m = rates_s2m.repeat(1, rates_m2s.shape[1])
        return torch.concat([rates_m2s, rates_s2m], dim=-1)

    def assign_rate_coeffs(self, coeffs, params_med, y_in, inds_mant,
                           y_surf, y_mant, dy_surf_gain, dy_surf_loss):
        coeffs[:, self.inds_reac] = self(
            params_med, y_in, inds_mant, y_surf, y_mant,
            dy_surf_gain, dy_surf_loss
        )


def create_formula_dict_reaction_module(df_reac, df_spec, formula_dict, param_names):
    inds_fm_dict = defaultdict(list)
    for i_fm, fm in enumerate(df_reac["formula"]):
        inds_fm_dict[fm].append(i_fm)
    inds_reac = np.asarray(sum(inds_fm_dict.values(), start=[]))
    inds_reac = torch.as_tensor(inds_reac)
    params_reac = prepare_params_reac(df_reac, df_spec, param_names)
    return FormulaDictReactionModule(formula_dict, inds_fm_dict, params_reac, inds_reac)


def create_surface_mantle_transition(df_reac, df_spec, param_names, meta_params):
    inds_fm_dict = defaultdict(list)
    for i_fm, fm in enumerate(df_reac["formula"]):
        if fm == "mantle to surface":
            inds_fm_dict[fm].append(i_fm)
    for i_fm, fm in enumerate(df_reac["formula"]):
        if fm == "surface to mantle":
            inds_fm_dict[fm].append(i_fm)
    inds_reac = np.asarray(sum(inds_fm_dict.values(), start=[]))
    inds_reac = torch.as_tensor(inds_reac)
    params_reac = prepare_params_reac(df_reac, df_spec, param_names)
    return SurfaceMantleTransition(inds_fm_dict, inds_reac, params_reac, meta_params)


def prepare_formula_dict_indices(df_sub, rmat, inds_id_uni):
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
    inds_reac = lookup_sub.loc[inds_id_uni, "index_fm"].values
    inds_k = lookup_sub.loc[rmat.inds_k, "index_fm"].values
    return inds_fm_dict, inds_reac, inds_k


def reindex(rmat, inds_id_fm):
    lookup = pd.DataFrame(np.arange(len(inds_id_fm)), index=inds_id_fm, columns=["index_fm"])
    inds_id = lookup.loc[rmat.inds_id_uni, "index_fm"].values
    inds_k = lookup.loc[rmat.inds_k, "index_fm"].values
    return inds_id, inds_k


def prepare_params_reac(df_sub, df_spec, param_names):
    params_reac = data_frame_to_tensor_dict(df_sub[param_names])
    # inds_r is used to extract specie propeties.
    inds_r = df_spec.index.get_indexer(df_sub["reactant_1"])
    inds_r = np.vstack([inds_r, np.zeros_like(inds_r)])
    cond = (df_sub["reactant_2"] != "").values
    inds_r[1, cond] = df_spec.index.get_indexer(df_sub.loc[cond, "reactant_2"].values)
    inds_r = torch.tensor(inds_r.T)
    params_reac.add("inds_r", inds_r)
    return params_reac