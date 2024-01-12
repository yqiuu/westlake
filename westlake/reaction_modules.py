import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from .utils import data_frame_to_tensor_dict


class FormulaDictReactionModule(nn.Module):
    def __init__(self, formula_dict, inds_fm_dict, params_reac, inds_reac):
        super(FormulaDictReactionModule, self).__init__()
        for i_fm, inds in enumerate(inds_fm_dict.values()):
            setattr(self, f"_params_reac_{i_fm}", params_reac.indexing(inds))
        self.formula_list = nn.ModuleList([formula_dict[key] for key in inds_fm_dict])
        self.register_buffer("inds_reac", inds_reac)

    def forward(self, params_med, y_in=None, params_extra=None, coeffs=None):
        batch_size = next(iter(params_med.values())).shape[0]
        def compute_rates_sub(i_fm):
            params_reac_sub = getattr(self, f"_params_reac_{i_fm}")()
            if y_in is None:
                rates = self.formula_list[i_fm](params_med, params_reac_sub)
            else:
                rates = self.formula_list[i_fm](
                    params_med, params_reac_sub,
                    y_in=y_in, coeffs=coeffs, params_extra=params_extra
                )
            if rates.dim() == 1:
                rates = rates.repeat(batch_size, 1)
            return rates

        return torch.concat(
            [compute_rates_sub(i_fm) for i_fm in range(len(self.formula_list))], dim=-1)

    def assign_rate_coeffs(self, coeffs, params_med, y_in=None, params_extra=None):
        # out: (B, R)
        coeffs[:, self.inds_reac] = self(params_med, y_in, params_extra, coeffs)


class SurfaceMantleTransition(nn.Module):
    def __init__(self, inds_fm_dict, inds_reac, params_reac, config):
        super().__init__()
        self._params_reac_m2s = params_reac.indexing(inds_fm_dict["mantle to surface"])
        self._params_reac_s2m = params_reac.indexing(inds_fm_dict["surface to mantle"])
        self.register_buffer("inds_reac", inds_reac)
        self.register_buffer("layer_factor", torch.tensor(config.layer_factor))
        self.register_buffer("alpha_gain", self.layer_factor/config.num_active_layers)

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


class ConstantRateModule(nn.Module):
    """Constant rate module.

    Args:
        coeffs: (R,)
        inds_reac: (R,) or (N,)
    """
    def __init__(self, coeffs, inds_reac):
        super().__init__()
        self.register_buffer("coeffs", coeffs[inds_reac])
        self.register_buffer("inds_reac", inds_reac)

    def assign_rate_coeffs(self, coeffs, params_med):
        coeffs[:, self.inds_reac] = self.coeffs


def create_formula_dict_reaction_module(df_reac, df_spec, formula_dict, formula_dict_ex):
    check_reaction_formulae(df_reac, formula_dict, formula_dict_ex)

    inds_fm_dict_all = defaultdict(list)
    for i_fm, fm in enumerate(df_reac["formula"]):
        inds_fm_dict_all[fm].append(i_fm)
    inds_fm_dict = {}
    inds_fm_dict_ex = {}
    for fm in list(inds_fm_dict_all.keys()):
        if fm in formula_dict:
            inds_fm_dict[fm] = inds_fm_dict_all.pop(fm)
        elif fm in formula_dict_ex:
            inds_fm_dict_ex[fm] = inds_fm_dict_all.pop(fm)
    if len(inds_fm_dict_all) > 0:
        reac_str = ", ".join(inds_fm_dict_all.keys())
        raise ValueError(f"Unknown reactions: {reac_str}.")

    param_names = {}
    for fm in formula_dict.values():
        for prop in fm.required_props:
            param_names[prop] = 0
    for fm in formula_dict_ex.values():
        for prop in fm.required_props:
            param_names[prop] = 0
    param_names = list(param_names.keys())
    params_reac = prepare_params_reac(df_reac, df_spec, param_names)
    #
    inds_reac = np.asarray(sum(inds_fm_dict.values(), start=[]))
    inds_reac = torch.as_tensor(inds_reac)
    rmod = FormulaDictReactionModule(formula_dict, inds_fm_dict, params_reac, inds_reac)
    #
    if len(inds_fm_dict_ex) > 0:
        inds_reac = np.asarray(sum(inds_fm_dict_ex.values(), start=[]))
        inds_reac = torch.as_tensor(inds_reac)
        rmod_ex = FormulaDictReactionModule(
            formula_dict_ex, inds_fm_dict_ex, params_reac, inds_reac
        )
    else:
        rmod_ex = None
    #
    return rmod, rmod_ex


def check_reaction_formulae(df_reac, formula_dict, formula_dict_ex):
    fm_set_1 = set(df_reac["formula"])
    fm_set_2 = set(formula_dict.keys())
    fm_set_2.update(formula_dict_ex.keys())
    missing = fm_set_1 - fm_set_2
    if len(missing) != 0:
        cond = np.full(len(df_reac), False)
        for fm in missing:
            cond |= df_reac["formula"] == fm
        print(df_reac.loc[cond, ["key", "formula"]].to_string(), file=sys.stderr)
        raise ValueError("Undefined reaction formulae")


def create_surface_mantle_transition(df_reac, df_spec, config):
    inds_fm_dict = defaultdict(list)
    for i_fm, fm in enumerate(df_reac["formula"]):
        if fm == "mantle to surface":
            inds_fm_dict[fm].append(i_fm)
    for i_fm, fm in enumerate(df_reac["formula"]):
        if fm == "surface to mantle":
            inds_fm_dict[fm].append(i_fm)
    inds_reac = np.asarray(sum(inds_fm_dict.values(), start=[]))
    inds_reac = torch.as_tensor(inds_reac)
    params_reac = prepare_params_reac(df_reac, df_spec, [])
    return SurfaceMantleTransition(inds_fm_dict, inds_reac, params_reac, config)


def prepare_params_reac(df_sub, df_spec, param_names):
    params_reac = data_frame_to_tensor_dict(df_sub[param_names])
    # inds_r is used to extract specie propeties.
    inds_r = df_spec.index.get_indexer(df_sub["reactant_1"])
    inds_r = np.vstack([inds_r, np.zeros_like(inds_r)]) # (2, N)
    cond = (df_sub["reactant_2"] != "").values
    inds_r[1, cond] = df_spec.index.get_indexer(df_sub.loc[cond, "reactant_2"].values)
    #
    is_mant = df_spec.index.map(lambda name: name.startswith("K")).values
    is_mant = is_mant[inds_r[0]]

    inds_r = torch.tensor(inds_r.T) # (N, 2)
    params_reac.add("inds_r", inds_r)
    is_mant = torch.tensor(is_mant)
    params_reac.add("is_mant_r1", is_mant)
    return params_reac