import math

import numpy as np
import torch
from torch import nn

from .constants import M_ATOM, K_B, FACTOR_VIB_FREQ


class ThermalEvaporation(nn.Module):
    def __init__(self, meta_params):
        super(ThermalEvaporation, self).__init__()
        self.register_buffer("T_dust_0", torch.tensor(meta_params.T_dust_0))

    def forward(self, params_env, params_reac, T_gas, mask_T, E_d, freq_vib, **kwargs):
        rate = compute_evaporation_rate(params_reac["alpha"], E_d, freq_vib, self.T_dust_0)
        return rate


class CosmicRayEvaporation(nn.Module):
    def __init__(self, meta_params):
        super().__init__()
        prefactor = (meta_params.rate_cr_ion + meta_params.rate_x_ion)/1.3e-17 \
            *(meta_params.rate_fe_ion*meta_params.tau_cr_peak)
        self.register_buffer("prefactor", torch.tensor(prefactor))
        self.register_buffer("T_grain_cr_peak", torch.tensor(meta_params.T_grain_cr_peak))

    def forward(self, params_env, params_reac, T_gas, mask_T, E_d, freq_vib, **kwargs):
        prefactor = self.prefactor*params_reac["alpha"]
        rate = compute_evaporation_rate(prefactor, E_d, freq_vib, self.T_grain_cr_peak)
        return rate


class SurfaceAccretion(nn.Module):
    def __init__(self, meta_params):
        super(SurfaceAccretion, self).__init__()
        self.register_buffer("dtg_num_ratio_0", torch.tensor(meta_params.dtg_num_ratio_0))

    def forward(self, params_env, params_reac, T_gas, mask_T, factor_rate_acc, **kwargs):
        factor = factor_rate_acc*params_reac["alpha"]
        return factor*params_env["T_gas"].sqrt()*params_env["den_H"]*self.dtg_num_ratio_0


class DummyZero(nn.Module):
    def forward(self, params_env, params_reac, *args, **kwargs):
        return torch.zeros_like(params_reac["alpha"])


def compute_evaporation_rate(factor, E_d, freq_vib, T_dust):
    return factor*freq_vib*torch.exp(-E_d/T_dust)


def compute_thermal_hoping_rate(E_barr, freq_vib, T_dust, num_sites_per_grain):
    return freq_vib*torch.exp(-E_barr/T_dust)/num_sites_per_grain


def compute_surface_params(spec_table, meta_params, special_dict=None):
    """Compute surface parameters.

    This is an inplace operation.

    Args:
        spec_table (pd.DataFrame): Specie table.
        meta_params (MetaPrameters): Meta parameters.
        special_dict (dict, optional): Defaults to None.
    """
    compute_vibration_frequency(spec_table, meta_params)
    compute_factor_rate_acc(spec_table, meta_params, special_dict)


def compute_vibration_frequency(spec_table, meta_params):
    spec_table["freq_vib"] \
        = np.sqrt(FACTOR_VIB_FREQ*meta_params.site_density*spec_table["E_deso"]/spec_table["ma"])


def compute_factor_rate_acc(spec_table, meta_params, special_dict=None):
    charge = spec_table["charge"].values
    sticking_coeff = np.zeros_like(spec_table["ma"].values)
    sticking_coeff[charge == 0] = meta_params.sticking_coeff_neutral
    sticking_coeff[charge > 0] = meta_params.sticking_coeff_positive
    sticking_coeff[charge < -1] = meta_params.sticking_coeff_negative
    if special_dict is not None:
        for key, val in spec_table.items():
            sticking_coeff[spec_table == key] = val

    grain_radius = meta_params.grain_radius
    factor = math.pi*grain_radius*grain_radius*math.sqrt(8.*K_B/math.pi/M_ATOM)
    spec_table["factor_rate_acc"] = factor*sticking_coeff/np.sqrt(spec_table["ma"])

    cond = spec_table.index.map(lambda name: name.startswith("J"))
    inds = spec_table[cond].index.map(lambda name: name[1:])
    spec_table.loc[inds, "factor_rate_acc"] = spec_table.loc[cond, "factor_rate_acc"].values


def assign_surface_params(df_reac, spec_table):
    """Assigin surface parameters.

    This is an inplace operation.

    Args:
        df_reac (pd.DataFrame): Data frame of reactions.
        spec_table (pd.DataFrame): Specie table.
    """
    columns = ["E_deso", "E_barr", "freq_vib", "factor_rate_acc"]
    for col in columns:
        df_reac[f"{col}_r1"] = spec_table.loc[df_reac["reactant_1"], col].values

    columns = ["E_barr", "freq_vib"]
    df_tmp = df_reac.loc[df_reac["reactant_2"] != "", "reactant_2"]
    for col in columns:
        df_reac.loc[df_tmp.index, f"{col}_r2"] = spec_table.loc[df_tmp, col].values
    df_reac.fillna(0., inplace=True)