import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from astropy import units, constants

from .constants import M_ATOM, K_B, H_BAR, FACTOR_VIB_FREQ


def builtin_surface_reactions_1st(meta_params):
    return {
        'thermal evaporation': ThermalEvaporation(meta_params),
        'CR evaporation': CosmicRayEvaporation(meta_params),
        'complexes reaction': NoReaction(),
        'UV photodesorption': NoReaction(),
        'CR photodesorption': NoReaction(),
        'surface accretion': SurfaceAccretion(meta_params),
    }


def builtin_surface_reactions_2nd(meta_params):
    return {
        'surface reaction': SurfaceReaction(meta_params),
        'Eley-Rideal': NoReaction(),
    }


class ThermalEvaporation(nn.Module):
    def __init__(self, meta_params):
        super(ThermalEvaporation, self).__init__()

    def forward(self, params_env, params_reac, **params_extra):
        return compute_evaporation_rate(
            params_reac["alpha"], params_reac["freq_vib_r1"],
            params_reac["E_deso_r1"], params_env["T_dust"]
        )


class CosmicRayEvaporation(nn.Module):
    def __init__(self, meta_params):
        super(CosmicRayEvaporation, self).__init__()
        prefactor = (meta_params.rate_cr_ion + meta_params.rate_x_ion)/1.3e-17 \
            *(meta_params.rate_fe_ion*meta_params.tau_cr_peak)
        self.register_buffer("prefactor", torch.tensor(prefactor))
        self.register_buffer("T_grain_cr_peak", torch.tensor(meta_params.T_grain_cr_peak))

    def forward(self, params_env, params_reac, **params_extra):
        return compute_evaporation_rate(
            self.prefactor*params_reac["alpha"], params_reac["freq_vib_r1"],
            params_reac["E_deso_r1"], self.T_grain_cr_peak
        )


class SurfaceAccretion(nn.Module):
    def __init__(self, meta_params):
        super(SurfaceAccretion, self).__init__()
        self.register_buffer("dtg_num_ratio_0", torch.tensor(meta_params.dtg_num_ratio_0))

    def forward(self, params_env, params_reac, **params_extra):
        return params_reac["alpha"]*params_reac["factor_rate_acc_r1"] \
            *(params_env["T_gas"].sqrt()*params_env["den_H"]*self.dtg_num_ratio_0)


class SurfaceReaction(nn.Module):
    def __init__(self, meta_params):
        super(SurfaceReaction, self).__init__()
        self.register_buffer("num_sites_per_grain", torch.tensor(meta_params.num_sites_per_grain))
        self.register_buffer("inv_dtg_num_ratio_0", torch.tensor(1./meta_params.dtg_num_ratio_0))

    def forward(self, params_env, params_reac, **params_extra):
        rate_diff_r1 = compute_thermal_hoping_rate(
            params_reac["E_barr_r1"], params_reac["freq_vib_r1"],
            params_env["T_dust"], self.num_sites_per_grain
        )
        rate_diff_r2 = compute_thermal_hoping_rate(
            params_reac["E_barr_r2"], params_reac["freq_vib_r2"],
            params_env["T_dust"], self.num_sites_per_grain
        )
        barr = 1.
        return params_reac["alpha"]*params_reac["branching_ratio"]*barr\
            *(rate_diff_r1 + rate_diff_r2)*self.inv_dtg_num_ratio_0/params_env["den_H"]


class NoReaction(nn.Module):
    def forward(self, params_env, params_reac, **params_extra):
        return torch.zeros_like(params_reac["alpha"])


def compute_evaporation_rate(factor, freq_vib, E_d, T_dust):
    return factor*freq_vib*torch.exp(-E_d/T_dust)


def compute_thermal_hoping_rate(E_barr, freq_vib, T_dust, num_sites_per_grain):
    return freq_vib*torch.exp(-E_barr/T_dust)/num_sites_per_grain


def prepare_surface_reaction_params(df_reac, df_surf, df_act, spec_table, meta_params,
                                    use_builtin_spec_params=True, specials_barr=None):
    """Prepare surface reaction parameters.

    Assign the surface reaction parameters to the input reaction dataframe. This
    is an inplace operation.

    Args:
        df_reac (pd.DataFrame):
        df_surf (pd.DataFrame):
        df_act (pd.DataFrame):
        spec_table (pd.DataFrame):
        meta_params (MetaPrameters):
        use_builtin_spec_params (bool, optional): Defaults to True.
        specials_barr (dict | None, optional): Defaults to None.
    """
    if use_builtin_spec_params:
        df_surf = prepare_surface_specie_params(df_surf, spec_table, meta_params, specials_barr)
    assign_surface_params(df_reac, df_surf)
    assign_activation_energy(df_reac, df_act)
    compute_branching_ratio(df_reac, df_surf, meta_params)


def prepare_surface_specie_params(df_surf, spec_table, meta_params, specials_barr=None):
    """Prepare surface specie parameters.

    Args:
        df_surf_ret (pd.DataFrame): Surface parameters.
        spec_table (pd.DataFrame): Specie table.
        meta_params (MetaPrameters): Meta parameters.
        special_dict (dict, optional): Defaults to None.
    """
    df_surf_ret = spec_table[["charge", "num_atoms", "ma"]].copy()
    df_surf_ret = df_surf_ret.join(df_surf)
    #df_surf_ret.fillna(0., inplace=True)
    assign_columns_to_normal_counterparts(df_surf_ret, ["E_deso", "dHf"])
    compute_vibration_frequency(df_surf_ret, meta_params)
    compute_factor_rate_acc(df_surf_ret, meta_params)
    compute_barrier_energy(df_surf_ret, meta_params, specials_barr)
    compute_rate_tunneling(df_surf_ret, meta_params)
    return df_surf_ret


def compute_vibration_frequency(spec_table, meta_params):
    spec_table["freq_vib"] \
        = np.sqrt(FACTOR_VIB_FREQ*meta_params.site_density*spec_table["E_deso"]/spec_table["ma"])


def compute_factor_rate_acc(spec_table, meta_params):
    """Compute the factor to calculate surface accretion rates.

    For surface accretion, the product is a surface specie, while the reactant
    is a normal specie. In practice, we need to set the factor to the
    corresponding normal specie.

    Args:
        spec_table (pd.Dataframe): Specie table.
        meta_params (MetaParams): Meta parameters.
        special_dict (dict | None, optional): A dict to specify special cases.
        Defaults to None.
    """
    charge = spec_table["charge"].values
    sticking_coeff = np.zeros_like(spec_table["ma"].values)
    sticking_coeff[charge == 0] = meta_params.sticking_coeff_neutral
    sticking_coeff[charge > 0] = meta_params.sticking_coeff_positive
    sticking_coeff[charge < -1] = meta_params.sticking_coeff_negative

    grain_radius = meta_params.grain_radius
    factor = math.pi*grain_radius*grain_radius*math.sqrt(8.*K_B/math.pi/M_ATOM)
    spec_table["factor_rate_acc"] = factor*sticking_coeff/np.sqrt(spec_table["ma"])
    assign_columns_to_normal_counterparts(spec_table, "factor_rate_acc")


def compute_barrier_energy(spec_table, meta_params, specials=None):
    spec_table["E_barr"] = meta_params.surf_diff_to_deso_ratio*spec_table["E_deso"]
    if specials is not None:
        for key, val in specials.items():
            spec_table.loc[key, "E_barr"] = val


def compute_rate_tunneling(spec_table, meta_params):
    spec_table["rate_tunneling_a"] = spec_table["dE_band"]*(.25*K_B/H_BAR)

    exponent = -2.*meta_params.diffusion_barrier_thickness/H_BAR \
        *np.sqrt(2*K_B*M_ATOM**spec_table["ma"])
    spec_table["rate_tunneling_b"] = spec_table["freq_vib"]*np.exp(exponent)


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


def assign_activation_energy(df_reac, df_act):
    index = np.intersect1d(df_reac["key"], df_act.index)
    df_tmp = pd.DataFrame(df_reac.index, index=df_reac["key"], columns=["index"])
    df_reac["E_act"] = 0.
    df_reac.loc[df_tmp.loc[index, "index"], "E_act"] = df_act.loc[index, "E_act"].values


def compute_branching_ratio(df_reac, spec_table, meta_params):
    u_dHf = units.imperial.kilocal.cgs.scale/K_B/constants.N_A.value # Convert kilocal/mol to K
    cond = df_reac["formula"] == 'surface reaction'
    df_tmp = df_reac.loc[cond, ["reactant_1", "reactant_2", "products", "E_act"]].copy()

    inds_dict = defaultdict(list)
    counts_dict = defaultdict(int)
    for idx, reac_1, reac_2, prods in zip(
        df_tmp.index.values,
        df_tmp["reactant_1"].values,
        df_tmp["reactant_2"].values,
        df_tmp["products"].values,
    ):
        reacs = f"{reac_1};{reac_2}"
        inds_dict[reacs].append(idx)
        if prods.startswith("J"):
            counts_dict[reacs] += 1
    branching_ratios = []
    for inds, n_reac in zip(inds_dict.values(), counts_dict.values()):
        if n_reac == 0:
            raise ValueError("A surface reaction has no surface spiece products.")
        branching_ratios.extend(len(inds)*[1./n_reac])
    df_tmp["branching_ratio"] = 1.
    inds = sum(inds_dict.values(), start=[])
    df_tmp.loc[inds, "branching_ratio"] = branching_ratios
    df_tmp.loc[df_tmp["reactant_1"] == df_tmp["reactant_2"], "branching_ratio"] *= .5

    prod_first = [] # List of the first product. TODO: Check the order

    # TODO: Understanding
    lookup_E_deso = spec_table["E_deso"].to_dict()
    E_deso_max = np.zeros(len(df_tmp))
    lookup_dHf = spec_table["dHf"].to_dict()
    dHf_sum = spec_table.loc[df_tmp["reactant_1"], "dHf"].values \
        + spec_table.loc[df_tmp["reactant_2"], "dHf"].values
    for i_reac, prods in enumerate(df_tmp["products"].values):
        prods = prods.split(";")
        prod_first.append(prods[0])
        if len(prods) == 1:
            for name in prods:
                dHf_sum[i_reac] -= lookup_dHf[name]
        else:
            # We will set the branching ratio to be 0 for dHf_sum < 0.
            dHf_sum[i_reac] = -1.
        E_deso_max[i_reac] = max([lookup_E_deso[key] for key in prods])
    dHf_sum *= u_dHf # To K
    dHf_sum -= df_tmp["E_act"].values

    num_atoms = spec_table.loc[prod_first, "num_atoms"].values
    frac_deso = np.zeros_like(E_deso_max)
    cond = dHf_sum != 0.
    frac_deso[cond] = 1. - E_deso_max[cond]/dHf_sum[cond]
    cond = num_atoms == 2
    frac_deso[cond] = frac_deso[cond]**(3*num_atoms[cond] - 5)
    cond = num_atoms > 2
    frac_deso[cond] = frac_deso[cond]**(3*num_atoms[cond] - 6)
    frac_deso *= meta_params.vib_to_dissip_freq_ratio
    frac_deso = frac_deso/(1. + frac_deso)

    frac_deso[(dHf_sum <= 0.) | np.isnan(dHf_sum)] = 0.
    frac_deso[frac_deso < 0.] = 0.
    frac_deso[frac_deso > 1.] = 1.

    cond = list(map(lambda name: name.startswith("J"), prod_first))

    frac_deso[cond] = 1 - frac_deso[cond]
    df_tmp["branching_ratio"] *= frac_deso
    df_reac["branching_ratio"] = 1.
    df_reac.loc[df_tmp.index, "branching_ratio"] = df_tmp["branching_ratio"].values


def assign_columns_to_normal_counterparts(df_surf, columns):
    """Assign properties from the surface species to their normal counterparts.

    This is an inplace operation.

    Args:
        df_surf (pd.DataFrame): Surface parameter dataframe.
        columns (str | list): Property names.
    """
    cond = df_surf.index.map(lambda name: name.startswith("J")).values \
        & np.isin(df_surf.index.map(lambda name: name[1:]).values, df_surf.index)
    spec = df_surf[cond].index.map(lambda name: name[1:])
    df_surf.loc[spec, columns] = df_surf.loc[cond, columns].values