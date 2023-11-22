import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from astropy import units, constants

from .reaction_rates import ReactionRate, NoReaction
from ..constants import M_ATOM, K_B, H_BAR, FACTOR_VIB_FREQ


def builtin_surface_reactions(config):
    return {
        'thermal evaporation': ThermalEvaporation(config),
        'CR evaporation': CosmicRayEvaporation(config),
        'complexes reaction': NoReaction(),
        'UV photodesorption': UVPhotodesorption(config),
        'CR photodesorption': CRPhotodesorption(config),
        'surface accretion': SurfaceAccretion(config),
        "surface to mantle": NoReaction(),
        "mantle to surface": NoReaction(),
        'surface reaction': SurfaceReaction(config),
        'Eley-Rideal': NoReaction(),

    }


class ThermalEvaporation(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "freq_vib_r1", "E_deso_r1"])

    def forward(self, params_env, params_reac, **params_extra):
        return compute_evaporation_rate(
            params_reac["alpha"], params_reac["freq_vib_r1"],
            params_reac["E_deso_r1"], params_env["T_dust"]
        )


class CosmicRayEvaporation(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "freq_vib_r1", "E_deso_r1"])
        prefactor = (config.rate_cr_ion + config.rate_x_ion)/1.3e-17 \
            *(config.rate_fe_ion*config.tau_cr_peak)
        self.register_buffer("prefactor", torch.tensor(prefactor))
        self.register_buffer("T_grain_cr_peak", torch.tensor(config.T_grain_cr_peak))

    def forward(self, params_env, params_reac, **params_extra):
        return compute_evaporation_rate(
            self.prefactor*params_reac["alpha"], params_reac["freq_vib_r1"],
            params_reac["E_deso_r1"], self.T_grain_cr_peak
        )


class UVPhotodesorption(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha"])
        self.register_buffer("prefactor",
            torch.tensor(1e8/config.site_density*config.uv_flux))

    def forward(self, params_env, params_reac, decay_factor=None, **params_extra):
        rate = self.prefactor*params_reac["alpha"]*torch.exp(-2.*params_env["Av"])
        if decay_factor is not None:
            rate = rate*decay_factor
        return rate


class CRPhotodesorption(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha"])
        self.register_buffer("prefactor",
            torch.tensor(1e4/config.site_density*1.3e-17/config.rate_cr_ion)
        )

    def forward(self, params_env, params_reac, decay_factor=None, **params_extra):
        rate = self.prefactor*params_reac["alpha"]
        if decay_factor is not None:
            rate = rate*decay_factor
        return rate


class SurfaceAccretion(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "factor_rate_acc_r1"])
        self.register_buffer("dtg_num_ratio_0", torch.tensor(config.dtg_num_ratio_0))

    def forward(self, params_env, params_reac, **params_extra):
        return params_reac["alpha"]*params_reac["factor_rate_acc_r1"] \
            *(params_env["T_gas"].sqrt()*params_env["den_gas"]*self.dtg_num_ratio_0)


class SurfaceReaction(ReactionRate):
    def __init__(self, config):
        super().__init__(["alpha", "branching_ratio", "E_act", "log_prob_surf_tunl"])
        self.register_buffer("inv_dtg_num_ratio_0", torch.tensor(1./config.dtg_num_ratio_0))

    def forward(self, params_med, params_reac, **params_extra):
        rate_hopping = params_med["rate_hopping"][:, params_reac["inds_r"]].sum(dim=-1)
        log_prob = -params_reac["E_act"]/params_med["T_dust"] # (B, R)
        log_prob = torch.maximum(log_prob, params_reac["log_prob_surf_tunl"].unsqueeze(0))
        prob = log_prob.exp()
        return params_reac["alpha"]*params_reac["branching_ratio"]/params_med["den_gas"] \
            *self.inv_dtg_num_ratio_0*rate_hopping*prob


def compute_evaporation_rate(factor, freq_vib, E_d, T_dust):
    return factor*freq_vib*torch.exp(-E_d/T_dust)


def prepare_surface_reaction_params(df_reac, df_spec, df_surf, config,
                                    df_act=None, df_br=None,
                                    df_barr=None, df_gap=None, df_ma=None):
    """Prepare surface reaction parameters.

    Assign the surface reaction parameters to the input reaction dataframe. This
    is an inplace operation.

    Args:
        df_reac (pd.DataFrame):
        df_spec (pd.DataFrame):
        df_surf (pd.DataFrame):
        config (Config): Config.
        df_act (pd.DataFrame, optional):
        df_br (pd.DataFrame, optional): Additonal Branching ratio.
        df_barr (pd.DataFrame, optional): Defaults to None.
        df_gap (pd.DataFrame, optional): Gap energy [K].
    """
    df_surf = prepare_surface_specie_params(
        df_surf, df_spec, config,
        df_barr=df_barr, df_gap=df_gap, df_ma=df_ma
    )
    assign_surface_params(df_reac, df_spec, df_surf)
    assign_activation_energy(df_reac, df_act)
    compute_branching_ratio(df_reac, df_surf, config)
    assign_branching_ratio(df_reac, df_br)
    assign_surface_tunneling_probability(df_reac, df_surf, config)


def prepare_surface_specie_params(df_surf, spec_table, config, df_barr, df_gap, df_ma):
    """Prepare surface specie parameters.

    Args:
        df_surf_ret (pd.DataFrame): Surface parameters.
        spec_table (pd.DataFrame): Specie table.
        config (Config): Config.
    """
    df_surf_ret = spec_table[["charge", "num_atoms", "ma"]].copy()
    df_surf_ret = df_surf_ret.join(df_surf[["E_deso", "dHf"]])
    assign_columns_to_normal_counterparts(df_surf_ret, ["E_deso", "dHf"])
    assign_vibration_frequency(df_surf_ret, config, df_ma)
    compute_factor_rate_acc(df_surf_ret, config)
    compute_barrier_energy(df_surf_ret, config, df_barr)
    compute_rate_tunneling(df_surf_ret, df_gap, config)
    return df_surf_ret


def assign_vibration_frequency(df_surf, config, df_ma=None):
    df_surf["freq_vib"] = compute_vibration_frequency(
        df_surf["ma"].values, df_surf["E_deso"].values, config.site_density)

    if df_ma is not None:
        # TODO: Do we need this check?
        df_ma = df_ma[np.isin(df_ma.index.values, df_surf.index.values)]
        df_surf.loc[df_ma.index, "freq_vib"] = compute_vibration_frequency(
            df_ma["ma"].values,
            df_surf.loc[df_ma.index, "E_deso"].values,
            config.site_density
        )


def compute_vibration_frequency(ma, E_deso, site_density):
    return np.sqrt(FACTOR_VIB_FREQ*site_density*E_deso/ma)


def compute_factor_rate_acc(spec_table, config):
    """Compute the factor to calculate surface accretion rates.

    For surface accretion, the product is a surface specie, while the reactant
    is a normal specie. In practice, we need to set the factor to the
    corresponding normal specie.

    Args:
        spec_table (pd.Dataframe): Specie table.
        config (Config): Config.
        special_dict (dict | None, optional): A dict to specify special cases.
        Defaults to None.
    """
    charge = spec_table["charge"].values
    sticking_coeff = np.zeros_like(spec_table["ma"].values)
    sticking_coeff[charge == 0] = config.sticking_coeff_neutral
    sticking_coeff[charge > 0] = config.sticking_coeff_positive
    sticking_coeff[charge < -1] = config.sticking_coeff_negative

    grain_radius = config.grain_radius
    factor = math.pi*grain_radius*grain_radius*math.sqrt(8.*K_B/math.pi/M_ATOM)
    spec_table["factor_rate_acc"] = factor*sticking_coeff/np.sqrt(spec_table["ma"])
    assign_columns_to_normal_counterparts(spec_table, "factor_rate_acc")


def compute_barrier_energy(spec_table, config, df_barr=None):
    spec_table["E_barr"] = 0.
    cond = spec_table.index.map(lambda name: name.startswith("J"))
    spec_table.loc[cond, "E_barr"] = config.surf_diff_to_deso_ratio \
        *spec_table.loc[cond, "E_deso"].values
    cond = spec_table.index.map(lambda name: name.startswith("K"))
    spec_table.loc[cond, "E_barr"] = config.mant_diff_to_deso_ratio \
        *spec_table.loc[cond, "E_deso"].values
    if df_barr is not None:
        spec_table.loc[df_barr.index, "E_barr"] = df_barr["E_barr"].values


def compute_rate_tunneling(df_surf, df_gap, config):
    df_surf["dE_band"] = 0.
    if df_gap is not None:
        # TODO: Do we need this check?
        df_gap = df_gap[np.isin(df_gap.index.values, df_surf.index.values)]
        df_surf.loc[df_gap.index, "dE_band"] = df_gap["dE_band"].values
    df_surf["rate_tunneling_a"] = df_surf["dE_band"]*(.25*K_B/H_BAR)

    exponent = -2.*config.diffusion_barrier_thickness/H_BAR \
        *np.sqrt(2*K_B*M_ATOM**df_surf["ma"])
    df_surf["rate_tunneling_b"] = df_surf["freq_vib"]*np.exp(exponent)


def assign_surface_params(df_reac, df_spec, df_surf):
    """Assigin surface parameters.

    This is an inplace operation.

    Args:
        df_reac (pd.DataFrame): Data frame of reactions.
        df_spec (pd.DataFrame): Specie table.
        df_surf (pd.DataFrame): Surface parameters of species.
    """
    columns = ["E_barr", "freq_vib"]
    df_spec[columns] = df_surf[columns].values
    df_spec.fillna(0., inplace=True)

    columns = ["E_deso", "freq_vib", "factor_rate_acc"]
    for col in columns:
        df_reac[f"{col}_r1"] = df_surf.loc[df_reac["reactant_1"], col].values

    #columns = ["E_barr", "freq_vib"]
    #df_tmp = df_reac.loc[df_reac["reactant_2"] != "", "reactant_2"]
    #for col in columns:
    #    df_reac.loc[df_tmp.index, f"{col}_r2"] = df_surf.loc[df_tmp, col].values
    df_reac.fillna(0., inplace=True)


def assign_activation_energy(df_reac, df_act):
    df_reac["E_act"] = 0.
    if df_act is None:
        return

    index = np.intersect1d(df_reac["key"], df_act.index)
    df_tmp = pd.DataFrame(df_reac.index, index=df_reac["key"], columns=["index"])
    df_reac.loc[df_tmp.loc[index, "index"], "E_act"] = df_act.loc[index, "E_act"].values

    # When we have the following reactions:
    # - JA + JB > X + Y
    # - JA + JB > JX + JY
    # The activation energy is only given for the second reaction. The code
    # below assign the activation energy for the first reaction.
    cond = df_reac["formula"] == 'surface reaction'
    df_tmp = df_reac.loc[cond, ["key", "reactant_1", "reactant_2", "products", "E_act"]].copy()
    df_tmp["index"] = df_tmp.index.values
    df_lookup = df_tmp[["key", "index", "E_act"]].set_index("key")
    lookup_idx = df_lookup["index"].to_dict()

    inds = []
    inds_surf = []
    for idx, reac_1, reac_2, prods in zip(
        df_tmp.index.values,
        df_tmp["reactant_1"].values,
        df_tmp["reactant_2"].values,
        df_tmp["products"].values,
    ):
        if prods.startswith("J") or prods.startswith("K"):
            continue
        prods = prods.split(";")
        prods = [f"J{prod}" for prod in prods]
        prods = ";".join(prods)
        key = f"{reac_1};{reac_2}>{prods}"
        inds.append(idx)
        inds_surf.append(lookup_idx[key])
    df_reac.loc[inds, "E_act"] = df_reac.loc[inds_surf, "E_act"].values


def compute_branching_ratio(df_reac, spec_table, config):
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
        if prods.startswith("J") or prods.startswith("K"):
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
    df_reac["branching_ratio"] = 1.
    df_reac.loc[df_tmp.index, "branching_ratio"] = df_tmp["branching_ratio"].values

    compute_branching_ratio_rrk_desorption(df_reac, spec_table, config)


def compute_branching_ratio_rrk_desorption(df_reac, spec_table, config):
    u_dHf = units.imperial.kilocal.cgs.scale/K_B/constants.N_A.value # Convert kilocal/mol to K
    cond = (df_reac["formula"] == 'surface reaction') \
        & df_reac["reactant_1"].map(lambda name: name.startswith("J"))
    cols = ["reactant_1", "reactant_2", "products", "E_act", "branching_ratio"]
    df_tmp = df_reac.loc[cond, cols].copy()

    # TODO: Understanding
    prod_first = []
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
    frac_deso *= config.vib_to_dissip_freq_ratio
    frac_deso = frac_deso/(1. + frac_deso)

    frac_deso[(dHf_sum <= 0.) | np.isnan(dHf_sum)] = 0.
    frac_deso[frac_deso < 0.] = 0.
    frac_deso[frac_deso > 1.] = 1.

    cond = list(map(lambda name: name.startswith("J"), prod_first))

    frac_deso[cond] = 1 - frac_deso[cond]
    df_tmp["branching_ratio"] *= frac_deso
    df_reac.loc[df_tmp.index, "branching_ratio"] = df_tmp["branching_ratio"].values


def assign_branching_ratio(df_reac, df_br):
    """Assign external branching ration.

    Args:
        df_reac (pd.DataFrame): Reaction dataframe.
        df_br (pd.DataFrame): Branching ratios.
    """
    if df_br is None:
        return
    inds = df_br.index.get_indexer(df_reac["key"])
    cond = inds != -1
    df_reac.loc[cond, "branching_ratio"] = df_br.iloc[inds[cond]]["branching_ratio"].values


def assign_surface_tunneling_probability(df_reac, df_surf, config):
    cond = df_reac["formula"] == "surface reaction"
    df_sub = df_reac.loc[cond, ["reactant_1", "reactant_2", "E_act"]]
    ma_1 = df_surf.loc[df_sub["reactant_1"], "ma"].values
    ma_2 = df_surf.loc[df_sub["reactant_2"], "ma"].values
    ma_reduce = ma_1*ma_2/(ma_1 + ma_2)
    log_prob_surf_tunl = compute_surface_tunneling_probability(
        ma_reduce, df_sub["E_act"].values, config.chemical_barrier_thickness)
    df_reac["log_prob_surf_tunl"] = -np.inf
    df_reac.loc[df_sub.index, "log_prob_surf_tunl"] = log_prob_surf_tunl


def compute_surface_tunneling_probability(ma_reduce, E_act, barrier_thickness):
    return -2.*barrier_thickness/H_BAR*np.sqrt(2.*ma_reduce*M_ATOM*K_B*E_act)


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
