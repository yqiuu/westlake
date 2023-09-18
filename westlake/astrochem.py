import numpy as np
import pandas as pd

from .reaction_rates import create_formula_dict_reaction_module, create_surface_mantle_transition
from .gas_reactions import builtin_gas_reactions_1st, builtin_gas_reactions_2nd
from .surface_reactions import (
    builtin_surface_reactions_1st,
    builtin_surface_reactions_2nd,
    prepare_surface_reaction_params
)
from .preprocesses import prepare_piecewise_rates
from .medium import SequentialMedium, ThermalHoppingRate
from .reaction_matrices import ReactionMatrix
from .reaction_terms import TwoPhaseTerm, ThreePhaseTerm
from .solver import solve_rate_equation


def builtin_astrochem_reaction_param_names():
    return [
        "is_leftmost", "is_rightmost", "T_min", "T_max",
        "alpha", "beta", "gamma",
        "E_deso_r1", "freq_vib_r1", "factor_rate_acc_r1",
        "E_act", "branching_ratio", "log_prob_surf_tunl"
    ]


def builtin_astrochem_reactions_1st(meta_params):
    return {
        **builtin_gas_reactions_1st(meta_params),
        **builtin_surface_reactions_1st(meta_params)
    }


def builtin_astrochem_reactions_2nd(meta_params):
    return {
        **builtin_gas_reactions_2nd(meta_params),
        **builtin_surface_reactions_2nd(meta_params)
    }


def create_two_phase_model(df_reac, df_spec, df_surf, medium, meta_params,
                           df_act=None, df_ma=None, df_barr=None,
                           param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    prepare_piecewise_rates(df_reac)
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    prepare_surface_reaction_params(
        df_reac, df_spec, df_surf, meta_params, df_act, df_ma, df_barr)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(
        df_reac, df_spec, rmat_1st, formula_dict_1st, param_names
    )
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(
        df_reac, df_spec, rmat_2nd, formula_dict_2nd, param_names
    )
    medium = add_hopping_rate_module(medium, df_spec, meta_params)
    return TwoPhaseTerm(rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, medium)


def create_three_phase_model(df_reac, df_spec, df_surf, medium, meta_params,
                             df_act=None, df_ma=None, df_barr=None,
                             param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    prepare_piecewise_rates(df_reac)
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    prepare_surface_reaction_params(
        df_reac, df_spec, df_surf, meta_params, df_act, df_ma, df_barr)

    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    df_reac_sub = pd.concat([
        df_reac[df_reac["formula"] == "mantle to surface"],
        df_reac[df_reac["formula"] == "surface to mantle"],
    ], axis=0)
    reaction_matrix = ReactionMatrix(df_reac_sub, df_spec)
    rmat_1st, _ = reaction_matrix.create_index_matrices()
    rmod_smt, rmat_smt = create_surface_mantle_transition(
        df_reac_sub, df_spec, rmat_1st, param_names, meta_params
    )

    cond = (df_reac["formula"] != "mantle to surface") \
        & (df_reac["formula"] != "surface to mantle")
    df_reac = df_reac[cond]
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(
        df_reac, df_spec, rmat_1st, formula_dict_1st, param_names
    )
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(
        df_reac, df_spec, rmat_2nd, formula_dict_2nd, param_names
    )
    medium = add_hopping_rate_module(medium, df_spec, meta_params)

    rmat_1st_surf, rmat_1st_other = split_surface_reactions(df_reac, rmat_1st)
    rmat_1st_surf_gain, rmat_1st_surf_loss = split_gain_loss(rmat_1st_surf)

    rmat_2nd_surf, rmat_2nd_other = split_surface_reactions(df_reac, rmat_2nd)
    rmat_2nd_surf_gain, rmat_2nd_surf_loss = split_gain_loss(rmat_2nd_surf)

    inds_surf = df_spec.index.map(lambda name: name.startswith("J")).values
    inds_mant = df_spec.index.map(lambda name: name.startswith("K")).values

    return ThreePhaseTerm(
        rmod_smt, rmat_smt,
        rmod_1st, rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss, rmat_1st_other,
        rmod_2nd, rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss, rmat_2nd_other,
        inds_surf, inds_mant, medium
    )


def split_surface_reactions(df_reac, rmat):
    cond = df_reac["key"].loc[rmat.inds_id].map(lambda name: "J" in name)
    return rmat.split(cond)


def split_gain_loss(rmat):
    cond = rmat.rate_sign > 0
    return rmat.split(cond)


def add_hopping_rate_module(medium, df_spec, meta_params):
    module = ThermalHoppingRate(
        df_spec["E_barr"].values, df_spec["freq_vib"].values, meta_params
    )
    if isinstance(medium, SequentialMedium):
        medium.append(module)
    else:
        medium = SequentialMedium(medium, module)
    return medium


def solve_rate_equation_astrochem(reaction_term, t_span, ab_0_dict, df_spec, meta_params,
                                  t_eval=None, device="cpu", show_progress=True):
    ab_0 = dervie_initial_abundances(ab_0_dict, df_spec, meta_params)
    res = solve_rate_equation(
        reaction_term, t_span, ab_0,
        method=meta_params.solver,
        rtol=meta_params.rtol,
        atol=meta_params.atol,
        t_eval=t_eval,
        u_factor=meta_params.to_second,
        device=device,
        show_progress=show_progress
    )
    return res


def dervie_initial_abundances(ab_0_dict, spec_table, meta_params):
    """Derive the initial abundances of grains and electrons.

    Args:
        ab_0 (dict): Initial abundance of each specie.
        spec_table (pd.DataFrame): Specie table.
        meta_params (MetaParameters): Meta parameters.
        ab_0_min (float, optional): Mimimum initial abundances. Defaults to 0.
        dtype (str, optional): Data type of the return abundances. Defaults to 'tuple'.

    Returns:
        tuple: Initial abundances.
    """
    if not all(np.in1d(list(ab_0_dict.keys()), spec_table.index.values)):
        raise ValueError("Find unrecognized species in 'ab_0'.")

    ab_0 = np.full(len(spec_table), meta_params.ab_0_min)
    ab_0[spec_table.loc[ab_0_dict.keys()]["index"].values] = list(ab_0_dict.values())

    # Derive the grain abundances
    ab_0[spec_table.loc["GRAIN0", "index"]] = meta_params.dtg_mass_ratio_0/meta_params.grain_mass
    ab_0[spec_table.loc["GRAIN-", "index"]] = 0.

    # Derive the electron abundance aussming the system is neutral
    ab_0[spec_table.loc["e-", "index"]] = 0.
    ab_0[spec_table.loc["e-", "index"]] = np.sum(spec_table["charge"].values*ab_0)

    return ab_0