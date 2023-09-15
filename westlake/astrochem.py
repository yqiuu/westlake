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
from .reaction_rates import ConstantReactionRate
from .reaction_terms import ConstantRateTerm, TwoPhaseTerm, ThreePhaseTerm

import pandas as pd


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


def create_constant_rate_model(df_reac):
    reaction_matrix = ReactionMatrix(df_reac)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    rate_1st = ConstantReactionRate(rmat_1st, df_reac["rate"].values)
    rate_2nd = ConstantReactionRate(rmat_2nd, df_reac["rate"].values)
    # The rate signs are included in the rates, and therefore, they are
    # unnecessary.
    rmat_1st.rate_sign = None
    rmat_2nd.rate_sign = None
    return ConstantRateTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd)


def create_two_phase_model(df_reac, df_spec, df_surf, medium, meta_params,
                           df_act=None, df_ma=None, df_barr=None,
                           param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    prepare_piecewise_rates(df_reac)
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    prepare_surface_reaction_params(
        df_reac, reaction_matrix.df_spec, df_surf, meta_params, df_act, df_ma, df_barr)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(
        df_reac, reaction_matrix.df_spec, rmat_1st, formula_dict_1st, param_names
    )
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(
        df_reac, reaction_matrix.df_spec, rmat_2nd, formula_dict_2nd, param_names
    )
    medium = add_hopping_rate_module(medium, reaction_matrix.df_spec, meta_params)
    return TwoPhaseTerm(rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, medium)


def create_three_phase_model(df_reac, df_spec, df_surf, medium, meta_params,
                             df_act=None, df_ma=None, df_barr=None,
                             param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    prepare_piecewise_rates(df_reac)
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    prepare_surface_reaction_params(
        df_reac, reaction_matrix.df_spec, df_surf, meta_params, df_act, df_ma, df_barr)
    df_spec = reaction_matrix.df_spec

    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    df_reac_sub = pd.concat([
        df_reac[df_reac["formula"] == "mantle to surface"],
        df_reac[df_reac["formula"] == "surface to mantle"],
    ], axis=0)
    reaction_matrix = ReactionMatrix(df_reac_sub, df_spec)
    rmat_1st, _ = reaction_matrix.create_index_matrices()
    rmod_smt, rmat_smt = create_surface_mantle_transition(
        df_reac_sub, reaction_matrix.df_spec, rmat_1st, param_names, meta_params
    )

    cond = (df_reac["formula"] != "mantle to surface") \
        & (df_reac["formula"] != "surface to mantle")
    df_reac = df_reac[cond]
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(
        df_reac, reaction_matrix.df_spec, rmat_1st, formula_dict_1st, param_names
    )
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(
        df_reac, reaction_matrix.df_spec, rmat_2nd, formula_dict_2nd, param_names
    )
    medium = add_hopping_rate_module(medium, reaction_matrix.df_spec, meta_params)

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