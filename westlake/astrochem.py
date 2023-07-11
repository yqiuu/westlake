from .reaction_rates import create_formula_dict_reaction_module
from .gas_reactions import builtin_gas_reactions_1st, builtin_gas_reactions_2nd
from .surface_reactions import builtin_surface_reactions_1st, builtin_surface_reactions_2nd
from .reaction_matrices import ReactionMatrix
from .equation import ReactionTerm, ThreePhaseTerm


def builtin_astrochem_reaction_param_names():
    return [
        "is_unique", "T_min", "T_max", "alpha", "beta", "gamma",
        "E_deso_r1", "E_barr_r1", "freq_vib_r1", "factor_rate_acc_r1",
        "E_barr_r2", "freq_vib_r2",
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


def create_two_phase_model(reaction_matrix, df_reac, medium, meta_params,
                           param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(df_reac, rmat_1st, formula_dict_1st, param_names)
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(df_reac, rmat_2nd, formula_dict_2nd, param_names)
    return ReactionTerm(rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, medium)


def create_three_phase_model(df_reac, medium, meta_params,
                             param_names=None, formula_dict_1st=None, formula_dict_2nd=None):
    cond = (df_reac["formula"] == "mantle to surface") \
        & (df_reac["formula"] == "surface to mantle")
    df_reac_sub = df_reac[cond]

    df_reac = df_reac[~cond]
    reaction_matrix = ReactionMatrix(df_reac["reactant_1"], df_reac["reactant_2"], df_reac["products"])
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    if formula_dict_1st is None:
        formula_dict_1st = builtin_astrochem_reactions_1st(meta_params)
    rmod_1st, rmat_1st = create_formula_dict_reaction_module(df_reac, rmat_1st, formula_dict_1st, param_names)
    if formula_dict_2nd is None:
        formula_dict_2nd = builtin_astrochem_reactions_2nd(meta_params)
    rmod_2nd, rmat_2nd = create_formula_dict_reaction_module(df_reac, rmat_2nd, formula_dict_2nd, param_names)

    rmat_1st_surf, rmat_1st_other = split_surface_reactions(df_reac, rmat_1st)
    rmat_1st_surf_gain, rmat_1st_surf_loss = split_gain_loss(rmat_1st_surf)

    rmat_2nd_surf, rmat_2nd_other = split_surface_reactions(df_reac, rmat_2nd)
    rmat_2nd_surf_gain, rmat_2nd_surf_loss = split_gain_loss(rmat_2nd_surf)

    inds_surf = reaction_matrix.df_spec.index.map(lambda name: name.startswith("J")).values
    inds_mant = reaction_matrix.df_spec.index.map(lambda name: name.startswith("K")).values

    return ThreePhaseTerm(
        rmod_1st, None, None, rmat_1st_surf_gain, rmat_1st_surf_loss, rmat_1st_other,
        rmod_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss, rmat_2nd_other,
        inds_surf, inds_mant, medium
    )


def split_surface_reactions(df_reac, rmat):
    cond = df_reac["key"].iloc[rmat.inds_id].map(lambda name: "J" in name)
    return rmat.split(cond)

def split_gain_loss(rmat):
    cond = rmat.rate_sign > 0
    return rmat.split(cond)