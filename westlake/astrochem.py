from .reaction_rates import create_formula_dict_reaction_module
from .gas_reactions import builtin_gas_reactions_1st, builtin_gas_reactions_2nd
from .surface_reactions import builtin_surface_reactions_1st, builtin_surface_reactions_2nd
from .equation import ReactionTerm


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