from .utils import data_frame_to_tensor_dict
from .reaction_rates import FormulaDictReactionRate
from .gas_reactions import builtin_gas_reactions_1st, builtin_gas_reactions_2nd
from .surface_reactions import builtin_surface_reactions_1st, builtin_surface_reactions_2nd


def builtin_astrochem_reaction_param_names():
    return [
        "is_unique", "T_min", "T_max", "alpha", "beta", "gamma",
        "E_deso_r1", "E_barr_r1", "freq_vib_r1", "factor_rate_acc_r1",
        "E_barr_r2", "freq_vib_r2", "branching_ratio"
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


def create_astrochem_reactions(df_reac, rmat, module_env, meta_params,
                                   param_names=None, formula_dict=None):
    if param_names is None:
        param_names = builtin_astrochem_reaction_param_names()
    if formula_dict is None:
        if rmat.order == 1:
            formula_dict = builtin_astrochem_reactions_1st(meta_params)
        if rmat.order == 2:
            formula_dict = builtin_astrochem_reactions_2nd(meta_params)

    df_sub = df_reac.iloc[rmat.inds]
    return FormulaDictReactionRate(
        formula_dict,
        df_sub["formula"].values,
        rmat,
        module_env,
        data_frame_to_tensor_dict(df_sub[param_names]),
    )

"""
def create_gas_reaction_module_1st(formula, rmat, module_env, params_reac, meta_params):
    return FormulaDictReactionRate(
        builtin_gas_reaction_formulae_1st(meta_params), formula, rmat, module_env, params_reac)


def create_gas_reaction_module_2nd(formula, rmat, module_env, params_reac, meta_params):
    return FormulaDictReactionRate(
        builtin_gas_reaction_formulae_2nd(meta_params), formula, rmat, module_env, params_reac)
"""

