from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import data_frame_to_tensor_dict
from .reaction_rates import create_formula_dict_reaction_module
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

    return create_formula_dict_reaction_module(
        df_reac, rmat, formula_dict, module_env, param_names)