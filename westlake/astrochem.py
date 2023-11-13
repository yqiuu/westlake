import numpy as np
import pandas as pd

from .reaction_rates import create_formula_dict_reaction_module, create_surface_mantle_transition
from .gas_reactions import builtin_gas_reactions
from .surface_reactions import (
    builtin_surface_reactions,
    prepare_surface_reaction_params,
    NoReaction,
)
from .preprocesses import prepare_piecewise_rates
from .medium import StaticMedium, SequentialMedium, ThermalHoppingRate
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


def builtin_astrochem_reactions(meta_params):
    return {
        **builtin_gas_reactions(meta_params),
        **builtin_surface_reactions(meta_params)
    }


def create_astrochem_model(df_reac, df_spec, df_surf, meta_params,
                           medium=None, df_act=None, df_ma=None, df_barr=None,
                           param_names=None, formula_dict=None):
    if meta_params.use_static_medium and medium is None:
        medium = StaticMedium({
            'Av': meta_params.Av,
            "den_gas": meta_params.den_gas,
            "T_gas": meta_params.T_gas,
            "T_dust": meta_params.T_dust
        })

    prepare_piecewise_rates(df_reac)
    prepare_surface_reaction_params(
        df_reac, df_spec, df_surf, meta_params, df_act, df_ma, df_barr)
    medium = add_hopping_rate_module(medium, df_spec, meta_params)

    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    param_names_ = builtin_astrochem_reaction_param_names()
    if param_names is not None:
        param_names_.extend(param_names)
    formula_dict_ = builtin_astrochem_reactions(meta_params)
    if formula_dict is not None:
        formula_dict_.update(formula_dict)
    if not meta_params.use_photodesorption:
        formula_dict_["CR photodesorption"] = NoReaction()
        formula_dict_["UV photodesorption"] = NoReaction()
    rmod = create_formula_dict_reaction_module(
        df_reac, df_spec, formula_dict_, param_names_
    )

    if meta_params.model == "two phase":
        return TwoPhaseTerm(rmod, rmat_1st, rmat_2nd, medium)
    elif meta_params.model == "three phase":
        rmod_smt = create_surface_mantle_transition(
            df_reac, df_spec, param_names_, meta_params
        )

        rmat_1st_surf, rmat_1st_other = split_surface_reactions(df_reac, rmat_1st)
        rmat_1st_surf_gain, rmat_1st_surf_loss = split_gain_loss(rmat_1st_surf)

        rmat_2nd_surf, rmat_2nd_other = split_surface_reactions(df_reac, rmat_2nd)
        rmat_2nd_surf_gain, rmat_2nd_surf_loss = split_gain_loss(rmat_2nd_surf)

        if meta_params.use_photodesorption:
            formula_list = ["CR photodesorption", "UV photodesorption"]
            rmat_photodeso = extract_by_formula(rmat_1st, df_reac, formula_list)
        else:
            rmat_photodeso = None

        inds_surf = df_spec.index.map(lambda name: name.startswith("J")).values
        inds_mant = df_spec.index.map(lambda name: name.startswith("K")).values

        return ThreePhaseTerm(
            rmod, rmod_smt,
            rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss,
            rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss,
            rmat_photodeso, inds_surf, inds_mant, medium
        )
    else:
        raise ValueError(f"Unknown model: {meta_params.model}")


def split_reac_table_by_formula(df_reac, formula_list):
    conditions = np.vstack([df_reac["formula"].values == formula for formula in formula_list])
    df_reac_target = pd.concat([df_reac[cond] for cond in conditions], axis=0)
    df_reac_remain = df_reac[~np.any(conditions, axis=0)]
    return df_reac_target, df_reac_remain


def extract_by_formula(rmat, df_reac, formula_list):
    df_sub = df_reac["formula"].loc[rmat.inds_id_uni].values
    cond = np.full(len(df_sub), False)
    for formula in formula_list:
        cond |= df_sub == formula
    return rmat.extract(cond, use_id_uni=True)


def split_surface_reactions(df_reac, rmat):
    cond = df_reac["key"].loc[rmat.inds_id_uni].map(lambda name: "J" in name)
    return rmat.split(cond, use_id_uni=True)


def split_gain_loss(rmat):
    cond = rmat.rate_sign > 0
    return rmat.split(cond, use_id_uni=False)


def add_hopping_rate_module(medium, df_spec, meta_params):
    module = ThermalHoppingRate(
        df_spec["E_barr"].values, df_spec["freq_vib"].values, meta_params
    )
    if isinstance(medium, SequentialMedium):
        medium.append(module)
    else:
        medium = SequentialMedium(medium, module)
    return medium


def solve_rate_equation_astrochem(reaction_term, ab_0_dict, df_spec, meta_params,
                                  t_span=None, t_eval=None, method=None, rtol=None, atol=None,
                                  device="cpu", show_progress=True):
    """Solve the rate equations for astrochemical problems.

    Args:
        reaction_term (nn.Module): Definition of the rate equations.
        ab_0_dict (dict): Initial abundances.
        df_spec (pd.DataFrame): Specie table.
        show_progress: If True, print messages to show the progress.
            Defaults to True.

    Returns:
        object: A result object returned by a scipy ODE solver.
    """
    ab_0 = derive_initial_abundances(ab_0_dict, df_spec, meta_params)
    if t_span is None:
        t_span = (meta_params.t_start, meta_params.t_end)
    if method is None:
        method = meta_params.solver
    if rtol is None:
        rtol = meta_params.rtol
    if atol is None:
        atol = meta_params.atol
    res = solve_rate_equation(
        reaction_term, t_span, ab_0,
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        u_factor=meta_params.to_second,
        device=device,
        show_progress=show_progress
    )
    return res


def derive_initial_abundances(ab_0_dict, spec_table, meta_params):
    """Derive the initial abundances of grains and electrons.

    Args:
        ab_0_dict (dict): Initial abundance of each specie.
        spec_table (pd.DataFrame): Specie table.
        meta_params (MetaParameters): Meta parameters.

    Returns:
        array: Initial abundances.
    """
    if not all(np.in1d(list(ab_0_dict.keys()), spec_table.index.values)):
        raise ValueError("Find unrecognized species in 'ab_0'.")

    ab_0 = np.full(len(spec_table), meta_params.ab_0_min)
    ab_0[spec_table.index.get_indexer(ab_0_dict.keys())] = list(ab_0_dict.values())

    # Derive the grain abundances
    ab_0[spec_table.index.get_indexer(["GRAIN0"]).item()] = meta_params.dtg_mass_ratio_0/meta_params.grain_mass
    ab_0[spec_table.index.get_indexer(["GRAIN-"]).item()] = 0.

    # Derive the electron abundance aussming the system is neutral
    ab_0[spec_table.index.get_indexer(["e-"]).item()] = 0.
    ab_0[spec_table.index.get_indexer(["e-"]).item()] = np.sum(spec_table["charge"].values*ab_0)

    return ab_0