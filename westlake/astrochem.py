import pickle
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from .utils import get_specie_index
from .reaction_modules import (
    create_formula_dict_reaction_module,
    create_surface_mantle_transition,
    ConstantRateModule
)
from .reaction_rates import (
    builtin_gas_reactions,
    builtin_surface_reactions,
    prepare_surface_reaction_params,
    NoReaction,
    H2Shielding_Lee1996,
    COShielding_Lee1996,
    SurfaceReactionWithCompetition
)
from .preprocesses import prepare_piecewise_rates
from .medium import Medium
from .reaction_matrices import ReactionMatrix
from .reaction_terms import (
    TwoPhaseTerm,
    ThreePhaseTerm,
    VariableModule,
    ThermalHoppingRate,
    EvaporationRate,
)
from .solvers import solve_ivp_torch, solve_ivp_scipy


def builtin_astrochem_reactions(config):
    return {
        **builtin_gas_reactions(config),
        **builtin_surface_reactions(config)
    }


def create_astrochem_model(df_reac, df_spec, df_surf, config,
                           medium=None, df_act=None, df_br=None,
                           df_barr=None, df_gap=None, df_ma=None,
                           formula_dict=None, formula_dict_ex=None,
                           use_copy=True):
    """Create the primary object for computing the chemical model.

    Args:
        df_reac (pd.DataFrame): The reaction dataframe should include:
            - index: Nautral number.
            - 'reactant_1': The first reactant.
            - 'reactant_2': The second reactant. Use '' for first order
              reactions.
            - 'products": Each product should be joined using ';'. For instance,
            'H;OH'.
            - 'formula': Formula name.
            - 'alpha': Reaction parameter.
            - 'beta': Reaction parameter.
            - 'gamma': Reaction parameter.
            - 'T_min': Minimum reaction temperature.
            - 'T_max': Maximum reaction temperature.

        df_spec (pd.DataFrame): The specie dataframe should include:
            - index: Specie name.
            - 'charge': Charge.
            - 'num_atoms': Number of atoms.
            - 'ma': Molecular or atomic weight [amu].

        df_surf (pd.DataFrame): Surface parameters. The dataframe should
            include:
            - index: Specie name.
            - 'E_deso': Desorption energy [K].
            - 'dHf': Enthalpy of formation [kcal/mol].

        config (Config): Config.
        medium (Medium, optional): Medium. Defaults to None.
        df_act (pd.DataFrame, optional): Additional Activation energy. The
        default value is 0. Defaults to None.
            The dataframe should include:
            - index: Reaction key.
            - 'E_act': Activation energy [K].

        df_br (pd.DataFrame, optional): Additional Branching ratio. This will
            overwrite all builtin branching ratios. Defaults to None.
            The dataframe should include:
            - index: Reaction key.
            - 'branching_ratio': Branching ratio.

        df_barr (pd.DataFrame, optional): Additional diffusion energy. This will
            overwrite all builtin values, which are obtained by multiplying a
            factor to the desorption energy. Defaults to None. The dataframe
            should include:
            - index: Specie name.
            - 'E_barr': Diffusion energy [K].

        df_gap (pd.DataFrame, optional): Additional gap energy. Defaults to
            None. The dataframe should include:
            - index: Specie name.
            - 'dE_band': Gap energy.

        df_ma (pd.DataFrame, optional): Additional molecular weights. Defaults
            to None. The dataframe should include:
            - index: Specie name.
            - 'ma': molecular weight [amu].

        formula_dict (dict, optional): User-defined reaction formulae.
            Defaults to None.
            - key: Formula name.
            - value: An instance of `ReactionRate`.

        formula_dict_ex (dict, optional): User-defined extra reaction formulae.
            Defaults to None.
            - key: Formula name.
            - value: An instance of `ReactionRate`.

        use_copy (bool, optional): If True, copy the input reaction and specie
            dataframe. Defaults to True.

    Returns:
        nn.Module: A callable that can be passed to a ODE solver.
    """
    validate_specie_params(df_spec, "'df_spec'")
    validate_specie_params(df_surf, "'df_surf'")

    if use_copy:
        df_reac = df_reac.copy()
        df_spec = df_spec.copy()

    # Preprocess reactions
    df_reac = assign_reaction_key(df_reac)
    remove_special_species(df_reac, config)
    prepare_piecewise_rates(df_reac)
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()

    if medium is None:
        medium = Medium(config)
    if config.model != "simple":
        prepare_surface_reaction_params(
            df_reac, df_spec, df_surf, config,
            df_act=df_act, df_br=df_br,
            df_barr=df_barr, df_gap=df_gap, df_ma=df_ma,
        )
        vmod = VariableModule()
        vmod.add_variable("rate_hopping", create_hopping_rate_module(df_spec, config))
    else:
        vmod = None
    # Prepare formula dict
    formula_dict_ = builtin_gas_reactions(config)
    formula_dict_ex_ = {}

    # Shielding
    df_reac = add_H2_shielding(df_reac, df_spec, config, formula_dict_ex_)
    df_reac = add_CO_shielding(df_reac, df_spec, config, formula_dict_ex_)

    # Add surface reactions
    formula_dict_surf = builtin_surface_reactions(config)
    if config.model == "simple":
        for key in formula_dict_surf:
            formula_dict_[key] = NoReaction()
    else:
        formula_dict_["surface H2 formation"] = NoReaction()
        formula_dict_["surface H accretion"] = NoReaction()
        formula_dict_.update(formula_dict_surf)

    # Competition
    if config.use_competition:
        formula_dict_ex_["surface reaction"] = SurfaceReactionWithCompetition(config)
        formula_dict_.pop("surface reaction")
    elif config.model == "three phase":
        formula_dict_ex_["surface reaction"] = formula_dict_.pop("surface reaction")

    if config.model == "three phase":
        formula_dict_ex_["UV photodesorption"] = formula_dict_.pop("UV photodesorption")
        formula_dict_ex_["CR photodesorption"] = formula_dict_.pop("CR photodesorption")

    # Add user-defined formulae
    if formula_dict is not None:
        formula_dict_.update(formula_dict)
    if formula_dict_ex is not None:
        formula_dict_ex_.update(formula_dict_ex)

    rmod, rmod_ex =  create_formula_dict_reaction_module(
        df_reac, df_spec, formula_dict_, formula_dict_ex_,
    )

    #
    vmod_ex = VariableModule()
    vmod_ex.add_variable("k_evapor", create_evapor_rate_module(df_reac, df_spec))

    if config.model == "simple" or config.model == "two phase":
        return TwoPhaseTerm(rmod, rmod_ex, vmod, vmod_ex, rmat_1st, rmat_2nd, medium)
    elif config.model == "three phase":
        rmod_smt = create_surface_mantle_transition(df_reac, df_spec, config)

        rmat_1st_surf, rmat_1st_other = split_surface_reactions(df_reac, rmat_1st)
        rmat_1st_surf_gain, rmat_1st_surf_loss = split_gain_loss(rmat_1st_surf)

        rmat_2nd_surf, rmat_2nd_other = split_surface_reactions(df_reac, rmat_2nd)
        rmat_2nd_surf_gain, rmat_2nd_surf_loss = split_gain_loss(rmat_2nd_surf)

        inds_surf = df_spec.index.map(lambda name: name.startswith("J")).values
        inds_mant = df_spec.index.map(lambda name: name.startswith("K")).values

        return ThreePhaseTerm(
            rmod, rmod_ex, vmod, vmod_ex, rmod_smt,
            rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss,
            rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss,
            inds_surf, inds_mant, medium, config
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")


def assign_reaction_key(df_reac):
    keys = []
    for reac_1, reac_2, prods in zip(
        df_reac["reactant_1"], df_reac["reactant_2"], df_reac["products"]
    ):
        if reac_2 == "":
            key = f"{reac_1}>{prods}"
        else:
            key = f"{reac_1};{reac_2}>{prods}"
        keys.append(key)
    if "key" in df_reac:
        df_reac["key"] = keys
    else:
        df_reac.insert(0, "key", keys)
    return df_reac


def remove_special_species(df_reac, config):
    "Replace special species such as CR with ''."
    replace_dict = {key: "" for key in config.special_species}
    df_reac.replace(replace_dict, inplace=True)

    def _remove(products):
        return ";".join([prod for prod in products.split(";")
                         if prod not in config.special_species])
    df_reac["products"] = df_reac["products"].map(_remove).values


def add_H2_shielding(df_reac, df_spec, config, formula_dict_ex):
    """Add H2 shielding to `formula_dict_ex` if applicable.

    This function changes `formula_dict_ex` inplace. The specie table must
    have H2.

    Returns:
        pd.DataFrame: Reaction data with updated H2 shielding formula.
    """
    if config.H2_shielding == "Lee+1996":
        cond = (df_reac["formula"] == "photodissociation") & (df_reac["reactant_1"] == "H2")
        n_reac = np.count_nonzero(cond)
        if n_reac == 0:
            pass
        elif n_reac == 1:
            idx_reac = df_reac[cond].index.item()
            fm_name = "H2 Shielding"
            df_reac = df_reac.copy()
            df_reac.loc[idx_reac, "formula"] = fm_name
            idx_H2 = get_specie_index(df_spec, "H2")
            formula_dict_ex[fm_name] = H2Shielding_Lee1996(idx_H2, config)
        else:
            raise ValueError("Multiple H2 shielding reactions.")
    elif config.H2_shielding is None:
        pass
    else:
        raise ValueError("Unknown H2 shielding: '{}'.".format(config.H2_shielding))
    return df_reac


def add_CO_shielding(df_reac, df_spec, config, formula_dict_ex):
    """Add CO shielding to `formula_dict_ex` if applicable.

    This function changes `formula_dict_ex` inplace. The specie table must
    have H2 and CO.

    Returns:
        pd.DataFrame: Reaction data with updated CO shielding formula.
    """
    if config.CO_shielding == "Lee+1996":
        cond = (df_reac["formula"] == "photodissociation") & (df_reac["reactant_1"] == "CO")
        n_reac = np.count_nonzero(cond)
        if n_reac == 0:
            pass
        elif n_reac == 1:
            idx_reac = df_reac[cond].index.item()
            fm_name = "CO Shielding (Lee+1996)"
            df_reac = df_reac.copy()
            df_reac.loc[idx_reac, "formula"] = fm_name
            idx_CO = get_specie_index(df_spec, "CO")
            idx_H2 = get_specie_index(df_spec, "H2")
            formula_dict_ex[fm_name] = COShielding_Lee1996(idx_CO, idx_H2, config)
        else:
            raise ValueError("Multiple CO shielding reactions.")
    elif config.CO_shielding is None:
        pass
    else:
        raise ValueError("Unknown CO shielding: '{}'.".format(config.CO_shielding))
    return df_reac


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


def create_hopping_rate_module(df_spec, config):
    return ThermalHoppingRate(
        df_spec["E_barr"].values, df_spec["freq_vib"].values, config
    )


def create_evapor_rate_module(df_reac, df_spec):
    cond = (df_reac["formula"] == "thermal evaporation") \
        | (df_reac["formula"] == "CR evaporation") \
        | (df_reac["formula"] == "UV photodesorption") \
        | (df_reac["formula"] == "CR photodesorption")
    inds_evapor = np.where(cond.values)[0]
    inds_evapor = torch.tensor(inds_evapor)

    # Evaporation reactions by definition only have one reactant
    reacs = df_reac.loc[cond, "reactant_1"]
    inds_r = df_spec.index.get_indexer(reacs)
    inds_r = torch.tensor(inds_r)

    n_spec = len(df_spec)
    return EvaporationRate(inds_evapor, inds_r, n_spec)


def solve_rate_equation_astrochem(reaction_term, ab_0_dict, df_spec, config, *,
                                  medium_list=None, t_span=None, t_eval=None,
                                  use_scipy_solver=None, method=None,
                                  rtol=None, atol=None, use_auto_jac=None,
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
    def _solve(reac_term, df_spec, t_span, ab_0, method, use_scipy_solver, kwargs):
        reac_term = replace_with_constant_rate_module(reac_term, df_spec)
        if use_scipy_solver:
            res = solve_ivp_scipy(reaction_term, t_span, ab_0, method=method, **kwargs)
        else:
            res = solve_ivp_torch(reaction_term, t_span, ab_0, **kwargs)
        t_in = torch.as_tensor(res.t)[:, None] # (N_time, 1)
        y_in = torch.as_tensor(res.y).T # (N_time, N_spec)
        coeffs, den_gas = reac_term.compute_rate_coeffs(t_in, y_in)
        den_gas = np.ravel(den_gas.numpy())
        if not config.save_rate_coeffs:
            coeffs = None
        else:
            coeffs = coeffs.numpy().T # (N_reac, N_time)
        return Result(
            message=res.message,
            success=res.success,
            time=res.t,
            ab=res.y,
            species=tuple(df_spec.index),
            den_gas=den_gas,
            coeffs=coeffs
        )

    if t_span is None:
        t_span = (config.t_start, config.t_end)
    kwargs = {
        "t_eval": t_eval,
        "u_factor": config.to_second,
        "device": device,
        "show_progress": show_progress,
    }
    if method is None:
        method = config.method
    if use_auto_jac is None:
        kwargs["use_auto_jac"] = config.use_auto_jac
    else:
        kwargs["use_auto_jac"] = use_auto_jac
    if rtol is None:
        kwargs["rtol"] = config.rtol
    else:
        kwargs["rtol"] = rtol
    if atol is None:
        kwargs["atol"] = config.atol
    else:
        kwargs["atol"] = atol
    if use_scipy_solver is None:
        use_scipy_solver = config.use_scipy_solver

    ab_0 = derive_initial_abundances(ab_0_dict, df_spec, config)
    if medium_list is None:
        return _solve(reaction_term, df_spec, t_span, ab_0, method, use_scipy_solver, kwargs)

    res_tot = None
    reaction_term = deepcopy(reaction_term)
    for i_stage, module_med in enumerate(medium_list):
        reaction_term.module_med = module_med
        t_span_sub = (t_span[i_stage], t_span[i_stage + 1])
        res = _solve(reaction_term, df_spec, t_span_sub, ab_0, kwargs)
        if not res.success:
            raise ValueError(res.message)
        ab_0 = res.last()
        if res_tot is None:
            res_tot = res
        else:
            res_tot.append(res)
    return res_tot


def derive_initial_abundances(ab_0_dict, df_spec, config):
    """Derive the initial abundances of grains and electrons.

    Args:
        ab_0_dict (dict): Initial abundance of each specie.
        df_spec (pd.DataFrame): Specie table.
        config (Config): Config.

    Returns:
        array: Initial abundances.
    """
    if not all(np.in1d(list(ab_0_dict.keys()), df_spec.index.values)):
        raise ValueError("Find unrecognized species in 'ab_0'.")

    ab_0 = np.full(len(df_spec), config.ab_0_min)
    ab_0[df_spec.index.get_indexer(ab_0_dict.keys())] = list(ab_0_dict.values())

    # Derive the grain abundances
    idx = get_specie_index(df_spec, "GRAIN0", raise_error=False)
    if idx is not None:
        ab_0[idx] = config.dtg_mass_ratio/config.grain_mass
    idx = get_specie_index(df_spec, "GRAIN-", raise_error=False)
    if idx is not None:
        ab_0[idx] = 0.

    # Derive the electron abundance aussming the system is neutral
    idx = get_specie_index(df_spec, "e-")
    ab_0[idx] = 0.
    ab_0[idx] = np.sum(df_spec["charge"].values*ab_0)
    return ab_0


def replace_with_constant_rate_module(reac_term, df_spec):
    """Use constant module to improve the performance if possible."""
    if reac_term.module_med.is_static():
        reac_term_new = deepcopy(reac_term)
        t_in = torch.zeros(1)
        y_in = torch.zeros(len(df_spec))
        coeffs = reac_term_new.reproduce_rate_coeffs(t_in, y_in)
        rmod = ConstantRateModule(coeffs, reac_term_new.rmod.inds_reac.clone())
        reac_term_new.rmod = rmod
        return reac_term_new
    return reac_term


def validate_specie_params(df_spec, var_name):
    counts = Counter(df_spec.index.values)
    dups = []
    for spec, num in counts.items():
        if num != 1:
            dups.append(spec)
    if len(dups) != 0:
        raise ValueError(f"Find duplicated species in {var_name}: " + ", ".join(dups))


def save_result(res, fname):
    pickle.dump(res.to_dict(), open(fname, "wb"))


def load_result(fname):
    return Result(**pickle.load(open(fname, "rb")))


class Result:
    """Simulation result.

    Args:
        message (str): Message returned by the ODE solver.
        success (bool): Success status returned by the ODE solver.
        time (array): (N_time,). Simulation time.
        ab (array): (N_spec, N_time). Abundances.
        species (list): (N_spec,). Species.
        den_gas (array): (N_time,). Gas density.
        coeffs (array): (N_reac, N_time). Rate coefficients. This can be `None`.
        stages (list): Time index pair of each stage. If this is `None`, the
            code assumes that there is only one stage.
    """
    def __init__(self, message, success, time, ab, species, den_gas, coeffs, stages=None):
        self._message = message
        self._success = success
        self._time = time
        self._ab = ab
        self._species = {key: idx for idx, key in enumerate(species)}
        self._den_gas = den_gas
        self._coeffs = coeffs
        if stages is None:
            self._stages = [(0, len(time) - 1)]
        else:
            self._stages = stages

    def __repr__(self):
        text = f"message: {self._message}\n" \
            + f"success: {self._success}.\n" \
            + f"species: Specie list ({len(self.species)},).\n" \
            + f"stages: Time index pair of each stage ({len(self.stages)},).\n" \
            + f"time: Time {self.time.shape}.\n" \
            + f"ab: Abundances {self.ab.shape}.\n" \
            + f"den_gas: Gas density {self.den_gas.shape}."
        if self.coeffs is not None:
            text += f"\ncoeffs: Rate coefficients {self.coeffs.shape}."
        return text

    def __getitem__(self, key):
        if key == "time":
            return self._time
        else:
            return self._ab[self._species[key]]

    @property
    def message(self):
        return self._message

    @property
    def success(self):
        return self._success

    @property
    def species(self):
        """Species."""
        return tuple(self._species.keys())

    @property
    def stages(self):
        """Time index pair of each stage."""
        return self._stages

    @property
    def time(self):
        """Time (N_time,)."""
        return self._time

    @property
    def ab(self):
        """Abundance data (N_spec, N_time)."""
        return self._ab

    @property
    def den_gas(self):
        """Gas density (N_time,)."""
        return self._den_gas

    @property
    def coeffs(self):
        """Rate coefficients (N_reac, N_time)."""
        return self._coeffs

    def to_dict(self):
        return {
            "message": self._message,
            "success": self._success,
            "time": self._time,
            "ab": self._ab,
            "species": self._species,
            "den_gas": self._den_gas,
            "coeffs": self._coeffs,
            "stages": self._stages
        }

    def last(self):
        return self._ab[:, -1]

    def append(self, res):
        offset = self._stages[-1][-1] + 1
        stages = []
        for i_stage, (idx_b, idx_e) in enumerate(res.stages):
            # The first is removed
            if i_stage == 0:
                stages.append((idx_b + offset, idx_e + offset - 1))
            else:
                stages.append((idx_b + offset - 1, idx_e + offset - 1))
        self._stages.extend(stages)
        self._time = np.append(self._time, res._time[1:])
        self._ab = np.concatenate([self._ab, res._ab[:, 1:]], axis=-1)
        self._den_gas = np.append(self._den_gas, res._den_gas[1:])
        self._coeffs = np.concatenate([self._coeffs, res._coeffs[:, 1:]], axis=-1)