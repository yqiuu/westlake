import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import jacrev

from dataclasses import dataclass

from .utils import TensorDict, data_frame_to_tensor_dict
from .meta_params import MetaParameters
from .reaction_matrices import ReactionMatrix
from .assemblers import Assembler


class ConstantRateTerm(nn.Module):
    def __init__(self, rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, den_norm=None):
        super(ConstantRateTerm, self).__init__()
        self.register_reactions("1st", rmat_1st, rmod_1st)
        self.register_reactions("2nd", rmat_2nd, rmod_2nd)
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni
        if den_norm is None:
            den_norm = torch.ones(1)
        else:
            den_norm = torch.tensor(float(den_norm))
        self.register_buffer("den_norm", den_norm)

    def register_reactions(self, postfix, rmat, rmod):
        setattr(self, f"asm_{postfix}", Assembler(rmat))
        setattr(self, f"rmod_{postfix}", rmod)

    def forward(self, t_in, y_in):
        rates_1st, rates_2nd = self.compute_rates()
        return self.asm_1st(y_in, rates_1st, self.den_norm) \
            + self.asm_2nd(y_in, rates_2nd, self.den_norm)

    def jacobian(self, t_in, y_in):
        rates_1st, rates_2nd = self.compute_rates()
        return self.asm_1st.jacobain(y_in, rates_1st, self.den_norm) \
            + self.asm_2nd.jacobain(y_in, rates_2nd, self.den_norm)

    def compute_rates(self):
        rates_1st = self.rmod_1st()
        rates_2nd = self.rmod_2nd()
        return rates_1st, rates_2nd


class TwoPhaseTerm(nn.Module):
    def __init__(self, rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, module_med=None):
        super(TwoPhaseTerm, self).__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")

        self.register_reactions("1st", rmat_1st, rmod_1st)
        self.register_reactions("2nd", rmat_2nd, rmod_2nd)
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni

    def register_reactions(self, postfix, rmat, rmod):
        setattr(self, f"asm_{postfix}", Assembler(rmat))
        setattr(self, f"rmod_{postfix}", rmod)

    def forward(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in, **params_extra)
        return self.asm_1st(y_in, rates_1st, den_norm) \
            + self.asm_2nd(y_in, rates_2nd, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in, **params_extra)
        return self.asm_1st.jacobain(y_in, rates_1st, den_norm) \
            + self.asm_2nd.jacobain(y_in, rates_2nd, den_norm)

    def compute_rates(self, t_in, y_in, **params_extra):
        if self.module_med is None:
            params_med = None
            den_norm = None
        else:
            params_med = self.module_med(t_in, **params_extra)
            den_norm = params_med['den_gas']
        rates_1st = self.rmod_1st(t_in, params_med)
        rates_2nd = self.rmod_2nd(t_in, params_med)
        return rates_1st, rates_2nd, den_norm

    def reproduce_reaction_rates(self, t_in=None):
        if t_in is None:
            t_in = torch.tensor([0.])

        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_1st) + len(inds_id_2nd)
        rates = torch.zeros([len(t_in), n_reac])
        with torch.no_grad():
            params_med = self.module_med(t_in)
            rates[:, inds_id_1st] = self.rmod_1st.compute_rates_reac(t_in, params_med)
            rates[:, inds_id_2nd] = self.rmod_2nd.compute_rates_reac(t_in, params_med)
        rates = rates.T.squeeze()
        return rates


class ThreePhaseTerm(nn.Module):
    def __init__(self, rmod_smt, rmat_smt,
                 rmod_1st, rmat_1st, rmat_1st_surf_gain, rmat_1st_surf_loss, rmat_1st_other,
                 rmod_2nd, rmat_2nd, rmat_2nd_surf_gain, rmat_2nd_surf_loss, rmat_2nd_other,
                 inds_surf, inds_mant, module_med=None):
        super().__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")
        #
        self.rmod_smt = rmod_smt
        self.asm_smt = Assembler(rmat_smt)
        # First order reactions
        self.rmod_1st = rmod_1st
        self.asm_1st = Assembler(rmat_1st)
        self.asm_1st_surf_gain = Assembler(rmat_1st_surf_gain)
        self.asm_1st_surf_loss = Assembler(rmat_1st_surf_loss)
        self.asm_1st_other = Assembler(rmat_1st_other)
        # Second order reactions
        self.rmod_2nd = rmod_2nd
        self.asm_2nd = Assembler(rmat_2nd)
        self.asm_2nd_surf_gain = Assembler(rmat_2nd_surf_gain)
        self.asm_2nd_surf_loss = Assembler(rmat_2nd_surf_loss)
        self.asm_2nd_other = Assembler(rmat_2nd_other)
        #
        self.register_buffer("inds_surf", torch.tensor(inds_surf))
        self.register_buffer("inds_mant", torch.tensor(inds_mant))
        #
        self.inds_id_smt = rmat_smt.inds_id_uni
        self.inds_id_1st = rmat_1st.inds_id_uni
        self.inds_id_2nd = rmat_2nd.inds_id_uni

    def forward(self, t_in, y_in, **params_extra):
        rates_smt, rates_1st, rates_2nd, den_norm = self.compute_rates(t_in, y_in)
        return self.asm_smt(y_in, rates_smt, den_norm) \
            + self.asm_1st(y_in, rates_1st, den_norm) \
            + self.asm_2nd(y_in, rates_2nd, den_norm)

    def jacobian(self, t_in, y_in, **params_extra):
        return jacrev(self, argnums=1)(t_in, y_in)

    def compute_rates(self, t_in, y_in, **params_extra):
        if self.module_med is None:
            params_med = None
            den_norm = None
        else:
            params_med = self.module_med(t_in, **params_extra)
            den_norm = params_med['den_gas']
        rates_1st = self.rmod_1st(t_in, params_med)
        rates_2nd = self.rmod_2nd(t_in, params_med)

        y_in = torch.atleast_2d(y_in)
        rates_1st = torch.atleast_2d(rates_1st)
        rates_2nd = torch.atleast_2d(rates_2nd)

        dy_1st_gain = self.asm_1st_surf_gain(y_in, rates_1st, den_norm)[:, self.inds_surf]
        dy_2nd_gain = self.asm_2nd_surf_gain(y_in, rates_2nd, den_norm)[:, self.inds_surf]
        dy_surf_gain = torch.sum(dy_1st_gain + dy_2nd_gain, dim=-1, keepdim=True)

        dy_1st_loss = self.asm_1st_surf_loss(y_in, rates_1st, den_norm)[:, self.inds_surf]
        dy_2nd_loss = self.asm_2nd_surf_loss(y_in, rates_2nd, den_norm)[:, self.inds_surf]
        dy_surf_loss = -torch.sum(dy_1st_loss + dy_2nd_loss, dim=-1, keepdim=True)

        rates_smt = self.rmod_smt(
            params_med, y_in, self.inds_surf, self.inds_mant, dy_surf_gain, dy_surf_loss,
        )
        return rates_smt, rates_1st, rates_2nd, den_norm

    def reproduce_reaction_rates(self, t_in, y_in):
        inds_id_smt = self.inds_id_smt
        inds_id_1st = self.inds_id_1st
        inds_id_2nd = self.inds_id_2nd
        n_reac = len(inds_id_smt) + len(inds_id_1st) + len(inds_id_2nd)
        rates = torch.zeros([len(t_in), n_reac])
        with torch.no_grad():
            rates_smt, rates_1st, rates_2nd, _ = self.compute_rates(t_in, y_in)
            rates[:, inds_id_smt] = rates_smt
            rates[:, inds_id_1st] = rates_1st[:, self.rmod_1st.inds_reac]
            rates[:, inds_id_2nd] = rates_2nd[:, self.rmod_2nd.inds_reac]
        rates = rates.T.squeeze()
        return rates


@dataclass(frozen=True)
class AstrochemProblem:
    spec_table: pd.DataFrame
    rmat_1st: ReactionMatrix
    rmat_2nd: ReactionMatrix
    reaction_term: nn.Module
    ab_0: torch.Tensor # Initial abundances

    def __repr__(self):
        return "Attributes: spec_table, rmat_1st, rmat_2nd, reaction_term, ab_0"


def create_astrochem_problem(df_reac, params_med, ab_0, spec_table_base=None, ab_0_min=0.):
    meta_params = MetaParameters()
    #
    spec_table, rmat_1st, rmat_2nd = create_reaction_data(
        df_reac["reactant_1"], df_reac["reactant_2"], df_reac["products"], spec_table_base)
    formulae = df_reac["formula"].values.astype(str)
    # First order reactions
    params_reac = data_frame_to_tensor_dict(
        df_reac[["is_unique", "T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_1st.inds],
    )
    rate_1st = create_gas_reaction_module_1st(
        formulae[rmat_1st.inds], rmat_1st, params_med, params_reac, meta_params)
    # Second order reactions
    params_reac = data_frame_to_tensor_dict(
        df_reac[["is_unique", "T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_2nd.inds])
    rate_2nd = create_gas_reaction_module_2nd(
        formulae[rmat_2nd.inds], rmat_2nd, params_med, params_reac, meta_params)
    #
    reaction_term = ReactionTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd)
    #
    ab_0_ = dervie_initial_abundances(ab_0, spec_table, meta_params, ab_0_min)

    return AstrochemProblem(spec_table, rmat_1st, rmat_2nd, reaction_term, ab_0_)


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


def compute_reaction_rates(problem, t_0=None):
    """Compute the reaction rates that match the input data frame."""
    if t_0 is None:
        t_0 = torch.tensor([0.])
    with torch.no_grad():
        inds_uni_1, to_uni_1 = np.unique(problem.rmat_1st.inds, return_index=True)
        rates_1 = problem.reaction_term.rate_1(t_0)[to_uni_1].numpy()
        inds_uni_2, to_uni_2 = np.unique(problem.rmat_2nd.inds, return_index=True)
        rates_2 = problem.reaction_term.rate_2(t_0)[to_uni_2].numpy()
    n_rate = max(max(inds_uni_1), max(inds_uni_2)) + 1
    # TODO: Consider the cases for multiple times.
    rates = np.empty(n_rate, dtype=np.float32)
    rates[inds_uni_1] = rates_1
    rates[inds_uni_2] = rates_2
    return rates