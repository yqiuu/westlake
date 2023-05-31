import numpy as np
import pandas as pd
import torch
from torch import nn

from dataclasses import dataclass

from .utils import TensorDict, data_frame_to_tensor_dict
from .meta_params import MetaParameters
from .reaction_matrices import ReactionMatrix


class ReactionTerm(nn.Module):
    def __init__(self, rmat_1st, rmod_1st, rmat_2nd, rmod_2nd, module_med=None):
        super(ReactionTerm, self).__init__()
        if module_med is None \
            or isinstance(module_med, TensorDict) or isinstance(module_med, nn.Module):
            self.module_med = module_med
        else:
            raise ValueError("Unknown 'module_med'.")

        self.rmod_1st = rmod_1st
        self.rmod_2nd = rmod_2nd

        # Save indices for assembling equations.
        self.register_buffer("inds_1r", torch.tensor(rmat_1st.inds_r))
        self.register_buffer("inds_1p", torch.tensor(rmat_1st.inds_p))
        self.register_buffer("inds_2r", torch.tensor(rmat_2nd.inds_r)) # (N, 2)
        self.register_buffer("inds_2p", torch.tensor(rmat_2nd.inds_p))

        # Save indices for computing jacobian.
        n_spec = max(max(rmat_1st.inds_p), max(rmat_2nd.inds_p)) + 1
        self.register_buffer("inds_1pr", self.inds_1p*n_spec + self.inds_1r)
        self.register_buffer(
            "inds_2pr", torch.ravel(self.inds_2p[:, None]*n_spec + self.inds_2r[:, [1, 0]]))

    def compute_rates(self, t_in, y_in, **params_extra):
        if self.module_med is None:
            params_med = None
        else:
            params_med = self.module_med(t_in, **params_extra)
        rates_1st = self.rmod_1st(t_in, params_med)
        rates_2nd = self.rmod_2nd(t_in, params_med)
        return rates_1st, rates_2nd

    def forward(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd = self.compute_rates(t_in, y_in, **params_extra)
        return self.assemble_equation(y_in, rates_1st, rates_2nd)

    def assemble_equation(self, y_in, rates_1st, rates_2nd):
        y_out = torch.zeros_like(y_in)
        if y_in.dim() == 1:
            term_1 = y_in[self.inds_1r]*rates_1st.squeeze()
            y_out.scatter_add_(0, self.inds_1p, term_1)
            term_2 = y_in[self.inds_2r].prod(dim=-1)*rates_2nd.squeeze()
            y_out.scatter_add_(0, self.inds_2p, term_2)
        else:
            batch_size = y_in.shape[0]
            inds_1p = self.inds_1p.repeat(batch_size, 1)
            inds_2p = self.inds_2p.repeat(batch_size, 1)
            term_1 = y_in[:, self.inds_1r]*rates_1st
            y_out.scatter_add_(1, inds_1p, term_1)
            term_2 = y_in[:, self.inds_2r].prod(dim=-1)*rates_2nd
            y_out.scatter_add_(1, inds_2p, term_2)
        return y_out

    def jacobian(self, t_in, y_in, **params_extra):
        rates_1st, rates_2nd = self.compute_rates(t_in, y_in, **params_extra)
        return self.assemble_jacobian(y_in, rates_1st, rates_2nd)

    def assemble_jacobian(self, y_in, rates_1st, rates_2nd):
        n_spec = y_in.shape[-1]
        jac = torch.zeros(n_spec*n_spec, dtype=y_in.dtype, device=y_in.device)
        term_1 = rates_1st.squeeze()
        jac.scatter_add_(0, self.inds_1pr, term_1)
        term_2 = y_in[self.inds_2r]*rates_2nd.view(-1, 1)
        jac.scatter_add_(0, self.inds_2pr, term_2.ravel())
        jac = jac.reshape(n_spec, n_spec)
        return jac


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


def reproduce_reaction_rates(reaction_term, rmat_1st, rmat_2nd, t_in=None):
    if t_in is None:
        t_in = torch.tensor([0.])

    inds_reac_1st = reaction_term.rmod_1st.inds_reac
    inds_reac_2nd = reaction_term.rmod_2nd.inds_reac
    n_reac = len(inds_reac_1st) + len(inds_reac_2nd)
    rates = torch.zeros([len(t_in), n_reac])
    with torch.no_grad():
        params_med = reaction_term.module_med(t_in)
        rates[:, rmat_1st.inds_id] = reaction_term.rmod_1st.compute_rates_reac(t_in, params_med)
        rates[:, rmat_2nd.inds_id] = reaction_term.rmod_2nd.compute_rates_reac(t_in, params_med)
    rates = rates.T.squeeze()
    return rates


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