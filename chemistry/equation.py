import numpy as np
import pandas as pd
import torch
from torch import nn

from dataclasses import dataclass

from .utils import data_frame_to_tensor_dict
from .meta_params import MetaParameters
from .reaction_matrices import ReactionMatrix, create_reaction_data
from .gas_reactions import create_gas_reaction_module_1st, create_gas_reaction_module_2nd


class ReactionTerm(nn.Module):
    def __init__(self, rmat_1st, rate_1st, rmat_2nd, rate_2nd):
        super(ReactionTerm, self).__init__()
        self.register_buffer("inds_1r", torch.tensor(rmat_1st.spec_r))
        self.register_buffer("inds_1p", torch.tensor(rmat_1st.spec_p))
        self.rate_1 = rate_1st

        self.register_buffer("inds_2r", torch.tensor(rmat_2nd.spec_r)) # (N, 2)
        self.register_buffer("inds_2p", torch.tensor(rmat_2nd.spec_p))
        self.rate_2 = rate_2nd

    def forward(self, t_in, y_in):
        y_out = torch.zeros_like(y_in)
        if y_in.dim() == 1:
            term_1 = y_in[self.inds_1r]*self.rate_1(t_in)
            y_out.scatter_add_(0, self.inds_1p, term_1)
            term_2 = y_in[self.inds_2r].prod(dim=-1)*self.rate_2(t_in)
            y_out.scatter_add_(0, self.inds_2p, term_2)
        else:
            batch_size = y_in.shape[0]
            inds_1p = self.inds_1p.repeat(batch_size, 1)
            inds_2p = self.inds_2p.repeat(batch_size, 1)
            term_1 = y_in[:, self.inds_1r]*self.rate_1(t_in)
            y_out.scatter_add_(1, inds_1p, term_1)
            term_2 = y_in[:, self.inds_2r].prod(dim=-1)*self.rate_2(t_in)
            y_out.scatter_add_(1, inds_2p, term_2)
        return y_out


@dataclass(frozen=True)
class AstrochemProblem:
    spec_table: pd.DataFrame
    rmat_1st: ReactionMatrix
    rmat_2nd: ReactionMatrix
    reaction_term: nn.Module
    ab_0: torch.Tensor # Initial abundances

    def __repr__(self):
        return "Attributes: spec_table, rmat_1st, rmat_2nd, reaction_term, ab_0"


def create_astrochem_problem(df_reac, params_env, ab_0, ab_0_dtype='list'):
    meta_params = MetaParameters()
    #
    spec_table, rmat_1st, rmat_2nd = create_reaction_data(df_reac["reactants"], df_reac["products"])
    formulae = df_reac["formula"].values.astype(str)
    # First order reactions
    params_reac = data_frame_to_tensor_dict(
        df_reac[["is_unique", "T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_1st.inds])
    rate_1st = create_gas_reaction_module_1st(
        formulae[rmat_1st.inds], rmat_1st, params_env, params_reac, meta_params)
    # Second order reactions
    params_reac = data_frame_to_tensor_dict(
        df_reac[["is_unique", "T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_2nd.inds])
    rate_2nd = create_gas_reaction_module_2nd(
        formulae[rmat_2nd.inds], rmat_2nd, params_env, params_reac, meta_params)
    #
    reaction_term = ReactionTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd)
    # Initial condition
    ab_0 = ab_0.copy()
    ab_0_ = [0.]*len(spec_table)
    for idx, spec in enumerate(spec_table.index):
        if spec in ab_0:
            ab_0_[idx] = ab_0.pop(spec)
        elif spec == "GRAIN0":
            ab_0_[idx] = meta_params.grain_ab_0
    if len(ab_0) != 0:
        raise ValueError("Invalid initial abundances.")

    if ab_0_dtype == 'list':
        pass
    elif ab_0_dtype == 'numpy':
        ab_0_ = np.asarray(ab_0_)
    elif ab_0_dtype == 'torch':
        ab_0_ = torch.tensor(ab_0_)

    return AstrochemProblem(spec_table, rmat_1st, rmat_2nd, reaction_term, ab_0_)


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