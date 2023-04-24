import numpy as np
import torch
from torch import nn

from .utils import data_frame_to_tensor_dict
from .meta_params import MetaParameters
from .reaction_matrices import create_reaction_data
from .gas_reactions import create_gas_reactions_1st, create_gas_reactions_2nd


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


def create_problem(df_reac, params_env, ab_0, ab_0_dtype='list'):
    spec_table, rmat_1st, rmat_2nd = create_reaction_data(df_reac["reactants"], df_reac["products"])
    formulae = df_reac["formula"].values.astype(str)
    #
    meta_params = MetaParameters()
    # First order reactions
    params_reac = data_frame_to_tensor_dict(df_reac[["T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_1st.inds])
    rate_1st = create_gas_reactions_1st(formulae[rmat_1st.inds], rmat_1st, params_env, params_reac, meta_params)
    # Second order reactions
    params_reac = data_frame_to_tensor_dict(df_reac[["T_min", "T_max", "alpha", "beta", "gamma"]].iloc[rmat_2nd.inds])
    rate_2nd = create_gas_reactions_2nd(formulae[rmat_2nd.inds], rmat_2nd, params_env, params_reac, meta_params)
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

    return ReactionTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd), ab_0_, spec_table