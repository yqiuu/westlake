import torch
from torch import nn


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