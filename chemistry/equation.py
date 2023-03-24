import torch
from torch import nn


class KineticEquation(nn.Module):
    def __init__(self, rmat_1st, rmat_2nd):
        super(KineticEquation, self).__init__()
        self.register_buffer("inds_1r", torch.tensor(rmat_1st.spec_r))
        self.register_buffer("inds_1p", torch.tensor(rmat_1st.spec_p))
        self.register_buffer("rate_1", torch.tensor(rmat_1st.rate, dtype=torch.float32))
        
        self.register_buffer("inds_2r0", torch.tensor(rmat_2nd.spec_r[0]))
        self.register_buffer("inds_2r1", torch.tensor(rmat_2nd.spec_r[1]))
        self.register_buffer("inds_2p", torch.tensor(rmat_2nd.spec_p))
        self.register_buffer("rate_2", torch.tensor(rmat_2nd.rate, dtype=torch.float32))
        
    def forward(self, t_in, y_in):
        y_out = torch.zeros_like(y_in)
        if y_in.dim() == 1:
            y_out.scatter_add_(0, self.inds_1p, y_in[self.inds_1r]*self.rate_1)
            y_out.scatter_add_(0, self.inds_2p, y_in[self.inds_2r0]*y_in[self.inds_2r1]*self.rate_2)
        else:
            batch_size = y_in.shape[0]
            inds_1p = self.inds_1p.repeat(batch_size, 1)
            inds_2p = self.inds_2p.repeat(batch_size, 1)
            tmp = y_in[:, self.inds_1r]*self.rate_1
            y_out.scatter_add_(1, inds_1p, y_in[:, self.inds_1r]*self.rate_1)
            y_out.scatter_add_(1, inds_2p, y_in[:, self.inds_2r0]*y_in[:, self.inds_2r1]*self.rate_2)
        return y_out
