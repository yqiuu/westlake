import torch
from torch import nn


class Assembler(nn.Module):
    def __init__(self, rmat):
        super(Assembler, self).__init__()
        if rmat.rate_sign is None:
            # If the rates are constant, the rate signs are included in the rates.
            # Therefore, the rate_signs are unnecessary in this case.
            self.inds_k = None
        else:
            self.register_buffer("inds_k", torch.tensor(rmat.inds_k))
            self.register_buffer(
                "rate_sign", torch.tensor(rmat.rate_sign, dtype=torch.get_default_dtype()))
        # Save indices for assembling equations.
        self.register_buffer("inds_r", torch.tensor(rmat.inds_r))
        self.register_buffer("inds_p", torch.tensor(rmat.inds_p))
        # Save indices for computing jacobian.
        if rmat.order == 1:
            inds_pr = self.inds_p*rmat.n_spec + self.inds_r
        else:
            inds_pr = torch.ravel(
                self.inds_p[:, None]*rmat.n_spec + self.inds_r[:, [1, 0]])
        self.register_buffer("inds_pr", inds_pr)
        #
        self.order = rmat.order

    def forward(self, y_in, rates, den_norm):
        rates = self.assemble_rates(y_in, rates, den_norm)
        y_out = torch.zeros_like(y_in)
        if y_in.dim() == 1:
            terms = y_in[self.inds_r]
            if self.order != 1:
                terms = terms.prod(dim=-1)
            terms = terms*rates.squeeze()
            y_out.scatter_add_(0, self.inds_p, terms)
        else:
            batch_size = y_in.shape[0]
            inds_p = self.inds_p.repeat(batch_size, 1)
            terms = y_in[:, self.inds_r]
            if self.order != 1:
                terms = terms.prod(dim=-1)
            terms = terms*rates
            y_out.scatter_add_(1, inds_p, terms)
        return y_out

    def jacobain(self, y_in, rates, den_norm):
        rates = self.assemble_rates(y_in, rates, den_norm)
        n_spec = y_in.shape[-1]
        jac = torch.zeros(n_spec*n_spec, dtype=y_in.dtype, device=y_in.device)
        if self.order == 1:
            terms = rates.squeeze()
        else:
            terms = y_in[self.inds_r]*rates.view(-1, 1)
            terms = terms.ravel()
        jac.scatter_add_(0, self.inds_pr, terms)
        jac = jac.reshape(n_spec, n_spec)
        return jac

    def assemble_rates(self, y_in, rates, den_norm):
        if self.order == 2 and den_norm is not None:
            rates = rates*den_norm
        if self.inds_k is not None:
            rates = rates[:, self.inds_k]*self.rate_sign
        return rates