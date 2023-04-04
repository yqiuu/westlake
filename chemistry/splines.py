import torch
from torch import nn


class QuadraticSpline(nn.Module):
    def __init__(self, n_dim, n_bin):
        super(QuadraticSpline, self).__init__()
        degree = 2
        xl_base = torch.linspace(0., 1., degree + 1)
        self.register_buffer("coeff", compute_coeff(xl_base))
        self.register_buffer("x_node", torch.linspace(0, 1, n_bin + 1))
        self.n_dim = n_dim
        self.n_bin = n_bin
        self.eps_min = 1e-6
        self.eps_max = 1 - self.eps_min

    @property
    def params_shape(self):
        return self.n_dim, 2*self.n_bin + 1

    def forward(self, x_in, x_node, f_params):
        # x_in (B, D)
        # x_node (D, N + 1)
        # f_params (D, 2*N + 1) or (B, D, 2*N + 1)
        batch_size = x_in.shape[0]
        x_in = torch.clamp(x_in, self.eps_min, self.eps_max)

        x_node = x_node.repeat(batch_size, 1, 1) # (B, D, N + 1)
        inds_1 = torch.searchsorted(x_node, x_in[..., None].contiguous())
        inds_0 = inds_1 - 1

        xn_0 = torch.gather(x_node, 2, inds_0).squeeze(dim=2)
        xn_1 = torch.gather(x_node, 2, inds_1).squeeze(dim=2)
        x_local = (x_in - xn_0)/(xn_1 - xn_0)

        f_params = self._derive_f_params(f_params[..., None], batch_size)
        weights = gather(f_params, inds_0.squeeze(dim=2))
        f_out = basis_function(x_local, self.coeff)
        print(f_out.shape, weights.shape)
        f_out = torch.sum(f_out*weights, dim=-1)

        return f_out, x_local

    def _derive_f_params(self, f_params, batch_size):
        # f_params (B, D, 2*N + 1)
        if f_params.dim() != 4:
            f_params = f_params.repeat(batch_size, 1, 1, 1)
        f_node, f_0 = torch.split(f_params, (self.n_bin + 1, self.n_bin), dim=-2)
        f_params = torch.cat([f_node[..., :-1, :], f_0, f_node[..., 1:, :]], dim=-1)
        return f_params


def basis_function(x_in, coeff):
    xl_base = base_vector(x_in, len(coeff) - 1)
    return torch.matmul(xl_base, coeff)


def compute_coeff(xl_base):
    mat = base_vector(xl_base, len(xl_base) - 1).squeeze(dim=1)
    coeff = torch.linalg.inv(mat)
    return coeff


def base_vector(x_in, degree):
    # x_in (B, D)
    if len(x_in.shape) == 1:
        x_in = x_in.reshape(-1, 1)
    vec = x_in[..., None].repeat_interleave(degree + 1, dim=-1)
    vec[..., 0] = 1.
    if degree == 1:
        return vec
    return torch.cumprod(vec, dim=-1)


def gather(data, inds):
    # data (B, D, N, X)
    # inds (B, D)
    n_0, n_1, n_2, _ = data.shape
    inds_0 = torch.arange(n_0, device=inds.device).view(-1, 1)
    inds_1 = torch.arange(n_1, device=inds.device).view(1, -1)
    inds = inds + (inds_0*n_1 + inds_1)*n_2
    data_out = data.view(-1, data.shape[-1])[inds.ravel()]
    data_out = data_out.reshape(*inds.shape, data.shape[-1])
    return data_out