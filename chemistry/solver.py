import torch
from torch.autograd.functional import jacobian
from scipy.integrate import solve_ivp


def solve_kinetic(reaction_term, t_span, ab_0, meta_params, t_eval=None, device='cpu', **kwargs):
    reaction_term.to(device)
    dtype = torch.get_default_dtype()

    def wrapper(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=dtype, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=dtype, device=device)
        y_in = torch.atleast_1d(y_in)
        if y_in.ndim == 2:
            y_in = y_in.T
        y_out = reaction_term(t_in, y_in)
        if y_in.ndim == 2:
            y_out = y_out.T
        return y_out.cpu().numpy()

    def wrapper_jac(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=dtype, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=dtype, device=device)
        y_in = torch.atleast_1d(y_in)
        jac_out = jacobian(lambda y_in: reaction_term(t_in, y_in), y_in)
        return jac_out.cpu().numpy()

    kwargs_ = {
        "t_eval": t_eval,
        "rtol": meta_params.rtol,
        "atol": meta_params.atol,
        "method": "BDF",
        "vectorized": True,
    }
    kwargs_.update(**kwargs)
    return solve_ivp(wrapper, t_span, ab_0, jac=wrapper_jac, **kwargs_)
