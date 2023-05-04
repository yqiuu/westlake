import torch
from torch.autograd.functional import jacobian
from scipy.integrate import solve_ivp


def solve_ode(target, t_span, y0, device='cpu', vectorized=True, **kwargs):
    def wrapper(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=torch.float32, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=torch.float32, device=device)
        y_in = torch.atleast_1d(y_in)
        if y_in.ndim == 2:
            y_in = y_in.T
        y_out = target(t_in, y_in)
        if y_in.ndim == 2:
            y_out = y_out.T
        return y_out.cpu().numpy()

    def wrapper_jac(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=torch.float32, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=torch.float32, device=device)
        y_in = torch.atleast_1d(y_in)
        jac_out = jacobian(lambda y_in: target(t_in, y_in), y_in)
        return jac_out.cpu().numpy()

    return solve_ivp(wrapper, t_span, y0, jac=wrapper_jac, vectorized=vectorized, **kwargs)
