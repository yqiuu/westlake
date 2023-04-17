import torch
from scipy.integrate import solve_ivp


def solve_ode(target, t_span, y0, vectorized=True, **kwargs):
    def wrapper(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=torch.float32)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=torch.float32)
        y_in = torch.atleast_1d(y_in)
        if y_in.ndim == 2:
            y_in = y_in.T
        y_out = target(t_in, y_in)
        if y_in.ndim == 2:
            y_out = y_out.T
        return y_out.numpy()

    return solve_ivp(wrapper, t_span, y0, vectorized=vectorized, **kwargs)
