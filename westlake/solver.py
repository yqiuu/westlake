import torch
from scipy.integrate import solve_ivp


def solve_rate_equation(reaction_term, t_span, ab_0, method="LSODA", rtol=1e-4, atol=1e-20,
                        t_eval=None, u_factor=1., device="cpu", show_progress=True):
    reaction_term.to(device)
    dtype = torch.get_default_dtype()

    def wrapper(t_in, y_in):
        if show_progress:
            print(f"\r[{t_in/t_span[1]*100.:5.1f}%] t = {t_in/u_factor:<12.5e}", end='')

        t_in = torch.tensor(t_in, dtype=dtype, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=dtype, device=device)
        y_in = torch.atleast_1d(y_in)
        if y_in.ndim == 2:
            y_in = y_in.T
        y_out = reaction_term(t_in, y_in)
        if y_out.ndim == 2:
            y_out = y_out.T
        return y_out.cpu().numpy()

    def wrapper_jac(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=dtype, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=dtype, device=device)
        y_in = torch.atleast_1d(y_in)
        jac_out = reaction_term.jacobian(t_in, y_in)
        return jac_out.cpu().numpy()

    t_span = tuple(t*u_factor for t in t_span)
    if t_eval is not None:
        t_eval = t_eval*u_factor

    kwargs_ = {
        "t_eval": t_eval,
        "method": "BDF",
        "vectorized": True,
    }
    kwargs_.update(method=method, rtol=rtol, atol=atol)

    res = solve_ivp(wrapper, t_span, ab_0, jac=wrapper_jac, **kwargs_)
    res.t /= u_factor

    if show_progress:
        print(f"\r[{100.:.1f}%] t = {t_span[1]/u_factor:12.6e}")
    return res
