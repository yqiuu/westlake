import numpy as np
import torch
from torch.func import jacrev
from scipy.integrate import solve_ivp


def solve_rate_equation(reaction_term, t_span, ab_0, method="LSODA",
                        rtol=1e-4, atol=1e-20, t_eval=None, u_factor=1.,
                        use_auto_jac=False, device="cpu", show_progress=True):
    """Solve the rate equations.

    Args:
        reaction_term (nn.Module): Definition of the rate equations.
        t_span (tuple): (t_start, t_end). Time range to solve the equations
        ab_0 (array): Initial abundances.
        method (str, optional): ODE solver. Defaults to "LSODA".
        rtol (float, optional): Relative tolerance. Defaults to 1e-4.
        atol (float, optional): Absolute tolerance. Defaults to 1e-20.
        t_eval (array, optional): Time grid for the solution if this is given;
            otherwise use the points obtaiend by the solver. Defaults to None.
        u_factor (float, optional): Unit factor that is mutiplied by any input
            time variables such as `t_span`. Defaults to 1.
        use_auto_jac (bool, optional): If True, use `jacrev` in `torch` to
            compute jacobian.
        device (str, optional): The device for `reaction_term`.
            Defaults to "cpu".
        show_progress: If True, print messages to show the progress.
            Defaults to True.
    Returns:
        object: A result object returned by a scipy ODE solver.
    """
    reaction_term.to(device)
    dtype = torch.get_default_dtype()

    def wrapper(t_in, y_in):
        if show_progress:
            percent = (t_in - t_span[0])/(t_span[1] - t_span[0])*100
            t_show = t_in/u_factor
            print("\r[{:5.1f}%] t = {:<12.5e}".format(percent, t_show), end='')

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

    if use_auto_jac:
        jacobian = jacrev(reaction_term, argnums=1)

        def wrapper_jac(t_in, y_in):
            t_in = torch.tensor(t_in, dtype=dtype, device=device)
            t_in = torch.atleast_1d(t_in)
            y_in = torch.tensor(y_in, dtype=dtype, device=device)
            y_in = torch.atleast_1d(y_in)
            jac_out = jacobian(t_in, y_in)
            return jac_out.cpu().numpy()
    else:
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
        t_show = t_span[1]/u_factor
        print("\r[100.0%] t = {:<12.5e}".format(t_show))
    return res


class Result:
    def __init__(self, res, df_spec):
        self._message = res.message
        self._success = res.success
        self._time = res.t
        self._ab = res.y
        self._species = {key: idx for idx, key in enumerate(df_spec.index)}

    def __repr__(self):
        return f"message: {self._message}\nsuccess: {self._success}"

    def __getitem__(self, key):
        if key == "time":
            return self._time
        else:
            return self._ab[self._species[key]]

    @property
    def message(self):
        return self._message

    @property
    def success(self):
        return self._success

    @property
    def species(self):
        """List of all species."""
        return tuple(self._species.keys())

    @property
    def ab(self):
        """Abundance data."""
        return self._ab

    def last(self):
        return self._ab[:, -1]

    def append(self, res):
        self._time = np.append(self._time, res.t[1:])
        self._ab = np.concatenate([self._ab, res.y[:, 1:]], axis=-1)