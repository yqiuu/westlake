import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, csc_matrix, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, EPS, num_jac, validate_first_step,
                     warn_extraneous)
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
import torch


MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10


def solve(reaction_term, t_span, ab_0, method="BDF",
          rtol=1e-4, atol=1e-20, t_eval=None, u_factor=1.,
          use_auto_jac=False, device="cpu", show_progress=True):
    dtype = torch.get_default_dtype()

    def wrapper_fun(t_in, y_in):
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

    def wrapper_jac(t_in, y_in):
        t_in = torch.tensor(t_in, dtype=dtype, device=device)
        t_in = torch.atleast_1d(t_in)
        y_in = torch.tensor(y_in, dtype=dtype, device=device)
        y_in = torch.atleast_1d(y_in)
        jac_out = reaction_term.jacobian(t_in, y_in)
        return jac_out.cpu().numpy()

    t_span = tuple(t*u_factor for t in t_span)
    t_start, t_end = t_span

    t_ret = []
    y_ret = []
    solver = BDF(wrapper_fun, wrapper_jac, t_start, ab_0, rtol=rtol, atol=atol)
    while True:
        success, message = solver.step(t_end)
        t_new = solver.t
        y_new = solver.y
        t_ret.append(t_new)
        y_ret.append(y_new)
        if t_new >= t_end:
            break
    t_ret = np.array(t_ret)/u_factor
    y_ret = np.vstack(y_ret).T
    return t_ret, y_ret


def compute_R(order, factor):
    """Compute the matrix for changing the differences array."""
    I = np.arange(1, order + 1)[:, None]
    J = np.arange(1, order + 1)
    M = np.zeros((order + 1, order + 1))
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return np.cumprod(M, axis=0)


def change_D(D, order, factor):
    """Change differences array in-place when step size is changed."""
    R = compute_R(order, factor)
    U = compute_R(order, 1)
    RU = R.dot(U)
    D[:order + 1] = np.dot(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y)
        if not np.all(np.isfinite(f)):
            break

        dy = solve_lu(LU, c * f - psi - d)
        inds = np.argsort(dy)[::-1]
        dy_norm = norm(dy / scale)

        if dy_norm_old is None:
            rate = None
        else:
            rate = dy_norm / dy_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break

        y += dy
        d += dy

        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break

        dy_norm_old = dy_norm

    return converged, k + 1, y, d


class BDF(OdeSolver):
    """
    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    """
    def __init__(self, fun, jac, t0, y0, first_step=None, max_step=np.inf,
                 rtol=1e-4, atol=1e-20):
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        def fun_wrapped(t, y):
            if len(y.shape) == 1:
                self.nfev += 1
            else:
                self.nfev += y.shape[0]
            return fun(t, y)

        def jac_wrapped(t, y):
            self.njev += 1
            return jac(t, y)

        self.fun = fun_wrapped
        self.jac = jac_wrapped

        self.t = t0
        self.direction = 1
        dydt = fun_wrapped(t0, y0)
        self.J = jac_wrapped(t0, y0)
        n_y = len(y0)
        self.rtol, self.atol = validate_tol(rtol, atol, n_y)

        self.max_step = validate_max_step(max_step)
        if first_step is None:
            self.h_abs = select_initial_step(
                fun_wrapped, t0, y0, dydt, self.direction, 1, rtol, atol
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, np.inf)

        self.h_abs_old = None
        self.error_norm_old = None
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        def lu(A):
            self.nlu += 1
            return lu_factor(A, overwrite_a=True)

        def solve_lu(LU, b):
            return lu_solve(LU, b, overwrite_b=True)

        I = np.identity(n_y, dtype=y0.dtype)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        kappa = np.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
        self.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
        self.alpha = (1 - kappa) * self.gamma
        self.error_const = kappa * self.gamma + 1 / np.arange(1, MAX_ORDER + 2)

        D = np.empty((MAX_ORDER + 3, n_y), dtype=y0.dtype)
        D[0] = y0
        D[1] = dydt * self.h_abs * self.direction
        self.D = D

        self.order = 1
        self.n_equal_steps = 0
        self.LU = None

    def step(self, t_bound):
        t = self.t
        D = self.D

        max_step = self.max_step
        min_step = 2 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
        #atol = np.maximum(self.atol, self.rtol*np.abs(self.y))
        rtol = self.rtol
        order = self.order

        alpha = self.alpha
        gamma = self.gamma
        error_const = self.error_const

        J = self.J
        LU = self.LU
        current_jac = self.jac is None

        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - t_bound) > 0:
                t_new = t_bound
                h = t_new - t
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None

            #h = t_new - t
            h_abs = np.abs(h)

            y_predict = np.sum(D[:order + 1], axis=0)

            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]

            converged = False
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = self.lu(self.I - c * J)

                converged, n_iter, y_new, d = solve_bdf_system(
                    self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
                    scale, self.newton_tol)

                if not converged:
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True

            if not converged:
                #print("A")
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)

            if error_norm > 1:
                #idx = np.argmax(error)
                #print(idx, error[idx])

                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                # As we didn't have problems with convergence, we don't
                # reset LU here.
            else:
                step_accepted = True

        self.n_equal_steps += 1

        #y_new = np.maximum(y_new, self.rtol*self.atol)
        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.J = J
        self.LU = LU

        # Update differences. The principal relation here is
        # D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}. Keep in mind that D
        # contained difference for previous interpolating polynomial and
        # d = D^{k + 1} y_n. Thus this elegant code follows.
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]

        if self.n_equal_steps < order + 1:
            return True, None

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf

        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        #print("{:.4e}, {:.4e}".format(factor, self.h_abs))
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None

        return True, None