# The code below is derived from scipy under the following license
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math

import numpy as np
from scipy.integrate._ivp.common import (
    validate_max_step, validate_tol,
    EPS, validate_first_step
)
import torch


MAX_ORDER = 5
NEWTON_MAXITER = 4
MIN_FACTOR = 0.2
MAX_FACTOR = 10
MSG_SUCCESS = "The solver successfully reached the end of the integration interval."
MSG_FAIL = "Required step size is less than spacing between numbers."


def compute_R(order, factor, device):
    """Compute the matrix for changing the differences array."""
    I = torch.arange(1, order + 1, device=device)[:, None]
    J = torch.arange(1, order + 1, device=device)
    M = torch.zeros((order + 1, order + 1), device=device)
    M[1:, 1:] = (I - 1 - factor * J) / I
    M[0] = 1
    return torch.cumprod(M, dim=0)


def change_D(D, order, factor, device):
    """Change differences array in-place when step size is changed."""
    R = compute_R(order, factor, device)
    U = compute_R(order, 1, device)
    RU = torch.matmul(R, U)
    D[:order + 1] = torch.matmul(RU.T, D[:order + 1])


def solve_bdf_system(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol):
    """Solve the algebraic system resulting from BDF method."""
    d = 0
    y = y_predict.clone()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y)
        if not torch.all(torch.isfinite(f)):
            break

        dy = solve_lu(LU, c * f - psi - d)
        #inds = torch.argsort(dy)[::-1]
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


def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return math.inf

    scale = atol + torch.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    d1 = d1.cpu().numpy()
    d2 = d2.cpu().numpy()
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


def norm(x):
    return torch.linalg.norm(x)/math.sqrt(len(x))


class BDF:
    def __init__(self, fun, jac, t0, y0, first_step=None, max_step=math.inf,
                 rtol=1e-4, atol=1e-20, device="cpu"):
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        def fun_wrapped(t, y):
            t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
            t = torch.atleast_1d(t)
            if y.ndim == 1:
                self.nfev += 1
            else:
                y = y.T
                self.nfev += y.shape[0]
            f = fun(t, y)
            if f.ndim == 2:
                f = f.T
            return f

        def jac_wrapped(t, y):
            t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
            t = torch.atleast_1d(t)
            self.njev += 1
            return jac(t, y)

        self.fun = fun_wrapped
        self.jac = jac_wrapped

        y0 = torch.tensor(y0, dtype=torch.get_default_dtype(), device=device)
        self.t = t0
        dydt = fun_wrapped(t0, y0)
        self.J = jac_wrapped(t0, y0)
        self.direction = 1
        n_y = len(y0)
        self.rtol, self.atol = validate_tol(rtol, atol, n_y)
        self.atol = torch.tensor(self.atol, device=device)
        self.device = device

        self.max_step = validate_max_step(max_step)
        self.min_step = 10*abs(np.nextafter(0., self.direction * math.inf))
        if first_step is None:
            self.h_abs = select_initial_step(
                fun_wrapped, t0, y0, dydt, self.direction, 1, rtol, atol
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, math.inf)

        self.h_abs_old = None
        self.error_norm_old = None
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))

        def lu(A):
            self.nlu += 1
            return torch.linalg.lu_factor(A)

        def solve_lu(LU, b):
            return torch.linalg.lu_solve(LU.LU, LU.pivots, b[:, None])[:, 0]

        I = torch.eye(n_y, device=device)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        kappa = torch.tensor([0, -0.1850, -1/9, -0.0823, -0.0415, 0], device=device)
        self.gamma = torch.concat((
            torch.zeros(1, device=device),
            torch.cumsum(1/torch.arange(1, MAX_ORDER + 1, device=device), dim=0)
        ))
        self.alpha = (1 - kappa)*self.gamma
        self.error_const = kappa*self.gamma + 1/torch.arange(1, MAX_ORDER + 2, device=device)

        D = torch.empty((MAX_ORDER + 3, n_y), dtype=torch.get_default_dtype(), device=device)
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
        min_step = self.min_step
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs, self.device)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs, self.device)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs

        atol = self.atol
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
                return False, MSG_FAIL

            if isinstance(h_abs, torch.Tensor):
                h_abs = h_abs.item()

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - t_bound) > 0:
                t_new = t_bound
                h = t_new - t
                change_D(D, order, abs(t_new - t)/h_abs, device=self.device)
                self.n_equal_steps = 0
                LU = None

            h_abs = np.abs(h)
            y_predict = torch.sum(D[:order + 1], dim=0)

            scale = atol + rtol * torch.abs(y_predict)
            psi = torch.matmul(D[1: order + 1].T, gamma[1: order + 1])/alpha[order]

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
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor, self.device)
                self.n_equal_steps = 0
                LU = None
                continue

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                       + n_iter)

            scale = atol + rtol * torch.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)

            if error_norm > 1:
                factor = max(MIN_FACTOR,
                             safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor, self.device)
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
            return True, MSG_SUCCESS

        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
            error_m_norm = error_m_norm.item()
        else:
            error_m_norm = math.inf

        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
            error_p_norm = error_p_norm.item()
        else:
            error_p_norm = math.inf

        error_norm = error_norm.cpu().item()
        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))

        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order

        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor, device=self.device)
        self.n_equal_steps = 0
        self.LU = None

        return True, MSG_SUCCESS