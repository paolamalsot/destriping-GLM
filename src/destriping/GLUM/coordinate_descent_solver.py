"""Coordinate descent solver for the destriping GLM.

Alternates between:
  - One L-BFGS iteration on (intercept, h, w) — using glum's tabmat-based
    objective / gradient (``_get_obj_and_derivative``)
  - One analytical c update given (intercept, h, w) — via tabmat
    ``CategoricalMatrix`` for fast group-sum aggregation and broadcast.

Uses scipy's Fortran L-BFGS-B engine directly so we can insert the c update
between iterations.  The objective/gradient evaluation reuses glum's C-optimised
code through tabmat, so h/w are treated exactly as in the original solver.
"""

import functools
import warnings
import logging

import numpy as np
from numpy import float64, zeros, array
from scipy.optimize import _lbfgsb

from glum._solvers import _get_obj_and_derivative

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# c update helpers
#
# Use tabmat CategoricalMatrix (shape n_obs × n_p) for group operations:
#   P_cat.transpose_matvec(v, out=buf)  →  buf = P.T @ v  (group sums)
#   P_cat.matvec(v, out=buf)            →  buf = P   @ v  (broadcast)
#
# NOTE: transpose_matvec ACCUMULATES into out, so buf must be zeroed first.
# ---------------------------------------------------------------------------

def _update_c_poisson(coef, X, offset, P_cat, n_p, y, log_c, _buf1, _buf2, P2_c=None):
    """Closed-form  c_p = sum(k) / sum(h·w)  for Poisson."""
    if P2_c is not None and np.any(P2_c > 0):
        raise NotImplementedError(
            "Poisson closed-form c update not supported with L2 penalty on c."
        )
    eta_hw = coef[0] + X @ coef[1:]       # tabmat matvec, O(n)
    hw = np.exp(eta_hw)

    _buf1[:] = 0.0
    P_cat.transpose_matvec(y, out=_buf1)   # sum_k per group
    _buf2[:] = 0.0
    P_cat.transpose_matvec(hw, out=_buf2)  # sum_hw per group

    valid = _buf2 > 0
    log_c[valid] = np.log(np.maximum(_buf1[valid], 1e-300)) - np.log(_buf2[valid])
    offset[:] = 0.0
    P_cat.matvec(log_c, out=offset)


def _update_c_nb(coef, X, offset, P_cat, n_p, y, theta, log_c, _buf1, _buf2, P2_c=None):
    """One Newton step on log(c_p) for Negative Binomial."""
    eta_hw = coef[0] + X @ coef[1:]
    mu = np.exp(eta_hw + offset)

    score = (mu - y) / (1.0 + theta * mu)
    _buf1[:] = 0.0
    P_cat.transpose_matvec(score, out=_buf1)  # grad_c
    if P2_c is not None:
        _buf1 += P2_c * log_c  # L2 penalty gradient

    hess_w = mu * (1.0 + theta * y) / (1.0 + theta * mu) ** 2
    _buf2[:] = 0.0
    P_cat.transpose_matvec(hess_w, out=_buf2)  # hess_c
    if P2_c is not None:
        _buf2 += P2_c  # L2 penalty Hessian

    valid = _buf2 > 1e-300
    log_c[valid] -= _buf1[valid] / _buf2[valid]
    offset[:] = 0.0
    P_cat.matvec(log_c, out=offset)


# ---------------------------------------------------------------------------
# Main solver  (same signature as glum._solvers._lbfgs_solver + c-update args)
# ---------------------------------------------------------------------------

def coordinate_descent_lbfgs_solver(
    coef,
    X,
    y,
    sample_weight,
    P2,
    verbose,
    family,
    link,
    max_iter=100,
    tol=1e-4,
    offset=None,
    *,
    p_idx,
    n_p,
    log_c,
    P_cat,
    theta=None,
    n_c_updates=1,
    P2_c=None,
    factr=1e2,
    maxcor=10,
    maxls=20,
):
    """L-BFGS on (intercept, h, w) with c updated after each iteration.

    Parameters match ``glum._solvers._lbfgs_solver`` with extra keyword-only
    arguments for the c update.  *offset* and *log_c* are **mutated in place**.

    P_cat : tabmat.CategoricalMatrix
        One-hot indicator matrix (n_obs × n_p) for the p grouping variable.
        Used for fast group-sum aggregation and broadcast via optimised
        ``transpose_matvec`` / ``matvec``.
    theta : float or None
        Negative-Binomial dispersion. ``None`` selects the Poisson closed-form
        c update; a numeric value selects one NB Newton step.
    n_c_updates : int
        Number of c-update steps per L-BFGS iteration (default 1).

    Returns ``(coef, n_iter, -1, None)`` — same shape as ``_lbfgs_solver``.
    """
    # --- objective / gradient using glum's tabmat code --------------------
    func = functools.partial(
        _get_obj_and_derivative,
        X=X,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        family=family,
        link=link,
        offset=offset,          # captured by reference → sees in-place updates
    )

    # --- pre-allocate buffers for c-update aggregation --------------------
    _buf1 = zeros(n_p, float64)
    _buf2 = zeros(n_p, float64)

    # --- replicate scipy fmin_l_bfgs_b loop with setulb -------------------
    n = len(coef)
    m = maxcor
    x = np.array(coef, dtype=float64)

    fortran_int = _lbfgsb.types.intvar.dtype
    nbd = zeros(n, fortran_int)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)

    f = array(0.0, float64)
    g = zeros(n, float64)
    wa = zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, float64)
    iwa = zeros(3 * n, fortran_int)
    task = zeros(1, "S60")
    csave = zeros(1, "S60")
    lsave = zeros(4, fortran_int)
    isave = zeros(44, fortran_int)
    dsave = zeros(29, float64)

    iprint = 1 if verbose else -1
    task[:] = "START"
    n_iterations = 0

    while True:
        _lbfgsb.setulb(
            m, x, low_bnd, upper_bnd, nbd, f, g,
            factr, tol, wa, iwa, task, iprint,
            csave, lsave, isave, dsave, maxls,
        )
        task_str = task.tobytes()

        if task_str.startswith(b"FG"):
            obj, grad = func(x)
            f[()] = obj
            g[:] = grad

        elif task_str.startswith(b"NEW_X"):
            n_iterations += 1

            # ---- c update (mutates offset in place) ----------------------
            for _ in range(n_c_updates):
                if theta is None:
                    _update_c_poisson(x, X, offset, P_cat, n_p, y, log_c, _buf1, _buf2, P2_c=P2_c)
                else:
                    _update_c_nb(x, X, offset, P_cat, n_p, y, theta, log_c, _buf1, _buf2, P2_c=P2_c)

            if n_iterations >= max_iter:
                task[:] = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        else:
            break

    task_str = task.tobytes().strip(b"\x00").strip()
    converged = task_str.startswith(b"CONV")

    if not converged and n_iterations >= max_iter:
        warnings.warn(
            "Coordinate descent did not converge. Increase max_iter.",
            stacklevel=2,
        )

    return x, n_iterations, -1, None