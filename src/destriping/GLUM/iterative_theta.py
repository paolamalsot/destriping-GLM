import numpy as np
import warnings
import pandas as pd
from src.destriping.GLUM.sol import mu_from_sol


def theta_md(
    y,
    mu,
    dfr,
    weights=None,
    limit=20,
    eps=np.finfo(float).eps ** 0.25,
    mu_clip=10 ** (-10),
    min_t0 = 10 ** (-10)
):
    """
    Python translation of the R function:

      theta.md <- function(y, mu, dfr, weights, limit = 20, eps = .Machine$double.eps^0.25) { ... }

    Estimates the theta parameter (dispersion parameter) for a Negative Binomial
    distribution using an iterative method using a method of moments.

    Args:
        y (np.ndarray): The response variable (observed counts), must be a numpy array.
        mu (np.ndarray): The fitted mean values for the response variable (E[y|x]), must be a numpy array.
        dfr (int): Degrees of freedom for residuals.
        weights (np.ndarray, optional): Observation weights. Defaults to an array of ones.
        limit (int, optional): Maximum number of iterations for the estimation. Defaults to 20.
        eps (float, optional): Tolerance for convergence. Defaults to numpy's machine epsilon raised to the power of 0.25.

    Returns:
        float: The estimated theta (t0) parameter.
    """

    mu = np.clip(mu, a_min=mu_clip, a_max = None)

    if weights is None:
        weights = np.ones_like(y, dtype=float)

    # Ensure all inputs are numpy arrays for consistent operations
    y = np.asarray(y)
    mu = np.asarray(mu)
    weights = np.asarray(weights)

    n = np.sum(weights)
    denom = np.sum(weights * (y / mu - 1.0) ** 2)
    # Guard against zero denominator
    if denom <= 0:
        raise ZeroDivisionError("Initial denominator is non-positive; check inputs.")
    # Initial estimate of theta (t0) using the method of moments
    t0 = n / denom

    # a = 2 * sum(weights * y * log(pmax(1, y) / mu)) - dfr
    # pmax(1, y) ensures log is taken of at least 1 (R's behavior to handle y=0)
    log_y_over_mu = np.log(np.maximum(1, y) / mu)
    a = 2 * np.sum(weights * y * log_y_over_mu) - dfr

    it = 0
    delta = 1.0

    # Newton iteration
    while (it := it + 1) < limit and abs(delta) > eps:
        t0 = abs(t0)
        tmp = np.log((y + t0) / (mu + t0))

        top = a - 2.0 * np.sum(weights * (y + t0) * tmp)
        bot = 2.0 * np.sum(weights * (((y - mu) / (mu + t0)) - tmp))

        if bot == 0:
            # Prevent division by zero; break like a stalled Newton step
            break

        delta = top / bot
        t0 = t0 - delta

    # Truncate negative estimates at zero, with warning (mirrors R behavior)
    if t0 < 0:
        warnings.warn(f"estimate truncated at {min_t0}")
        t0 = max(t0, min_t0)

    return 1/float(t0)


def theta_cal(sol, df):
    dfr = len(df) - len(sol.h) - len(sol.w) - len(sol.c) + 2
    mu = mu_from_sol(sol, df)
    theta = theta_md(df["k"].values, mu.values, dfr)
    return theta
