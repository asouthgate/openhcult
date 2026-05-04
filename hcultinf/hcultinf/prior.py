from __future__ import annotations

import numpy as np
from scipy.special import betaln, log_ndtr, ndtr, ndtri


def _flat_log_prior(x):
    return 0.0


def _bounded_log_prior(lo, hi):
    def log_prior(x):
        return 0.0 if lo <= x <= hi else -np.inf

    return log_prior


def _beta_log_prior(a, b):
    log_norm = -betaln(a, b)

    def log_prior(x):
        if x <= 0.0 or x >= 1.0:
            return -np.inf
        return float((a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) + log_norm)

    return log_prior


def _truncated_normal_logpdf(x, mu, sigma, hi=np.inf):
    if x > hi:
        return -np.inf
    if np.isinf(hi):
        return float(
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
        )
    z = (x - mu) / sigma
    b = (hi - mu) / sigma
    log_normal = -0.5 * z * z - 0.5 * np.log(2 * np.pi) - np.log(sigma)
    log_Z = log_ndtr(b)
    return float(log_normal - log_Z)


def _truncated_normal_ppf(u, mu, sigma, hi=np.inf):
    if np.isinf(hi):
        return float(mu + sigma * ndtri(u))
    b = (hi - mu) / sigma
    Phi_b = ndtr(b)
    return float(mu + sigma * ndtri(u * Phi_b))


class MCMCPriors:
    """Configurable log-prior functions for MCMC parameters.

    Each attribute is a callable that takes a parameter value and returns
    the log-prior density. Return -np.inf for forbidden regions.

    Parameters
    ----------
    scale_log_prior : callable, optional
        Log-prior on log(scale). Default: flat (Jeffreys prior on scale,
        p(scale) ∝ 1/scale).
    k_log_prior : callable, optional
        Log-prior on each sensor's k (curvature) parameter.
        Default: uniform on [1e-4, 1e4].
    f_int_log_prior : callable, optional
        Log-prior on each sensor's f_int (intercept fraction) parameter.
        Default: Beta(1, 3) on (0, 1).
    log_sigma_log_prior : callable, optional
        Log-prior on each sensor's log(sigma) (log-noise) parameter.
        Default: uniform on [log(1e-3), log(10)].
    xmin_log_prior : callable, optional
        Log-prior on each sensor's xmin parameter. Called as
        ``xmin_log_prior(x, xmin_hi)`` where xmin_hi is the hard
        upper bound (data minimum). Default: None (not used in this class;
        xmin prior is handled directly in mcmc_log_joint).
    """

    def __init__(
        self,
        scale_log_prior=None,
        k_log_prior=None,
        f_int_log_prior=None,
        log_sigma_log_prior=None,
        xmin_log_prior=None,
    ):
        self.scale_log_prior = (
            scale_log_prior if scale_log_prior is not None else _flat_log_prior
        )
        self.k_log_prior = (
            k_log_prior if k_log_prior is not None else _bounded_log_prior(1e-4, 1e4)
        )
        self.f_int_log_prior = (
            f_int_log_prior
            if f_int_log_prior is not None
            else _beta_log_prior(1.0, 10.0)
        )
        self.log_sigma_log_prior = (
            log_sigma_log_prior
            if log_sigma_log_prior is not None
            else _bounded_log_prior(np.log(1e-3), np.log(10.0))
        )
        self.xmin_log_prior = xmin_log_prior
