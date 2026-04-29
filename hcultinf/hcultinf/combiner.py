from __future__ import annotations

import numpy as np


def combine_bayesian(calibrators, x_grid, sigma_bias=0.0):
    if len(calibrators) == 0:
        raise ValueError("Need at least one calibrator")
    if len(calibrators) == 1:
        c = calibrators[0]
        mean, lo, hi = c.predict(x_grid)
        return np.asarray(mean), np.asarray(lo), np.asarray(hi)

    means = np.array([c(x_grid) for c in calibrators])
    stds = np.array([c.std(x_grid) for c in calibrators])

    precision = 1.0 / (stds**2 + sigma_bias**2)
    total_precision = precision.sum(axis=0)
    combined_mean = (means * precision).sum(axis=0) / total_precision
    combined_std = 1.0 / np.sqrt(total_precision)
    return (
        combined_mean,
        combined_mean - 1.96 * combined_std,
        combined_mean + 1.96 * combined_std,
    )


def combine_empirical_bayes(calibrators, x_grid):
    if len(calibrators) < 2:
        raise ValueError("Empirical Bayes requires at least 2 calibrators")

    means = np.array([c(x_grid) for c in calibrators])
    stds = np.array([c.std(x_grid) for c in calibrators])

    between_var = np.var(means, axis=0)
    avg_within_var = np.mean(stds**2, axis=0)
    sigma_bias_sq = np.maximum(0.0, between_var - avg_within_var)
    sigma_bias = float(np.median(np.sqrt(sigma_bias_sq)))

    return (*combine_bayesian(calibrators, x_grid, sigma_bias=sigma_bias), sigma_bias)


def combine_posteriors(calibrators, x_grid, sigma_bias=0.0, method="bayesian"):
    for c in calibrators:
        if not hasattr(c, "posterior_samples_at"):
            raise TypeError(
                f"{c.__class__.__name__} does not support posterior sampling; "
                "combine_posteriors requires MCMC calibrators"
            )
    if method == "empirical_bayes":
        return combine_empirical_bayes(calibrators, x_grid)
    if method == "bayesian":
        return combine_bayesian(calibrators, x_grid, sigma_bias=sigma_bias)
    raise ValueError(f"Unknown method: {method!r}")
