from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)


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


def fuse_swc(calibrators, voltages_per_sensor, sigma_bias=0.0):
    """Fuse SWC estimates from multiple sensors at each time point.

    Args:
        calibrators: list of fitted calibrators, one per sensor
        voltages_per_sensor: list of arrays, voltages_mv[i] = readings from sensor i
            All arrays must have the same length (aligned in time).
        sigma_bias: bias variance for Bayesian fusion (0 = precision-weighted mean).

    Returns dict with:
        mean: fused SWC at each time point
        ci_low: lower 95% CI
        ci_high: upper 95% CI
        n_sensors: number of sensors with readings at each point
    """
    _logger.info(
        "fuse_swc: %d calibrators, sigma_bias=%.4f", len(calibrators), sigma_bias
    )

    n_points = len(voltages_per_sensor[0])
    means_list = []
    stds_list = []
    mask_list = []
    for i, cal in enumerate(calibrators):
        v = np.asarray(voltages_per_sensor[i])
        valid = np.isfinite(v)
        _logger.info(
            "fuse_swc: sensor %d — %d/%d valid readings, voltage mean=%.2f min=%.2f max=%.2f",
            i,
            valid.sum(),
            len(v),
            np.nanmean(v),
            np.nanmin(v),
            np.nanmax(v),
        )
        swc = np.full(n_points, np.nan)
        swc_std = np.full(n_points, np.nan)
        swc[valid] = np.asarray(cal(v[valid]))
        swc_std[valid] = np.asarray(cal.std(v[valid]))
        means_list.append(swc)
        stds_list.append(swc_std)
        mask_list.append(valid)

    means_arr = np.array(means_list)
    stds_arr = np.array(stds_list)
    masks_arr = np.array(mask_list)

    prec = np.where(masks_arr, 1.0 / (stds_arr**2 + sigma_bias**2), 0.0)
    total_prec = prec.sum(axis=0)
    n_sensors = masks_arr.sum(axis=0)

    weighted = np.where(masks_arr, means_arr * prec, 0.0)
    fused_mean = np.where(
        total_prec > 0,
        weighted.sum(axis=0) / total_prec,
        np.nan,
    )
    fused_std = np.where(total_prec > 0, 1.0 / np.sqrt(total_prec), np.nan)

    valid_mask = np.isfinite(fused_mean)
    if valid_mask.any():
        _logger.info(
            "fuse_swc: result — mean of fused mean=%.2f min=%.2f max=%.2f, "
            "mean std=%.4f, avg sensors per point=%.1f",
            np.nanmean(fused_mean),
            np.nanmin(fused_mean),
            np.nanmax(fused_mean),
            np.nanmean(fused_std[valid_mask]),
            np.mean(n_sensors[valid_mask]),
        )

    return dict(
        mean=fused_mean,
        ci_low=fused_mean - 1.96 * fused_std,
        ci_high=fused_mean + 1.96 * fused_std,
        n_sensors=n_sensors,
    )
