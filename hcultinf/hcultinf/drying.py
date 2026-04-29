from __future__ import annotations

import numpy as np


def linear_drying_rate(calibrator, times_ms, voltages_mv):
    voltages_mv = np.asarray(voltages_mv)
    times_ms = np.asarray(times_ms)
    if len(voltages_mv) < 2:
        return None

    swc = np.asarray(calibrator(voltages_mv))
    swc_std = np.asarray(calibrator.std(voltages_mv))
    t_days = (times_ms - times_ms[0]) / (24.0 * 3600.0 * 1000.0)

    w = 1.0 / np.maximum(swc_std**2, 1e-12)
    W = np.sum(w)
    Wx = np.sum(w * t_days)
    Wy = np.sum(w * swc)
    Wxx = np.sum(w * t_days**2)
    Wxy = np.sum(w * swc * t_days)

    det = W * Wxx - Wx**2
    if abs(det) < 1e-15:
        return None

    intercept = (Wxx * Wy - Wx * Wxy) / det
    slope = (W * Wxy - Wx * Wy) / det

    s2 = np.sum(w * (swc - intercept - slope * t_days) ** 2) / max(W - 2, 1)
    var_intercept = s2 * Wxx / det
    var_slope = s2 * W / det

    return dict(
        rate_ml_per_day=float(slope),
        rate_ci_low=float(slope - 1.96 * np.sqrt(max(var_slope, 0))),
        rate_ci_high=float(slope + 1.96 * np.sqrt(max(var_slope, 0))),
        intercept=float(intercept),
        intercept_ci_low=float(intercept - 1.96 * np.sqrt(max(var_intercept, 0))),
        intercept_ci_high=float(intercept + 1.96 * np.sqrt(max(var_intercept, 0))),
        n_points=len(voltages_mv),
    )
