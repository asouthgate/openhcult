from __future__ import annotations

import numpy as np

import cvxpy as cp

from hcultinf.logging import timed


@timed
def _trend_filter(y, lam):
    n = len(y)
    x = cp.Variable(n)
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    objective = cp.Minimize(0.5 * cp.sum_squares(x - y) + lam * cp.norm1(D2 @ x))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    return x.value


def _auto_lambda(values):
    sigma = np.std(np.diff(values))
    n = len(values)
    return sigma * np.sqrt(n * np.log(n))


@timed
def drying_rate(times, values, lambda_tv=None):
    times = np.asarray(times)
    values = np.asarray(values, dtype=float)
    if len(times) < 2:
        raise ValueError("Need at least 2 data points")

    if lambda_tv is None:
        lambda_tv = _auto_lambda(values)

    smoothed = _trend_filter(values, lambda_tv)

    if times.dtype.kind in ("i", "u"):
        dt_minutes = np.diff(times) / (60 * 1000)
    else:
        dt_minutes = np.diff(times).astype("timedelta64[m]").astype(float)
    dt_minutes[dt_minutes == 0] = 1e-9

    rate = np.diff(smoothed) / dt_minutes
    rate = np.concatenate([rate, rate[-1:]])

    valid = rate < 0

    return dict(times=times, rate=rate, valid=valid)
