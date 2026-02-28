"""Inference helpers for sensor time series."""

from __future__ import annotations

import numpy as np

from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import LSQUnivariateSpline, BSpline
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def compute_diff(values: np.ndarray, lag_arr: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for idx in range(values.size):
        lag_idx = idx - lag_arr[idx]
        if lag_idx < 0:
            continue
        if not np.isfinite(values[idx]) or not np.isfinite(values[lag_idx]):
            continue
        out[idx] = values[idx] - values[lag_idx]
    return out


def rolling_mad(values: np.ndarray, window_arr: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for idx in range(values.size):
        start = max(0, idx - window_arr[idx] + 1)
        window_slice = values[start : idx + 1]
        window_slice = window_slice[np.isfinite(window_slice)]
        if window_slice.size == 0:
            continue
        median = np.median(window_slice)
        mad = np.median(np.abs(window_slice - median))
        out[idx] = mad
    return out


def compute_zscore(values: np.ndarray, *, lag_arr: int, mad_window_arr: int, c: float) -> np.ndarray:
    diffs = compute_diff(values, lag_arr)
    mads = rolling_mad(diffs, mad_window_arr) + 1e-5  # Avoid division by zero
    sigma = c * mads
    z = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(diffs) & np.isfinite(sigma) & (sigma > 0)
    z[valid] = diffs[valid] / sigma[valid]
    return z


def run_lengths_at_starts(arr):
    arr = np.asarray(arr, dtype=bool)
    n = arr.size
    if n == 0:
        return np.array([], dtype=int)

    # run starts (True at the first index of each constant segment)
    run_start = np.r_[True, arr[1:] != arr[:-1]]
    starts = np.flatnonzero(run_start)

    # run lengths
    run_len = np.diff(np.r_[starts, n])

    out = np.zeros(n, dtype=int)
    out[starts] = run_len
    return out


def classify_events(times: np.ndarray, values: np.ndarray, lag_ms: np.int64, mad_window_ms: np.int64, c: float, pthresh: float) -> np.ndarray:
    """ Classify events in a time series based on z-scores of differences.

    Returns:
    - triggers: Boolean array indicating where events are triggered.
    - run_lengths: Array of the same shape as values, where each element is the length of the run of consecutive triggers starting at that index (0 if not a trigger).
    - starts: Boolean array indicating the start of runs of triggers that are at least as long as the lag.
    """
    # For each value point, we need to map lag_ms to lag and mad_window_ms to mad_window based on times.
    lag_arr = []
    mad_window_arr = []

    for idx in range(times.size):
        current_time = times[idx]
        lag_time = current_time - np.timedelta64(lag_ms, 'ms')
        mad_window_time = current_time - np.timedelta64(mad_window_ms, 'ms')

        # Find the indices of the lag and mad window
        lag_idx = np.searchsorted(times[:idx], lag_time, side='right') - 1
        mad_window_idx = np.searchsorted(times[:idx], mad_window_time, side='right') - 1

        lag_arr.append(idx - lag_idx if lag_idx >= 0 else 0)
        mad_window_arr.append(idx - mad_window_idx if mad_window_idx >= 0 else 0)

    lag_arr = np.array(lag_arr, dtype=int)
    mad_window_arr = np.array(mad_window_arr, dtype=int)
    lag_arr = np.ones(len(lag_arr), dtype=int) * 5
    mad_window_arr = np.ones(len(mad_window_arr), dtype=int) * 100
    zscores = compute_zscore(values, lag_arr=lag_arr, mad_window_arr=mad_window_arr, c=c)
    pvalues = zscore_pvalues(zscores)
    triggers = pvalues < pthresh
    run_lengths = run_lengths_at_starts(triggers)
    start_lengths = run_lengths * triggers
    starts = start_lengths >= lag_arr
    return triggers, run_lengths, starts


def zscore_pvalues(zscores: np.ndarray) -> np.ndarray:
    pvals = 2 * (1 - norm.cdf(np.abs(zscores)))
    return pvals

def fit_monotonic_spline(x, y, inner_knots, k=3):
    # Sort for the spline engine
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    x_min, x_max = xs.min(), xs.max()
    
    # 2. Use LSQUnivariateSpline just to get the "Full" knot vector (including pads)
    tmp_spline = LSQUnivariateSpline(xs, ys, inner_knots, k=k)
    t = tmp_spline.get_knots() # Full knot vector
    c0 = tmp_spline.get_coeffs() # Initial guess coefficients
    
    # 3. Objective: Minimize MSE of the BSpline
    def objective(coeffs):
        spl = BSpline(t, coeffs, k)
        return np.sum((spl(x) - y)**2)

    # 4. Constraint: Slope <= 0 (Decreasing) at 50 points
    x_check = np.linspace(x_min, x_max, 50)
    def monotonic_constraint(coeffs):
        spl = BSpline(t, coeffs, k)
        # We want -f'(x) >= 0 for decreasing
        return -spl(x_check, nu=1)

    res = minimize(objective, c0, constraints={'type': 'ineq', 'fun': monotonic_constraint})
    
    # Return the final optimized BSpline object
    return BSpline(t, res.x, k)


def fit_gp(x_raw, dy_noisy_raw, y_xmax_raw, inv_response_prior):

    x_raw_range = np.max(x_raw) - np.min(x_raw)
    x_raw_min = np.min(x_raw)
    # normalize x and y first for better GP performance
    x = np.array(x_raw, dtype=float)
    # linearly squash x into [0, 1]
    x -= x_raw_min
    x /= x_raw_range + 1e-5

    dy_noisy_median = np.median(dy_noisy_raw)
    dy_noisy_std = np.std(dy_noisy_raw)

    dy_noisy = np.array(dy_noisy_raw, dtype=float)
    dy_noisy -= dy_noisy_median
    dy_noisy /= dy_noisy_std + 1e-5
    y_xmax = float(y_xmax_raw) / dy_noisy_std + 1e-5

    X_train = x.reshape(-1, 1)

    # prior = np.median(dy_noisy)
    prior = 0
    y_train = dy_noisy - prior

    kernel =  ConstantKernel(
        0.2,
        constant_value_bounds=(1e-5, 1.0)
    ) * RBF(
        length_scale=0.2,
        length_scale_bounds=(0.01, 10000.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
    gp.fit(X_train, y_train)

    X_test = np.linspace(min(x), max(x), 400).reshape(-1, 1)

    # import matplotlib.pyplot as plt
    # plot std as shaded area
    y_pred, y_std = gp.predict(X_test, return_std=True)
    # plt.plot(X_test, y_pred + prior, color='red', label='GP Mean')
    # plt.fill_between(X_test.flatten(), y_pred + y_std + prior, y_pred - y_std + prior, color='red', alpha=0.3, label='GP Std Dev')
    # plt.scatter(x, dy_noisy, color='blue', alpha=0.5, label='Noisy Data')
    # plt.show()

    n_samples = 100
    dy_samples = gp.sample_y(X_test, n_samples=n_samples) + prior

    X = X_test.flatten()
    L = X[-1] - X[0]
    f_samples = cumulative_trapezoid(
        dy_samples,
        X_test,
        axis=0,
        initial=0
    )

    f_samples += y_xmax - f_samples[0, :]

    f_mean = np.mean(f_samples, axis=1)
    f_std  = np.std(f_samples, axis=1)

    # now scale predictions
    X_test_rescaled = X_test.flatten() * x_raw_range + x_raw_min
    y_pred_rescaled = y_pred * (dy_noisy_std + 1e-5) + dy_noisy_median
    y_std_rescaled = y_std * (dy_noisy_std + 1e-5)

    # plt.scatter(x_raw, dy_noisy_raw, color='blue', alpha=0.5, label='Rescaled GP Samples')
    # plt.fill_between(X_test_rescaled, y_pred_rescaled + y_std_rescaled, y_pred_rescaled - y_std_rescaled, color='red', alpha=0.3, label='Rescaled GP Std Dev')
    # plt.plot(X_test_rescaled, y_pred_rescaled, color='red', label='Rescaled GP Mean')
    # plt.show()


    f_mean = f_mean * (dy_noisy_std + 1e-5)
    f_std = f_std * (dy_noisy_std + 1e-5)

    dy_samples = dy_samples * (dy_noisy_std + 1e-5) + dy_noisy_median

    return X_test_rescaled, dy_samples, f_mean, f_std
