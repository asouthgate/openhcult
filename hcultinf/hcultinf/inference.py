"""Inference helpers for sensor time series."""

from __future__ import annotations

import numpy as np
import pandas as pd 

from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import LSQUnivariateSpline, BSpline
from scipy.interpolate import interp1d
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



def fit_parametric_monotonic_spline(x_anchors, z_anchors, x_der, dz_dx, knots=10, k=3, w_der=1.0):
    
    x_min, x_max = min(x_anchors.min(), x_der.min()), max(x_anchors.max(), x_der.max())
    
    # Map s to the data points
    s_der = (x_der - x_min) / (x_max - x_min)
    s_anc = (x_anchors - x_min) / (x_max - x_min)
    
    # if type(knots) is int:
    n_int = knots
    inner = np.linspace(0, 1, n_int + 2)[1:-1]
    t = np.concatenate(([0.0]*(k+1), inner, [1.0]*(k+1)))
    # else:
    #     # Assume n_int is already the inner knots
    #     inner = np.asarray(knots)
    #     t = np.concatenate(([0.0]*(k+1), inner, [1.0]*(k+1)))  
    n_c = len(t) - k - 1
    
    # Initial guess
    c0_x = np.linspace(x_min, x_max, n_c)
    c0_z = np.linspace(z_anchors.max(), z_anchors.min(), n_c)
    c0 = np.concatenate([c0_x, c0_z])

    def objective(coeffs):
        cx, cz = coeffs[:n_c], coeffs[n_c:]
        sx, sz = BSpline(t, cx, k), BSpline(t, cz, k)
        
        # 1. Anchor Errors (Position)
        err_anc = np.sum((sx(s_anc) - x_anchors)**2) + np.sum((sz(s_anc) - z_anchors)**2)
        
        # 2. X-Alignment (Ensures s_der actually corresponds to x_der)
        # This prevents the 'shift' by forcing sx(s) approx x
        err_x_align = np.sum((sx(s_der) - x_der)**2)
        
        # 3. Shape Error (Chain Rule: dz/ds = dz/dx * dx/ds)
        # dz_dx MUST be d(SWC)/d(Value)
        z_prime = sz(s_der, nu=1)
        x_prime = sx(s_der, nu=1)
        err_der = np.sum((z_prime - (dz_dx * x_prime))**2)
        
        # We give high priority to staying aligned with the X coordinates
        return (err_anc + 10.0 * err_x_align) + (w_der * err_der)

    def monotonic_con(coeffs):
        cz = coeffs[n_c:]
        # dz/ds <= 0 for decreasing SWC
        return -BSpline(t, cz, k)(np.linspace(0, 1, 50), nu=1)


    res = minimize(objective, c0, constraints={'type': 'ineq', 'fun': monotonic_con})
    return BSpline(t, res.x[:n_c], k), BSpline(t, res.x[n_c:], k)



def bootstrap_parametric_spline(x_anchors, z_anchors, x_der, dz_dx, knots=10, k=2, w_der=0.5, n_boots=50):
    """
    Performs bootstrapping on the derivative data to produce a distribution of splines.
    """
    boot_results = []
    n_der = len(x_der)
    
    print(f"Starting bootstrap ({n_boots} iterations)...")
    
    for i in range(n_boots):
        print(f"Bootstrap iteration {i+1}/{n_boots}")
        # 1. Resample derivative indices with replacement
        indices = np.random.choice(n_der, size=n_der, replace=True)
        
        x_resampled = x_der[indices]
        dz_dx_resampled = dz_dx[indices]
        
        # 2. Fit the model using the resampled derivatives
        try:
            sx, sz = fit_parametric_monotonic_spline(
                x_anchors, z_anchors, 
                x_resampled, dz_dx_resampled, 
                knots=knots, k=k, w_der=w_der
            )
            boot_results.append((sx, sz))
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue
            
    return boot_results


def bootstrap_monotonic_spline(x, y, inner_knots, k=3, n_boots=50):
    boot_splines = []
    n = len(x)
    
    print(f"Starting bootstrap ({n_boots} iterations)...")
    
    for i in range(n_boots):
        print(f"Bootstrap iteration {i+1}/{n_boots}")
        # Resample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_resampled = x[indices]
        y_resampled = y[indices]
        
        try:
            spline = fit_monotonic_spline(x_resampled, y_resampled, inner_knots, k=k)
            boot_splines.append(spline)
        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            continue
            
    return boot_splines


def compute_lookup_table_from_bootstrap(boot_splines, x_min, x_max, n_points=100):
    x_grid = np.linspace(x_min, x_max, n_points)
    z_grid = np.array([spline(x_grid) for spline in boot_splines])
    
    # Compute mean and confidence intervals
    z_mean = np.mean(z_grid, axis=0)
    z_lower = np.percentile(z_grid, 2.5, axis=0)
    z_upper = np.percentile(z_grid, 97.5, axis=0)
    z_std = np.std(z_grid, axis=0)

    lookup_df = pd.DataFrame({
        "x": x_grid,
        "swc": z_mean,
        "swc_std": z_std,
        "swc_upper_95%": z_upper,
        "swc_lower_95%": z_lower,
    })

    
    return lookup_df

def compute_lookup_table_parametric_forward(boot_mod_splines, s_min=0, s_max=1, n_points=1000):
    # 1. Create a master s_grid to evaluate the 'Average' curve
    s_grid = np.linspace(s_min, s_max, n_points)
    
    # We need to collect X and Z for every bootstrap sample
    x_samples = []
    z_samples = []
    
    for sx, sz in boot_mod_splines:
        x_samples.append(sx(s_grid))
        z_samples.append(sz(s_grid))
        
    x_samples = np.array(x_samples)
    z_samples = np.array(z_samples)
    
    # 2. Compute means
    # Note: These are 'Mean X' and 'Mean Z' for a given 's'
    x_mean = np.mean(x_samples, axis=0)
    z_mean = np.mean(z_samples, axis=0)
    
    # 3. Compute Standard Deviation of the SWC (z)
    z_std = np.std(z_samples, axis=0)
    z_lower = np.percentile(z_samples, 2.5, axis=0)
    z_upper = np.percentile(z_samples, 97.5, axis=0)

    lookup_df = pd.DataFrame({
        "s": s_grid,
        "x": x_mean,         # This is your Sensor Reading
        "swc": z_mean,       # This is your moisture
        "swc_std": z_std,
        "swc_upper_95%": z_upper,
        "swc_lower_95%": z_lower,
    })
    
    return lookup_df