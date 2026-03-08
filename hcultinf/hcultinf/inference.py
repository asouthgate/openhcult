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


# def compute_ewma(data, span=5):
#     """
#     Computes EWMA using Pandas.
#     Span is the most common parameter (N-day EWMA).
#     """
#     series = pd.Series(data)
#     # span corresponds to alpha = 2 / (span + 1)
#     return series.ewm(span=span, adjust=False).mean().values

def compute_ewma(times, values, tau_minutes=30.0):
    """
    tau_minutes: The 'memory' of the filter. 
    Larger tau = smoother, but stays 'stuck' longer after gaps.
    """
    # Convert times to float minutes
    t_min = times.astype('datetime64[m]').astype(float)
    
    n = len(values)
    smoothed = np.zeros(n)
    smoothed[0] = values[0] # Initialize
    
    for i in range(1, n):
        delta_t = t_min[i] - t_min[i-1]
        
        # Calculate dynamic alpha based on time gap
        alpha = 1 - np.exp(-delta_t / tau_minutes)
        
        smoothed[i] = (1 - alpha) * smoothed[i-1] + alpha * values[i]
        
    return smoothed

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


def compute_zscore(values: np.ndarray, diffs, *, mad_window_arr: int, c: float) -> np.ndarray:
    # we need to add the min to stop zero MAD, where any event thereafter is + np.abs(min(diffs))
    mads = rolling_mad(diffs, mad_window_arr) + np.min(np.abs(diffs))
    # mads = np.ones(len(diffs)) * 1
    sigma = c * mads
    z = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(diffs) & np.isfinite(sigma) & (sigma > 0)
    z[valid] = diffs[valid] / sigma[valid]
    return z

def compute_zscore_madval(values: np.ndarray, diffs, *, mad: int, c: float) -> np.ndarray:
    # we need to add the min to stop zero MAD, where any event thereafter is + np.abs(min(diffs))
    # mads = rolling_mad(diffs, mad_window_arr) + np.min(np.abs(diffs))
    # mads = np.ones(len(diffs)) * 1
    sigma = c * mad
    z = np.full(values.shape, np.nan, dtype=float)
    z = diffs / sigma
    return z


def compute_zscore_single(v1, v2, mad, c) -> np.ndarray:
    sigma = c * mad
    return v2 - v1 / sigma


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

def greedy_merge_event_times(event_times, dist):
    """
    Groups 1D events by a maximum distance threshold and returns 
    the first event (representative) of each cluster.
    """
    if not event_times:
        return []
    
    # Ensure events are sorted for 1D greedy processing
    sorted_events = sorted(event_times)
    filtered = [sorted_events[0]]
    
    for i in range(1, len(sorted_events)):
        # If the current event is outside the window of the last kept event
        if sorted_events[i] - filtered[-1] >= dist:
            filtered.append(sorted_events[i])
            
    return sorted(filtered)


def get_runs(times, boolean, max_time_dist):
    # Easy algorithm. 1 loop. 
    # Iterate over. If boolean[j], set last_trigger to j, continue in same event
    # Otherwise, if time dist(times[j], times[last_trigger]) < max_time_dist, keep going
    # When dist(times[j], times[last_trigger]) > max_time_dist, mark end, reset start
    res = []
    start = None
    last_j = None
    last_time = None
    for j, bj in enumerate(boolean):
        # if we are in a run but the time delta is too big, end regardless
        if not start is None:
            dt = times[j] - last_time
            if not boolean[j] :
                # end condition reached
                res.append((start, last_j))
                start = None
                last_j = None
                last_time = None
        if bj:
            if start is None:  # Not in an event period
                start = j
            last_j = j
            last_time = times[j]
            continue
        # It's a zero, but could be in the time period anyway
    return res

def get_runs_boolean(boolean):
    res = []
    start = None
    n = len(boolean)
    
    for j in range(n):
        bj = boolean[j]
        # Check neighbors safely (handling edges)
        bprev = boolean[j-1] if j > 0 else False
        bnext = boolean[j+1] if j < n-1 else False
        
        # 1. Detection of Start
        if bj and not bprev:
            start = j
            
        # 2. Detection of End
        if bj and not bnext:
            if start is not None:
                # Use (start, j) or (start, j+1) depending on your indexing preference
                res.append((start, j))
                start = None # CRITICAL: Reset memory for next event
                
    return res

def find_downward_regions(times, vel_emwa, delta=-0.05):
    """
    Finds start/end times where velocity < delta (negative threshold).
    Note: times and vel_emwa must be the same length.
    """
    # 1. Create mask (velocity is more negative than delta)
    is_down = vel_emwa < delta
    
    # 2. Find transitions
    # prepend/append False to handle cases starting or ending 'in-run'
    padded = np.r_[False, is_down, False]
    idx = np.flatnonzero(padded[1:] != padded[:-1])
    
    # 3. Pair starts and ends
    # Result is a list of (start_time, end_time)
    regions = []
    for i in range(0, len(idx), 2):
        start_idx = idx[i]
        end_idx = idx[i+1] - 1
        regions.append((times[start_idx], times[end_idx]))
        
    return regions

def find_regions_with_hysteresis(times, vel, trigger=-0.015, release=-0.005):
    regions = []
    active = False
    start_time = None
    
    for t, v in zip(times, vel):
        if not active and v < trigger:
            active = True
            start_time = t
        elif active and v > release:
            active = False
            regions.append((start_time, t))
            
    # Handle event still active at end of data
    if active:
        regions.append((start_time, times[-1]))
        
    return regions

def get_decreasing_regions(times: np.ndarray, values: np.ndarray, emwa_tau_minutes, thresh=0.05) -> np.ndarray:

    
    time_diff_minutes = (times[1:] - times[:-1]) / np.timedelta64(1, 'm')
    diff = (values[1:] - values[:-1])
    # vel = diff / time_diff_minutes

    values_emwa = compute_ewma(times, values, emwa_tau_minutes)
    dvalues_emwa_dt = (values_emwa[1:] - values_emwa[:-1]) / time_diff_minutes

    vel_emwa = compute_ewma(times, dvalues_emwa_dt, emwa_tau_minutes)

    series_emwa = pd.Series(values_emwa, index=pd.to_datetime(times))

    # 2. Resample to a regular 1-minute grid
    # 'mean' handles multiple points in a minute, 'interpolate' fills gaps
    resampled_emwa = series_emwa.resample('1min').mean().interpolate(method='linear')

    # 3. Calculate Velocity on the regular grid (dt is now constant = 1.0)
    # This is much cleaner than (times[1:] - times[:-1])
    vel_resampled = resampled_emwa.diff().fillna(0)

    # 4. Final Smooth of the velocity (optional but recommended for your "Double Smooth")
    vel_final = vel_resampled.ewm(span=emwa_tau_minutes).mean()


    # 5. Extract values for your plotting/detection logic
    times_reg = vel_final.index.to_numpy()
    v_vals = vel_final.values

    accel_series = vel_final.diff().fillna(0)
    accel_smooth = accel_series.ewm(span=emwa_tau_minutes).mean()
    print(accel_smooth)
    # accel_emwa = compute_ewma(times[2:], accel, emwa_tau_minutes)

    trigger_tresh = -0.45
    release_thresh = -0.25
    down_regions = find_regions_with_hysteresis(times_reg, v_vals, trigger_tresh, release_thresh)
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Axis: Raw Moisture Values
    ax1.scatter(times, values, color='tab:blue', label='Moisture (%)', alpha=0.5)
    ax1.plot(times_reg, resampled_emwa, color='tab:blue', label='Moisture (%)', alpha=1.0, linewidth=1)
    ax1.plot(times, values_emwa, color='tab:blue', label='Moisture (%)', alpha=1.0, linewidth=1)
    ax1.set_ylabel('Moisture Content', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Right Axis: Velocity (EWMA)
    ax2 = ax1.twinx()
    # Note: times[1:] aligns with the diff-based velocity
    # ax2.plot(times[1:], vel_emwa/max(vel_emwa), color='tab:red', label='Normalized Velocity (EWMA)', linewidth=2)
    ax2.plot(times_reg, v_vals/max(v_vals), color='tab:pink', label='Regular Velocity (EWMA)', linewidth=2)
    # ax2.plot(times_reg, v_vals/max(vel_emwa), color='tab:pink', label='Regular Velocity (EWMA)', linewidth=2)
    ax2.plot(times_reg, accel_smooth.values/max(accel_smooth), color='tab:purple', label='Regular Velocity (EWMA)', linewidth=2)

    ax2.axhline(trigger_tresh/max(vel_emwa), color='red', linestyle='--', alpha=0.6, label='Threshold')
    ax2.axhline(release_thresh/max(vel_emwa), color='red', linestyle='--', alpha=0.6, label='Threshold')

    # 2. Shaded regions for each detected event
    for start, end in down_regions:
        ax2.axvspan(start, end, color='gray', alpha=0.15)
    # ax2.plot(times[2:], accel/max(accel), color='tab:purple', label='Normalized Acceleration (EWMA)', linewidth=2)
    # ax2.plot(times[1:], dvalues_emwa_dt/max(dvalues_emwa_dt), color='tab:pink', label='EWMA derivative', alpha=0.5, linewidth=3)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.3) # Zero baseline
    ax2.set_ylabel('Velocity (Units/Min)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.legend()
    plt.title('Moisture Levels vs. Smoothed Velocity')
    fig.tight_layout()
    plt.show()
    
    return down_regions


def classify_events_shock(
    times: np.ndarray, values: np.ndarray, lag_ms: np.int64, mad_window_ms: np.int64, c: float, pthresh: float
) -> np.ndarray:
    """ Classify events in a time series based on z-scores of differences.

    Returns:
    - triggers: Boolean array indicating where events are triggered.
    - run_lengths: Array of the same shape as values, where each element is the length of the run of consecutive triggers starting at that index (0 if not a trigger).
    - starts: Boolean array indicating the start of runs of triggers that are at least as long as the lag.
    """
    # For each value point, we need to map lag_ms to lag and mad_window_ms to mad_window based on times.
    # Each lag_arr[i] gives the integer index lag corresponding to the lag_ms for i
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
    # print(lag_arr)
    MAD = 6
    lag_arr = np.ones(len(values)).astype(int) * 1
    lag_arr[0] = 0
    mad_window_arr = np.array(mad_window_arr, dtype=int)

    diffs = compute_diff(values, lag_arr)
    # zscores = compute_zscore(values, diffs, mad_window_arr=mad_window_arr, c=c)
    zscores = compute_zscore_madval(values, diffs, mad = MAD, c=c)

    pvalues = zscore_pvalues(zscores)
    confirmed_triggers = pvalues < pthresh # triggers array one at a trigger and zero at a non-trigger
    diff_triggers = times[np.where(confirmed_triggers.copy())[0]]


    # 2. FILTER BY HYSTERESIS (SIGNIFICANCE FROM )
    K = 10
    for trind in np.where(confirmed_triggers)[0]:
        x0ind = trind - 2
        x0 = values[x0ind]
        for tj in range(trind, trind + K):
            z_score = compute_zscore_single(x0, values[tj], MAD, c)
            pval = zscore_pvalues(z_score)
            if pval > pthresh:
                confirmed_triggers[trind] = False
    hyst_triggers = times[np.where(confirmed_triggers.copy())[0]]

    # 3. FILTER BY GREEDY 
    for tri in np.where(confirmed_triggers)[0]:
        endj = tri + 1
        for j in range(tri, len(confirmed_triggers) - 1):
            if not confirmed_triggers[j]:
                endj = j
                break
        confirmed_triggers[tri + 1:endj] = False
    greedy_triggers = times[np.where(confirmed_triggers.copy())[0]]

    # 4. FILTER BY EMWA
    emwa = compute_ewma(times, values)
    K_EMWA = 5
    THRESH_EMWA = 0.0
    # now require that emwa is decreasing for K samples after trigger
    for tri in np.where(confirmed_triggers)[0]:
        endj = min(tri + K_EMWA, len(confirmed_triggers) - 1)
        if emwa[tri] > emwa[tri + 1]:
            comp = lambda x, y: y - x < -THRESH_EMWA  # need emwa derivative to be negative if it's a 'negative event'
        else:
            comp = lambda x, y: y - x > THRESH_EMWA  # need derivative to be positive if it's a positive event
        for j in range(tri, endj):
            if comp(emwa[j], emwa[tri]):
            # if emwa[j] - emwa[tri] < THRESH_EMWA:
                confirmed_triggers[tri] = False
    emwa_triggers = times[np.where(confirmed_triggers.copy())[0]]

    # 5 FILTER BY SIGN
    signs_watering = diffs < 0  # only those decreasing are watering, increasing must be drying
    # confirmed_triggers = confirmed_triggers & signs_watering
    signed_triggers = times[np.where(confirmed_triggers.copy())[0]]

    # run_lengths = run_lengths_at_starts(confirmed_triggers)

    # start_lengths = run_lengths * confirmed_triggers
    # starts = start_lengths >= lag_arr

    final_triggers = np.where(confirmed_triggers)[0]
    trigger_times = times[final_triggers]
    return trigger_times, diff_triggers, hyst_triggers, greedy_triggers, emwa_triggers, signed_triggers


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

    lookup_df = pd.DataFrame({
        "s": s_grid,
        "sensor_val": x_mean,         # This is your Sensor Reading
        "swc": z_mean,       # This is your moisture
        "swc_std": z_std
    })
    
    return lookup_df


def compute_mrt(times, vals):
    """
    Computes the Mean Residence Time (MRT) of the sensor transition.
    Handles datetime64/timedelta64 and is direction-agnostic.
    
    Returns:
        float: MRT in seconds (the 'delta T' of the transition).
    """
    # 1. Convert time to numeric seconds
    if np.issubdtype(times.dtype, np.datetime64) or np.issubdtype(times.dtype, np.timedelta64):
        t_numeric = (times - times[0]) / np.timedelta64(1, 's')
    else:
        t_numeric = times - times[0]

    # 2. Extract boundaries with median filtering for noise robustness
    v_start = np.median(vals[:5])
    v_end = np.median(vals[-5:])
    
    # If the sensor didn't move, MRT is undefined/zero
    if np.isclose(v_start, v_end, atol=1e-7):
        return [], 0.0 
    
    # 3. Normalize Values to [0, 1]
    # This 'flips' the curve so that for both wetting and drying, 
    # the 'target' is 1 and the 'start' is 0.
    v_norm = np.clip((vals - v_start) / (v_end - v_start), 0, 1)
    
    # 4. Calculate MRT via Integration
    # MRT = Integral from 0 to T of (1 - v_norm) dt
    try:
        # Use trapezoid for NumPy 2.0+, fallback to trapz for older versions
        auc = np.trapezoid(v_norm, t_numeric)
    except AttributeError:
        auc = np.trapz(v_norm, t_numeric)
        
    # The MRT is the 'Area Above the Curve'
    total_duration = t_numeric[-1]
    mrt = total_duration - auc
    
    return vals, mrt


def fit_parametric_spline_with_residuals(x, y, xmid, dydx, n_inner_knots, k_spline, w_der, n_boots):
    """Compute a parametric spline using both x,y data as well as xmid (x_i+1/2) and derivative at midpoint data.
    
    Params:
        x, y, xmid, ydx: arrays
        n_inner_knots: int spline knots
        k_spline: int order of spline
        w_der: weight of derivative samples
        n_boots: number of bootstrap samples
    """

    spline_mod_x, spline_mod_y = fit_parametric_monotonic_spline(
        x, y, xmid, dydx, n_inner_knots, k_spline, w_der
    )

    boot_mod_splines = bootstrap_parametric_spline(
        x, y, xmid, dydx, n_inner_knots, k_spline, w_der, n_boots=n_boots)

    # After fitting the parametric splines, we now have to invert from x -> s so that we can calculate y(x)
    s_fine = np.linspace(0, 1, 1000)
    x_fine = spline_mod_x(s_fine)
    x_to_s_map = interp1d(x_fine, s_fine, bounds_error=False, fill_value="extrapolate")
    s_data = x_to_s_map(x)
    y_pred = spline_mod_y(s_data)
    abs_residuals = np.abs(y - y_pred)

    inner_knots = ( np.linspace(0, 1, n_inner_knots)**2 * (x.max() - x.min()) + x.min() ) [1:]
    inner_knots[-1] = (inner_knots[-1] + inner_knots[-2]) / 2

    residual_spline_mod = fit_monotonic_spline(
        x,
        abs_residuals,
        inner_knots=inner_knots,
        k=k_spline,
    )
    return spline_mod_x, spline_mod_y, boot_mod_splines, residual_spline_mod