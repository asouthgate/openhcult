
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from scipy.stats import norm

def compute_ewma(data, span=5):
    """
    Computes EWMA using Pandas.
    Span is the most common parameter (N-day EWMA).
    """
    series = pd.Series(data)
    # span corresponds to alpha = 2 / (span + 1)
    return series.ewm(span=span, adjust=False).mean().values


def compute_time_weighted_ewma(times, values, tau_minutes=30.0):
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


def compute_diff_arr(values: np.ndarray, lag_arr: int) -> np.ndarray:
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
    mads = rolling_mad(diffs, mad_window_arr) + np.min(np.abs(diffs))
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

def get_complementary_intervals(intervals, start_bound, end_bound):
    """
    Returns the gaps between disjoint intervals within a specific range.
    """
    # 1. Ensure intervals are sorted by their start index
    sorted_intervals = sorted(intervals)
    
    complementary = []
    current_pos = start_bound

    for start, end in sorted_intervals:
        # If there is space between the current position and the next interval
        if start > current_pos:
            complementary.append([current_pos, start - 1])
        
        # Move the cursor to just after the current interval
        current_pos = max(current_pos, end + 1)

    # 2. Check if there is a remaining gap after the last interval
    if current_pos <= end_bound:
        complementary.append([current_pos, end_bound])

    return complementary


class DisequilibriumIntervalDetector:
    """Takes sensor data and identifies regions in (dis)equilibrium."""
    def __init__(
            self,
            time_arr,
            values_arr,
            emwa_tau_minutes=30,
            trigger_thresh=-0.25,
            release_thresh = -0.15,
            trigger_thresh_acc = -0.005,
            release_thresh_acc = -0.001,
        ):
        self._time_arr = time_arr
        self._values_arr = values_arr
        self._emwa_tau_minutes = emwa_tau_minutes
        self._trigger_thresh = trigger_thresh
        self._release_thresh = release_thresh

        self._trigger_thresh_acc = trigger_thresh_acc
        self._release_thresh_acc = release_thresh_acc


        self._time_diff_minutes = (time_arr[1:] - time_arr[:-1]) / np.timedelta64(1, 'm')
        self._values_diff = (values_arr[1:] - values_arr[:-1])

        self._values_emwa = compute_time_weighted_ewma(time_arr, values_arr, emwa_tau_minutes)
        tmp_emwa_series = pd.Series(self._values_emwa, index=pd.to_datetime(time_arr))
        self._resampled_emwa = tmp_emwa_series.resample('1min').mean().interpolate(method='linear')
        self._resampled_vel = self._resampled_emwa.diff().fillna(0)
        self._resampled_vel_smoothed = self._resampled_vel.ewm(span=emwa_tau_minutes).mean()
        self._resampled_acc_smoothed = self._resampled_vel_smoothed.diff().fillna(0).ewm(span=emwa_tau_minutes).mean()
        self._resampled_times = self._resampled_emwa.index.to_numpy()

    def get_disequilibrium_intervals(self):
        diseq_regions = find_regions_with_hysteresis(
            self._resampled_times, self._resampled_vel_smoothed, self._trigger_thresh, self._release_thresh)
        return diseq_regions
    
    def get_equilibrium_intervals(self):
        diseq_regions = find_regions_with_hysteresis(
            self._resampled_times, self._resampled_vel_smoothed, self._trigger_thresh, self._release_thresh)
        eq_regions = get_complementary_intervals(diseq_regions, min(self._time_arr), max(self._time_arr))
        return eq_regions
    
    def get_acceleration_intervals(self):
        return find_regions_with_hysteresis(
            self._resampled_times, self._resampled_acc_smoothed, self._trigger_thresh_acc, self._release_thresh_acc
        )

    def debug_plot(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(self._time_arr, self._values_arr, color='black', label='Sensor values', alpha=0.5, marker='x')
        ax1.plot(self._resampled_times, self._resampled_emwa, color='black', label='Smoothed values (EWMA)', alpha=1.0, linewidth=1)
        # ax1.plot(self._time_arr, self._values_emwa, color='tab:blue', alpha=1.0, linewidth=1)
        ax1.set_ylabel('Sensor reading', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        ax2.plot(
            self._resampled_times,
            self._resampled_vel_smoothed/max(self._resampled_vel_smoothed),
            color='orange', label='Smoothed velocity (EWMA)', linewidth=2
        )
        ax2.plot(
            self._resampled_times-self._emwa_tau_minutes,
            self._resampled_acc_smoothed.values/max(self._resampled_acc_smoothed),
            color='purple',
            label='Smoothed acceleration (EWMA)',
            linewidth=2
        )

        # ax2.axhline(self._trigger_thresh/max(self._resampled_vel_smoothed), color='red', linestyle='--', alpha=0.6, label='Trigger threshold')
        # ax2.axhline(self._release_thresh/max(self._resampled_vel_smoothed), color='red', linestyle='--', alpha=0.6, label='Release threshold')

        deq_regions = self.get_disequilibrium_intervals()
        for start, end in deq_regions:
            ax2.axvspan(start, end, color='yellow', alpha=0.15)

        eq_regions = self.get_equilibrium_intervals()
        for start, end in eq_regions:
            ax2.axvspan(start, end, color='grey', alpha=0.15)

        acc_regions = self.get_acceleration_intervals()
        for start, end in acc_regions:
            ax2.axvspan(start, end, color='purple', alpha=0.15)


        ax2.axhline(0, color='black', linestyle='--', alpha=0.3) # Zero baseline
        ax2.set_ylabel('Normalized values (derivatives)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        # ax1.legend()
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        # 2. Combine them and call legend on just one of the axes
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title('Moisture Levels vs. Smoothed Velocity')
        fig.tight_layout()
        plt.show()



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
        if active and v > release:
            active = False
            regions.append((start_time, t))
        if not active and v < trigger:
            active = True
            start_time = t
            
    # Handle event still active at end of data
    if active:
        regions.append((start_time, times[-1]))
        
    return regions

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

    diffs = compute_diff_arr(values, lag_arr)
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
    emwa = compute_time_weighted_ewma(times, values)
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