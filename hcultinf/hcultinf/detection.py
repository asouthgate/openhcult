
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
            trigger_thresh_acc = -0.015,
            release_thresh_acc = -0.0075,
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
        self._trigger_arr, self._release_arr = lerp_thresholds(
            self._resampled_emwa,
            self._trigger_thresh_acc,
            self._release_thresh_acc,
            self._trigger_thresh_acc / 4.0,
            self._release_thresh_acc / 4.0
        )




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
        
        return find_regions_with_hysteresis_adapative_thresh(
            self._resampled_times,
            self._resampled_acc_smoothed,
            self._trigger_arr,
            self._release_arr
        )
    
    def get_thresholds(self):
        return self._trigger_arr, self._release_arr


    def debug_plot(self):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.scatter(self._time_arr, self._values_arr, color='tab:blue', label='Sensor values', alpha=0.5, marker='x')
        ax1.plot(self._resampled_times, self._resampled_emwa, color='tab:blue', label='Smoothed values (EWMA)', alpha=1.0, linewidth=1)
        # ax1.plot(self._time_arr, self._values_emwa, color='tab:tab:blue', alpha=1.0, linewidth=1)
        ax1.set_ylabel('Sensor reading', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(
            self._resampled_times,
            self._resampled_vel_smoothed/max(self._resampled_vel_smoothed),
            color='#ad444f', label='Smoothed velocity (EWMA)', linewidth=1
        )
        ax2.plot(
            self._resampled_times-self._emwa_tau_minutes,
            self._resampled_acc_smoothed.values/max(self._resampled_acc_smoothed),
            color='purple',
            label='Smoothed acceleration (EWMA)',
            linewidth=1
        )

        deq_regions = self.get_disequilibrium_intervals()
        for start, end in deq_regions:
            ax2.axvspan(start, end, color='green', alpha=0.15)

        eq_regions = self.get_equilibrium_intervals()
        for start, end in eq_regions:
            ax2.axvspan(start, end, color='grey', alpha=0.15)

        acc_regions = self.get_acceleration_intervals()
        for start, end in acc_regions:
            ax2.axvspan(start, end, color='orange', alpha=0.15)


        trigger, release = self.get_thresholds()
        ax2.axhline(0, color='tab:blue', linestyle='--', alpha=0.3) # Zero baseline
        ax2.plot(self._resampled_times, trigger, color='red', linestyle='--')
        ax2.plot(self._resampled_times, release, color='red', linestyle='--')

        ax2.set_ylabel('Normalized values (derivatives)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        # 2. Combine them and call legend on just one of the axes
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title('Moisture Levels vs. Smoothed Velocity')
        plt.savefig("segmentation.png")
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

def lerp_thresholds(val, trigger_high, release_high, trigger_low, release_low):
    # assert trigger_high >= trigger_low
    # assert release_high >= release_low

    # Linear interp thresholds
    maxx = max(val)
    minxx = min(val)
    rangex = maxx - minxx

    range_trigger = trigger_high - trigger_low
    range_release = release_high - release_low

    m_trigger = range_trigger / rangex
    m_release = range_release / rangex

    c_trigger = trigger_low
    c_release = release_low

    # trigger_max = m_trigger * (maxx-minxx) + c_trigger
    # trigger_min = m_trigger * (minxx-minxx) + c_trigger

    # release_max = m_release * (maxx-minxx) + c_release
    # release_min = m_release * (minxx-minxx) + c_release

    trigger_arr = np.zeros(len(val))
    release_arr = np.zeros(len(val))

    for i, v in enumerate(val):
        trigger = m_trigger * (v-minxx) + c_trigger
        release = m_release * (v-minxx) + c_release
        trigger_arr[i] = trigger
        release_arr[i] = release
            
    return trigger_arr, release_arr



def find_regions_with_hysteresis_adapative_thresh(times, val, trigger_arr, release_arr):
    regions = []
    active = False
    start_time = None

    for ti, tup in enumerate(zip(times, val)):
        t, v = tup
        if active and v > release_arr[ti]:
            active = False
            regions.append((start_time, t))
        if not active and v < trigger_arr[ti]:
            active = True
            start_time = t
            
    # Handle event still active at end of data
    if active:
        regions.append((start_time, times[-1]))
        
    return regions
