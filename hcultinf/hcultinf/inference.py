"""Inference helpers for sensor time series."""

from __future__ import annotations

from scipy.stats import norm

import numpy as np


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
