"""Inference helpers for sensor time series."""

from __future__ import annotations

from math import erf, sqrt
from typing import Tuple
from scipy.stats import norm

import numpy as np


def compute_diff(values: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError("lag must be >= 1")
    out = np.full(values.shape, np.nan, dtype=float)
    if values.size > lag:
        out[lag:] = values[lag:] - values[:-lag]
    return out


def rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be >= 1")
    out = np.full(values.shape, np.nan, dtype=float)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        window_slice = values[start : idx + 1]
        window_slice = window_slice[np.isfinite(window_slice)]
        if window_slice.size == 0:
            continue
        median = np.median(window_slice)
        mad = np.median(np.abs(window_slice - median))
        out[idx] = mad
    return out


def compute_zscore(values: np.ndarray, *, lag: int, mad_window: int, c: float) -> np.ndarray:
    diffs = compute_diff(values, lag)
    mads = rolling_mad(diffs, mad_window) + 1e-5  # Avoid division by zero
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

def classify_events(values: np.ndarray, lag: int, mad_window: int, c: float, pthresh: float) -> np.ndarray:
    zscores = compute_zscore(values, lag=lag, mad_window=mad_window, c=c)
    pvalues = zscore_pvalues(zscores)
    triggers = pvalues < pthresh

    run_lengths = run_lengths_at_starts(triggers)
    start_lengths = run_lengths * triggers
    starts = start_lengths >= lag
    
    return triggers, run_lengths, starts


def zscore_pvalues(zscores: np.ndarray) -> np.ndarray:
    pvals = 2 * (1 - norm.cdf(np.abs(zscores)))
    return pvals
