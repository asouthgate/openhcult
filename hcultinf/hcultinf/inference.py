"""Inference helpers for sensor time series."""

from __future__ import annotations

from math import erf, sqrt
from typing import Tuple

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


def compute_zscore(values: np.ndarray, *, lag: int, window: int, c: float) -> np.ndarray:
    diffs = compute_diff(values, lag)
    mads = rolling_mad(diffs, window) + 1e-5
    sigma = c * mads
    z = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(diffs) & np.isfinite(sigma) & (sigma > 0)
    z[valid] = diffs[valid] / sigma[valid]
    return z


def compute_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty(values.shape, dtype=float)
    if values.size == 0:
        return out
    out[0] = values[0]
    for idx in range(1, values.size):
        out[idx] = alpha * values[idx] + (1 - alpha) * out[idx - 1]
    return out


def zscore_pvalues(zscores: np.ndarray) -> np.ndarray:
    out = np.full(zscores.shape, 1.0, dtype=float)
    finite = np.isfinite(zscores)
    if not np.any(finite):
        return out
    abs_z = np.abs(zscores[finite])
    erf_vec = np.vectorize(erf)
    cdf = 0.5 * (1.0 + erf_vec(abs_z / sqrt(2.0)))
    out[finite] = 2.0 * (1.0 - cdf)
    return out


def detect_z_triggers(zscores: np.ndarray, p_thresh: float) -> np.ndarray:
    pvals = zscore_pvalues(zscores)
    return np.flatnonzero(pvals < p_thresh)


def detect_hysteresis(
    values: np.ndarray,
    baseline: np.ndarray,
    triggers: np.ndarray,
    *,
    window: int,
    threshold: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if window <= 0:
        raise ValueError("window must be >= 1")
    if k <= 0:
        raise ValueError("k must be >= 1")
    n = values.size
    confirmed = []
    flags = []
    for idx in triggers:
        if idx < 0 or idx >= n:
            continue
        start = idx
        end = min(n, idx + window)
        baseline_at_trigger = baseline[idx]
        window_slice = values[start:end] - baseline_at_trigger
        hits = np.sum(np.abs(window_slice) > threshold)
        ok = hits >= k
        confirmed.append(idx)
        flags.append(ok)
    return np.array(confirmed, dtype=int), np.array(flags, dtype=bool)


def merge_events(
    times_ms: np.ndarray, event_indices: np.ndarray, distance_ms: int
) -> np.ndarray:
    """Merge events by minimum separation distance."""
    if distance_ms <= 0:
        raise ValueError("distance_ms must be >= 1")
    if event_indices.size == 0:
        return event_indices
    ordered = event_indices[np.argsort(times_ms[event_indices])]
    kept = []
    last_time = None
    for idx in ordered:
        raw_time = times_ms[idx]
        if isinstance(raw_time, np.datetime64):
            event_time = int(raw_time.astype("datetime64[ms]").astype("int64"))
        elif hasattr(raw_time, "timestamp"):
            event_time = int(raw_time.timestamp() * 1000)
        else:
            event_time = int(raw_time)
        if last_time is None or event_time - last_time > distance_ms:
            kept.append(idx)
        last_time = event_time
    return np.array(kept, dtype=int)
