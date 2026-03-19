from __future__ import annotations

import numpy as np

from hcultinf.detection import SegmentDetector


def filter_confirmed_watering_events(observations):
    """Return observations that mention WATER but not AUTO (manually confirmed events)."""
    return [
        (obs_id, ts, note)
        for obs_id, ts, note in observations
        if "WATER" in note and "AUTO" not in note
    ]


def match_events(confirmed_events, candidate_times, epsilon_ms=900_000):
    """Match candidate timestamps to confirmed events within an epsilon window.

    Returns dict with matched, missed, spurious counts.
    """
    epsilon = np.timedelta64(epsilon_ms, "ms")
    confirmed_times = [ts for _, ts, _ in confirmed_events]

    matched_confirmed = set()
    matched_candidates = set()
    for i, ct in enumerate(confirmed_times):
        for j, dt in enumerate(candidate_times):
            if j not in matched_candidates and abs(ct - dt) <= epsilon:
                matched_confirmed.add(i)
                matched_candidates.add(j)
                break

    return {
        "matched": len(matched_confirmed),
        "missed": len(confirmed_times) - len(matched_confirmed),
        "spurious": len(candidate_times) - len(matched_candidates),
    }


def compute_loss(
    confirmed_events, timeseries, segmenter_params, match_window_ms=7_200_000
):
    """
    Run the segmenter over timeseries with given params and compute classification loss
    against confirmed watering events.

    Returns dict with matched, missed, spurious, and loss (fraction wrong).
    """
    window = np.timedelta64(match_window_ms, "ms")

    detected_times = []
    for sensor, points in timeseries.items():
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        detector = SegmentDetector(times, values, **segmenter_params)
        for event in detector.get_watering_events():
            t0, _ = event.get_model_active_interval()
            detected_times.append(t0)

    confirmed_times = [ts for _, ts, _ in confirmed_events]

    matched_confirmed = set()
    matched_detected = set()
    for i, ct in enumerate(confirmed_times):
        for j, dt in enumerate(detected_times):
            if j not in matched_detected and abs(ct - dt) <= window:
                matched_confirmed.add(i)
                matched_detected.add(j)
                break

    matched = len(matched_confirmed)
    missed = len(confirmed_times) - matched
    spurious = len(detected_times) - len(matched_detected)
    total = matched + missed + spurious
    loss = 1.0 - (matched / total) if total > 0 else 0.0

    return {"matched": matched, "missed": missed, "spurious": spurious, "loss": loss}
