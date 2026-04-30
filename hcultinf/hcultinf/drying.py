from __future__ import annotations

import numpy as np

from hcultinf.detection import SegmentDetector


def drying_rate(times, values, **detector_kwargs):
    times = np.asarray(times)
    values = np.asarray(values, dtype=float)
    if len(times) < 2:
        raise ValueError("Need at least 2 data points")

    detector = SegmentDetector(times, values, **detector_kwargs)

    event_intervals = detector.get_disequilibrium_intervals()
    resampled_times = detector._resampled_times
    rate = detector._resampled_vel_smoothed

    valid = np.ones(len(resampled_times), dtype=bool)
    for interval in event_intervals:
        valid &= ~(
            (resampled_times >= interval.start) & (resampled_times <= interval.end)
        )

    return dict(times=resampled_times, rate=rate, valid=valid)
