#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


from hcultinf.detection import DisequilibriumIntervalDetector, greedy_merge_event_times
from hcultutils.fetch_data import fetch_data


def main(args) -> int:
    print(args)
    if args.out:
        matplotlib.use("Agg")
    fetched = fetch_data(
        args.ctrl_url,
        args.start_utc,
        args.end_utc,
        sensor=args.sensor,
        device=args.device,
        plant_name=args.plant_name,
        limit=args.limit,
    )
    if not fetched:
        return 1
    series, _ = fetched
    sensor_names = sorted(series.keys())

    times_map = {}

    emwa_tau_minutes = 30

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        times_map[name] = times
        ded = DisequilibriumIntervalDetector(times, values, emwa_tau_minutes)
        ded.debug_plot()

