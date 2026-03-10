#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


from hcultinf.detection import DisequilibriumIntervalDetector, compute_ewma, greedy_merge_event_times, classify_events_shock, compute_zscore
from hcultutils.fetch_data import fetch_data


def _plot_raw_subsensor_readings(ax, raw_ax, times, values, name, locator, emwa_tau_minutes):
    emwa = compute_ewma(times, values, emwa_tau_minutes)
    ax.plot(times, values, label=name, linewidth=1.2)
    ax.plot(times, emwa, label=name, linewidth=1.2, linestyle='--')
    ax.scatter(times, values)
    raw_ax.plot(times, emwa, label=name, linewidth=1.2, linestyle='--')
    raw_ax.plot(times, values, label=name, linewidth=1.2)
    raw_ax.scatter(times, values, label=name, linewidth=1.2)
    raw_ax.legend()
    raw_ax.set_title(name)
    raw_ax.set_xlabel("Timestamp")
    raw_ax.set_ylabel("Value")
    raw_ax.set_ylim(1, 3300)
    raw_ax.xaxis.set_major_locator(locator)
    raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    raw_ax.tick_params(axis="x", rotation=30)


def merge_time_intervals(intervals, merge_tol=np.timedelta64(1, 'm')):
    """
    Greedily merge datetime64 intervals.

    Parameters
    ----------
    intervals : list of (np.datetime64, np.datetime64)
        List of (start, end) tuples.
    merge_tol : np.timedelta64
        Maximum allowed gap between intervals to merge them.

    Returns
    -------
    list of (np.datetime64, np.datetime64)
    """
    if not intervals:
        return []
    # Sort by start time
    # for i in intervals: print(i)
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        gap = start - cur_end  # this is np.timedelta64

        if gap <= merge_tol:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged

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

    locator = mdates.AutoDateLocator()
    fig = plt.figure(figsize=(14, 4 + 4 * math.ceil(len(sensor_names) / 2)))
    cols = 2
    rows = math.ceil(len(sensor_names) / cols)
    grid = fig.add_gridspec(rows + 1, cols)
    raw_axes = []
    for i in range(len(sensor_names)):
        r = i // cols
        c = i % cols
        raw_axes.append(fig.add_subplot(grid[r, c]))
    ax = fig.add_subplot(grid[rows, :])
    times_map = {}

    equilibrium_x_values = []
    equilibrium_delta_values = []
    equilibrium_sensors = []
    equilibrium_t_values = []
    equilbrium_intervals_index_values = []

    diff_lag_ms = args.diff_lag_seconds * 1000
    mad_window_ms = args.mad_window_seconds * 1000
    all_fast_trigger_times = []
    all_slow_trigger_times = []

    emwa_tau_minutes = 30

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        times_map[name] = times
        trigger_times, diff_triggers, hyst_triggers, greedy_triggers, emwa_triggers, signed_triggers = classify_events_shock(
            times, values, diff_lag_ms, mad_window_ms, args.mad_scale, args.z_pvalue)
        ded = DisequilibriumIntervalDetector(times, values, emwa_tau_minutes)
        slow_watering_intervals = ded.get_intervals()
        ded.debug_plot()
        all_fast_trigger_times += list(trigger_times)
        all_slow_trigger_times += list(slow_watering_intervals)

        for si, tt in enumerate(trigger_times):
            ei = si + 1
            val_subset = values[si:ei]
            times_subset = times[si:ei]
            if len(val_subset) > 50:
                deltas = np.diff(val_subset)
                equilibrium_x_values += list(val_subset[:-1])
                equilibrium_delta_values += list(deltas)
                equilibrium_sensors += [idx] * len(deltas)
                equilibrium_t_values += list(times_subset)

        # for tt in diff_triggers:
        #     raw_axes[idx].axvline(tt, color="red", alpha=1.0, linewidth=1)
        # for tt in hyst_triggers:
        #     raw_axes[idx].axvline(tt, color="orange", alpha=1.0, linewidth=1.2)
        # for tt in greedy_triggers:
        #     raw_axes[idx].axvline(tt, color="yellow", alpha=1.0, linewidth=1.4)
        # for tt in emwa_triggers:
        #     raw_axes[idx].axvline(tt, color="green", alpha=1.0, linewidth=1.6)
        # for tt in signed_triggers:
        #     raw_axes[idx].axvline(tt, color="blue", alpha=1.0, linewidth=1.8)
        # for start, end in slow_watering_intervals:
        #     raw_axes[idx].axvline(tt, color="green", alpha=0.2, linewidth=1.4)
        for start, end in slow_watering_intervals:
            
            # Optional: Keep the lines on the edges for extra definition
            # raw_axes[idx].axvline(start, color="black", alpha=1.0, linewidth=1.5)
            # raw_axes[idx].axvline(end, color="blue", alpha=0.8, linewidth=1.5)
            raw_axes[idx].axvspan(start, end, color="purple", alpha=0.1)
        # for tt in trigger_times:
        #     raw_axes[idx].axvline(tt, color="black", alpha=1.0, linewidth=2)
        _plot_raw_subsensor_readings(ax, raw_axes[idx], times, values, name, locator, emwa_tau_minutes)
        # now plot observations on the raw axes as well


    # all_slow_triggers_merged = greedy_merge_event_times(all_slow_trigger_times, np.timedelta64(5,'m'))
    for start, end in all_slow_trigger_times:
        ax.axvspan(start, end, color="purple", alpha=0.1)
        # print(ti, tt)
        # raw_axes[idx].axvline(tt, color="red", alpha=0.5, linewidth=1)
        # if ti == 0:
        #     ax.axvline(tt, color="blue", alpha=0.5, linewidth=1)
        # else:
        #     ax.axvline(tt, color="blue", alpha=0.5, linewidth=1)
    for ti, tt in enumerate(all_fast_trigger_times):
        # print(ti, tt)
        # raw_axes[idx].axvline(tt, color="red", alpha=0.5, linewidth=1)
        if ti == 0:
            ax.axvline(tt, color="red", alpha=0.5, linewidth=1)
        else:
            ax.axvline(tt, color="red", alpha=0.5, linewidth=1)

            
    ax.set_title("Sensor Readings")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.set_ylim(1, 3300)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        z_out = out_path.with_name(f"{out_path.stem}_z{out_path.suffix}")
        fig.savefig(z_out, dpi=150)
        print(f"Wrote {z_out}")
    else:
        plt.show()

    return 0
