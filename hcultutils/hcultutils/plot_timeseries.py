#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Dict, List, Tuple
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


from hcultinf.inference import compute_ewma, compute_zscore, classify_events
from hcultutils.fetch_data import fetch_data

def _plot_raw_subsensor_readings(ax, raw_ax, times, values, name, args, locator):
    ax.plot(times, values, label=name, linewidth=1.2)
    raw_ax.plot(times, values, label=name, linewidth=1.2)
    raw_ax.scatter(times, values, label=name, linewidth=1.2, s=0.5)
    raw_ax.legend()
    raw_ax.set_title(name)
    raw_ax.set_xlabel("Timestamp")
    raw_ax.set_ylabel("Value")
    raw_ax.set_ylim(1, 3300)
    raw_ax.xaxis.set_major_locator(locator)
    raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    raw_ax.tick_params(axis="x", rotation=30)

def _plot_z_subsensor_readings(ax, raw_ax, times, zscores, name, args, locator):
    ax.plot(times, zscores, label=name, linewidth=1.2)
    raw_ax.plot(times, zscores, label=name, linewidth=1.2)
    raw_ax.legend()
    raw_ax.set_title(f"{name} z(t)")
    raw_ax.set_xlabel("Timestamp")
    raw_ax.set_ylabel("z(t)")
    raw_ax.set_yscale("symlog", linthresh=1.0)
    raw_ax.xaxis.set_major_locator(locator)
    raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    raw_ax.tick_params(axis="x", rotation=30)

def _plot_residual_subsensor_readings(ax, r_raw_ax, times, residuals, name, args, locator):
    ax.plot(times, residuals, label=name, linewidth=1.2)
    r_raw_ax.plot(times, residuals, label=name, linewidth=1.2)
    r_raw_ax.legend()
    r_raw_ax.set_title(f"{name} residuals")
    r_raw_ax.set_xlabel("Timestamp")
    r_raw_ax.set_ylabel("x(t) - B(t)")
    r_raw_ax.xaxis.set_major_locator(locator)
    r_raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    r_raw_ax.tick_params(axis="x", rotation=30)

def _plot_observations(observations, sensor_names, ax, raw_axes):
    for _, obs_time, _ in observations:
        ax.axvline(obs_time, color="tab:orange", alpha=0.4, linewidth=1)
    for obs_id, obs_time, note in observations:
        if not note:
            continue
        short_note = note[:10]
        label = f"{obs_id}:{short_note}"
        ax.annotate(
            label,
            xy=(obs_time, 0.99),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="black",
        )
    for idx, name in enumerate(sensor_names):
        raw_ax = raw_axes[idx]
        key_tag = f"{name} "
        for obs_id, obs_time, note in observations:
            if not note or key_tag not in note:
                continue
            raw_ax.axvline(obs_time, color="tab:orange", alpha=0.4, linewidth=1)
            short_note = note[:10]
            label = f"{obs_id}:{short_note}"
            raw_ax.annotate(
                label,
                xy=(obs_time, 0.99),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="black",
            )

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
    for i in intervals: print(i)
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
    if args.out:
        matplotlib.use("Agg")

    fetched = fetch_data(args)
    if not fetched:
        return 1
    series, observations = fetched
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
    zscores_map: Dict[str, np.ndarray] = {}
    trigger_bools_map: Dict[str, np.ndarray] = {}
    values_map: Dict[str, np.ndarray] = {}
    times_map: Dict[str, np.ndarray] = {}
    trigger_times: Dict[str, np.ndarray] = {}

    equilibrium_x_values = []
    equilibrium_delta_values = []
    equilibrium_sensors = []
    equilibrium_t_values = []
    no_event_time_intervals = []
    equilbrium_intervals_index_values = []

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        zscores = compute_zscore(
            values, lag=args.diff_lag, mad_window=args.mad_window, c=args.mad_scale
        )
        zscores2 = compute_zscore(
            values, lag=args.diff_lag * 3, mad_window=args.mad_window, c=args.mad_scale
        )
        zscores_map[name] = zscores
        times_map[name] = times
        triggers, run_lengths, starts = classify_events(values, args.diff_lag, args.mad_window, args.mad_scale, args.z_pvalue)
        starts_t = times[starts]

        # compute the event time intervals
        for i in range(len(starts)):
            si = starts[i]
            ei = si + run_lengths[i]
            # event_time_intervals.append((times[si], times[ei-1]))

        starts_inds = np.where(starts)[0]
        for i in range(len(starts_inds) - 1):
            si = starts_inds[i]
            ei = si + run_lengths[si]
            next_si = starts_inds[i + 1]
            print(si, ei)
            print(f"[{ei} {next_si}]")
            val_subset = values[ei:next_si]
            no_event_time_intervals.append((times[ei], times[next_si-1]))
            if len(val_subset) > 50:
                deltas = np.diff(val_subset)
                equilibrium_x_values += list(val_subset[:-1])
                equilibrium_delta_values += list(deltas)
                equilibrium_sensors += [idx] * len(deltas)
                equilibrium_t_values += list(times[ei:next_si-1])

        for tt in starts_t:
            raw_axes[idx].axvline(tt, color="orange", alpha=0.5, linewidth=1)

        _plot_raw_subsensor_readings(ax, raw_axes[idx], times, values, name, args, locator)
        # now plot observations on the raw axes as well

    merged_no_event_intervals = merge_time_intervals(no_event_time_intervals)
    for idx in range(len(sensor_names)):
        for tt, tte in merged_no_event_intervals:
            raw_axes[idx].axvline(tt, color="green", alpha=0.5, linewidth=1)
            
    equilbrium_intervals_index_values = []
    for t in equilibrium_t_values:
        for i, (tt, tte) in enumerate(merged_no_event_intervals):
            if tt <= t <= tte:
                equilbrium_intervals_index_values.append(i)
                break

    assert len(equilbrium_intervals_index_values) == len(equilibrium_t_values)
    assert len(equilbrium_intervals_index_values) == len(equilibrium_x_values)

    ax.set_title("Sensor Readings")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.set_ylim(1, 3300)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    zfig = plt.figure(figsize=(14, 4 + 4 * math.ceil(len(sensor_names) / 2)))
    zgrid = zfig.add_gridspec(rows + 1, cols)
    zraw_axes = []
    for i in range(len(sensor_names)):
        r = i // cols
        c = i % cols
        zraw_axes.append(zfig.add_subplot(zgrid[r, c]))
    zax = zfig.add_subplot(zgrid[rows, :])

    for idx, name in enumerate(sensor_names):
        times = times_map[name]
        zscores = zscores_map[name]
        _plot_z_subsensor_readings(zax, zraw_axes[idx], times, zscores, name, args, locator)    

    zax.set_title("z(t) = d(t) / (c * MAD)")
    zax.set_xlabel("Timestamp")
    zax.set_ylabel("z(t)")
    zax.set_yscale("symlog", linthresh=1.0)
    zax.xaxis.set_major_locator(locator)
    zax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    zax.legend()
    zfig.autofmt_xdate()
    zfig.tight_layout()

    # rax.set_title("Residuals x(t) - B(t)")
    # rax.set_xlabel("Timestamp")
    # rax.set_ylabel("x(t) - B(t)")
    # rax.xaxis.set_major_locator(locator)
    # rax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    # rax.legend()
    # rfig.autofmt_xdate()
    # rfig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        z_out = out_path.with_name(f"{out_path.stem}_z{out_path.suffix}")
        zfig.savefig(z_out, dpi=150)
        print(f"Wrote {z_out}")
        # r_out = out_path.with_name(f"{out_path.stem}_resid{out_path.suffix}")
        # rfig.savefig(r_out, dpi=150)
        # print(f"Wrote {r_out}")
    else:
        plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    ax = axes.flatten()

    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    for sensor in set(equilibrium_sensors):
        sensor_mask = np.array(equilibrium_sensors) == sensor
        ax[0].scatter(
            np.array(equilibrium_x_values)[sensor_mask],
            np.array(equilibrium_delta_values)[sensor_mask],
            label=sensor_names[sensor],
            marker=markers[sensor % len(markers)],
            alpha=0.5
        )
        ax[1].hist(np.array(
            equilibrium_delta_values)[sensor_mask],
            bins=np.arange(-10.5, 10.5, 1.0), alpha=0.7, histtype='step',
            label=f"Sensor {sensor}"
        )
        ax[1].set_xlabel("$\Delta X_t$")
        ax[1].legend()

    # plt.scatter(equilibrium_x_values, equilibrium_delta_values, cmap="tab10", c=equilibrium_sensors, marker=sensor_markers, alpha=0.5)
    ax[0].set_xlabel("$X_t$")
    ax[0].set_ylabel("$\Delta x_t$")

    # equilibrium_x_values = []
    # equilibrium_delta_values = []
    # equilibrium_sensors = []
    # equilibrium_t_values = []
    # no_event_time_intervals = []
    # equilbrium_intervals_index_values = []

    equilbrium_intervals_index_values = np.array(equilbrium_intervals_index_values)
    equilibrium_x_values = np.array(equilibrium_x_values)
    equilibrium_sensors = np.array(equilibrium_sensors)

    unique_intervals = np.unique(equilbrium_intervals_index_values)

    sensor_ids = np.unique(equilibrium_sensors)
    n_sensors = len(sensor_ids)

    width = 0.15
    offsets = np.linspace(-0.3, 0.3, n_sensors)

    eiv_means = []
    eiv_standards = []
    eiv2eidx = {}
    for eidx, eiv in enumerate(unique_intervals):
        interval_mask = equilbrium_intervals_index_values == eiv
        x_vals_eiv = equilibrium_x_values[interval_mask]
        x_vals_eiv_std = np.std(x_vals_eiv)
        x_vals_eiv_mean = np.mean(x_vals_eiv)
        eiv_means.append(x_vals_eiv_mean)
        eiv_standards.append(x_vals_eiv_std)
        eiv2eidx[eiv] = eidx
    for i, sensor in enumerate(sensor_ids):
        sensor_mask = equilibrium_sensors == sensor
        x_vals = equilibrium_x_values[sensor_mask]
        x_means = np.array([eiv_means[eiv2eidx[idx]] for idx in equilbrium_intervals_index_values[sensor_mask]])
        x_stds = np.array([eiv_standards[eiv2eidx[idx]] for idx in equilbrium_intervals_index_values[sensor_mask]])
        x_vals_centered = (x_vals - x_means) / x_stds
            

        # x_vals_centered = [(xv - x_vals_eiv_mean) / x_vals_eiv_std ]
        if len(x_vals_centered) == 0:
            continue

        pos = i

        ax[2].boxplot(
            x_vals_centered,
            positions=[pos],
            widths=width,
            patch_artist=True
        )

        ax[2].set_ylabel("X, centered per-equilibrium-interval")
        ax[2].set_xlabel("Sensor")

    # Finally just plot the sensor readings in the whole domain, no equilibrium
    for si in sensor_ids:
        sensor_mask = equilibrium_sensors == si
        x_vals = equilibrium_x_values[sensor_mask]
        t_vals = np.array(equilibrium_t_values)[sensor_mask]
        ax[3].scatter(t_vals, x_vals, label=sensor_names[si], alpha=0.5)
        ax[3].plot(t_vals, x_vals, label=sensor_names[si], alpha=0.5)
        ax[3].set_xlabel("t")
        ax[3].set_ylabel("X")

    fig.suptitle("Equilibrium noise distribtions")
    fig.savefig("equilibrium_noise.png", dpi=300)

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
