#!/usr/bin/env python3
"""Fetch sensor data, classify drops, group sequences, and plot."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from hcultinf.inference import (
    compute_ewma,
    compute_zscore,
    detect_hysteresis,
    detect_z_triggers,
    merge_events,
)


@dataclass(frozen=True)
class Event:
    index: int
    time_ms: int
    before: float
    after: float
    before_start: int
    after_end: int


def _parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_series_from_ctrl(
    ctrl_url: str,
    *,
    sensor: str | None,
    device: str | None,
    plant_name: str | None,
    start_utc: str,
    end_utc: str,
    limit: int,
) -> Dict[str, List[Tuple[int, int]]]:
    series: Dict[str, List[Tuple[int, int]]] = {}
    params: Dict[str, str] = {
        "format": "json",
        "limit": str(limit),
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    if sensor:
        params["sensor"] = sensor
    if device:
        params["device"] = device
    if plant_name:
        params["plant"] = plant_name

    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/timeseries?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)

    if payload.get("count") == limit:
        print(
            "Warning: reached the row limit; results may be truncated. "
            "Try --limit or a narrower time window."
        )

    for row in payload.get("data", []):
        time_ms = row.get("adjusted_time_ms")
        if time_ms is None:
            continue
        device_name = row.get("device_name") or row.get("device_address") or "unknown"
        sensor_name = row.get("sensor") or "sensor"
        key = f"{device_name}:{sensor_name}"
        series.setdefault(key, []).append((int(time_ms), int(row.get("measurement", 0))))
    for key in series:
        series[key].sort(key=lambda item: item[0])
    return series


def _median_window(values: np.ndarray, start: int, end: int) -> float | None:
    window = values[start:end]
    if window.size == 0:
        return None
    return float(np.median(window))


def _classify_events(times_ms: np.ndarray, values: np.ndarray, args: argparse.Namespace) -> List[Event]:
    baseline = compute_ewma(values, args.ewma_alpha)
    zscores = compute_zscore(values, lag=args.diff_lag, window=args.mad_window, c=args.mad_scale)
    triggers = detect_z_triggers(zscores, args.z_pvalue)
    confirmed, flags = detect_hysteresis(
        values,
        baseline,
        triggers,
        window=args.hyst_window,
        threshold=args.resid_threshold,
        k=args.hyst_k,
    )
    confirmed = confirmed[flags]
    merged = merge_events(times_ms, confirmed, args.merge_distance_sec * 1000)

    events: List[Event] = []
    n = values.size
    for idx in merged:
        if idx <= 0 or idx >= n - 1:
            continue
        before_start = max(0, idx - args.level_window)
        before_end = idx
        after_start = idx + 1
        after_end = min(n, idx + 1 + args.level_window)
        before = _median_window(values, before_start, before_end)
        after = _median_window(values, after_start, after_end)
        if before is None or after is None:
            continue
        if after >= before:
            continue
        events.append(
            Event(
                index=int(idx),
                time_ms=int(times_ms[idx]),
                before=before,
                after=after,
                before_start=before_start,
                after_end=after_end,
            )
        )
    return events


def _group_events(events: List[Event], delta_ms: int, eps: float) -> List[List[Event]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda event: event.time_ms)
    groups = [[ordered[0]]]
    for event in ordered[1:]:
        prev = groups[-1][-1]
        if (event.time_ms - prev.time_ms) <= delta_ms and abs(prev.after - event.before) <= eps:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _plot_series(
    ax,
    name: str,
    times: np.ndarray,
    values: np.ndarray,
    groups: List[List[Event]],
) -> None:
    ax.plot(times, values, label=name, linewidth=1.2)
    colors = plt.cm.tab20.colors
    labels_used = set()
    for group_idx, group in enumerate(groups):
        color = colors[group_idx % len(colors)]
        label = f"group {group_idx + 1}"
        for event in group:
            before_start_time = times[event.before_start]
            before_end_time = times[event.index]
            after_start_time = times[event.index]
            after_end_time = times[event.after_end - 1]
            line_label = label if label not in labels_used else "_nolegend_"
            labels_used.add(label)
            ax.hlines(
                event.before,
                before_start_time,
                before_end_time,
                colors=color,
                linewidth=2.0,
                label=line_label,
            )
            ax.hlines(
                event.after,
                after_start_time,
                after_end_time,
                colors=color,
                linewidth=2.0,
                label="_nolegend_",
            )
    ax.set_title(name)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")


def run(args: argparse.Namespace) -> int:
    if not args.start_utc or not args.end_utc:
        raise SystemExit("fetch_event_sequences requires --start-utc and --end-utc")

    start = _parse_utc(args.start_utc)
    end = _parse_utc(args.end_utc)
    if end <= start:
        raise SystemExit("--end-utc must be after --start-utc")

    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")
    if not args.ctrl_url:
        raise SystemExit("fetch_event_sequences requires --ctrl-url")

    series = _fetch_series_from_ctrl(
        args.ctrl_url,
        sensor=args.sensor,
        device=args.device,
        plant_name=args.plant_name,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=args.limit,
    )

    if not series:
        print("No sensor readings found.")
        return 0

    if args.out:
        matplotlib.use("Agg")

    sensor_names = sorted(series.keys())
    cols = 2
    rows = int(np.ceil(len(sensor_names) / cols))
    fig = plt.figure(figsize=(14, 4 * max(rows, 1)))
    grid = fig.add_gridspec(rows, cols)
    axes = []
    for idx in range(len(sensor_names)):
        r = idx // cols
        c = idx % cols
        axes.append(fig.add_subplot(grid[r, c]))

    locator = mdates.AutoDateLocator()
    summaries = []

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times_ms = np.array([t for t, _ in points], dtype=np.int64)
        values = np.array([v for _, v in points], dtype=float)
        times = times_ms.astype("datetime64[ms]")
        events = _classify_events(times_ms, values, args)
        groups = _group_events(events, args.group_delta_sec * 1000, args.group_eps)
        summaries.append((name, len(events), len(groups)))
        _plot_series(axes[idx], name, times, values, groups)
        axes[idx].xaxis.set_major_locator(locator)
        axes[idx].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[idx].tick_params(axis="x", rotation=30)
        axes[idx].legend(loc="upper right", fontsize=8)

    fig.suptitle("Grouped Event Sequences")
    fig.tight_layout()

    for name, event_count, group_count in summaries:
        print(f"{name}: {event_count} events in {group_count} groups")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
    else:
        plt.show()

    return 0
