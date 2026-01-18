#!/usr/bin/env python3
"""Plot sensor time series from the configured SQLite database."""

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
from typing import Dict, List, Tuple

import matplotlib
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "openhcult" / "openhcult.conf"
    return Path.home() / ".config" / "openhcult" / "openhcult.conf"


def _load_db_path(config_path: Path) -> Path:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if "database" not in parser or "path" not in parser["database"]:
        raise ValueError(f"Missing database.path in {config_path}")
    configured = parser["database"]["path"].strip()
    if not configured:
        raise ValueError(f"Empty database.path in {config_path}")
    db_path = Path(configured).expanduser()
    if not db_path.is_absolute():
        db_path = config_path.parent / db_path
    return db_path


def _fetch_series(db_path: Path) -> Dict[str, List[Tuple[np.datetime64, int]]]:
    series: Dict[str, List[Tuple[np.datetime64, int]]] = {}
    query = (
        "SELECT sr.adjusted_time_ms, d.name, d.tag, d.address, sr.sensor, sr.measurement "
        "FROM sensor_readings sr "
        "JOIN devices d ON sr.device_id = d.id "
        "ORDER BY sr.adjusted_time_ms ASC, sr.id ASC"
    )
    with sqlite3.connect(db_path) as conn:
        for time_ms, device_name, device_tag, device_addr, sensor_name, value in conn.execute(query):
            if time_ms is None:
                continue
            timestamp = np.datetime64(int(time_ms), "ms")
            base = device_name or device_addr or "unknown"
            if device_tag:
                base = f"{base} ({device_tag})"
            series.setdefault(f"{base}:{sensor_name}", []).append(
                (timestamp, int(value))
            )
    return series


def _fetch_observations(db_path: Path) -> List[Tuple[np.datetime64, str]]:
    observations: List[Tuple[np.datetime64, str]] = []
    query = "SELECT observed_at, note FROM observations ORDER BY observed_at ASC, id ASC"
    with sqlite3.connect(db_path) as conn:
        try:
            for observed_at, note in conn.execute(query):
                if observed_at is None:
                    continue
                timestamp = np.datetime64(int(observed_at), "ms")
                observations.append((timestamp, str(note)))
        except sqlite3.OperationalError:
            return []
    return observations


def _fetch_series_from_ctrl(
    ctrl_url: str,
    *,
    sensor: str | None,
    device: str | None,
    start_ms: int | None,
    end_ms: int | None,
    start_utc: str | None,
    end_utc: str | None,
    limit: int,
) -> Dict[str, List[Tuple[np.datetime64, int]]]:
    series: Dict[str, List[Tuple[np.datetime64, int]]] = {}
    params: Dict[str, str] = {"format": "json", "limit": str(limit)}
    if sensor:
        params["sensor"] = sensor
    if device:
        params["device"] = device
    if start_ms is not None:
        params["start_ms"] = str(start_ms)
    if end_ms is not None:
        params["end_ms"] = str(end_ms)
    if start_utc:
        params["start_utc"] = start_utc
    if end_utc:
        params["end_utc"] = end_utc

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
        timestamp = np.datetime64(int(time_ms), "ms")
        device_name = row.get("device_name") or row.get("device_address") or "unknown"
        sensor_name = row.get("sensor") or "sensor"
        key = f"{device_name}:{sensor_name}"
        series.setdefault(key, []).append((timestamp, int(row.get("measurement", 0))))
    return series


def _fetch_observations_from_ctrl(
    ctrl_url: str,
    *,
    start_ms: int | None,
    end_ms: int | None,
    start_utc: str | None,
    end_utc: str | None,
    limit: int,
) -> List[Tuple[np.datetime64, str]]:
    params: Dict[str, str] = {"limit": str(limit)}
    if start_ms is not None:
        params["start_ms"] = str(start_ms)
    if end_ms is not None:
        params["end_ms"] = str(end_ms)
    if start_utc:
        params["start_utc"] = start_utc
    if end_utc:
        params["end_utc"] = end_utc

    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/observations?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)

    observations: List[Tuple[np.datetime64, str]] = []
    for row in payload.get("data", []):
        observed_at = row.get("observed_at")
        if observed_at is None:
            continue
        timestamp = np.datetime64(int(observed_at), "ms")
        observations.append((timestamp, str(row.get("note", ""))))
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot sensor time series from the configured SQLite database."
    )
    parser.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Path to openhcult.conf (default: XDG config)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override database path (otherwise read from config)",
    )
    parser.add_argument(
        "--ctrl-url",
        default=None,
        help="Query data from hcultctrl instead of SQLite (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--sensor",
        default=None,
        help="Filter to a single sensor name (e.g. sensor1)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Filter to a device name or BLE address",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=None,
        help="Start time in epoch milliseconds",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=None,
        help="End time in epoch milliseconds",
    )
    parser.add_argument(
        "--start-utc",
        default=None,
        help="Start time in UTC (ISO 8601, e.g. 2026-01-16T12:00:00Z)",
    )
    parser.add_argument(
        "--end-utc",
        default=None,
        help="End time in UTC (ISO 8601, e.g. 2026-01-16T13:00:00Z)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit number of rows when querying hcultctrl",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write PNG to this path instead of showing a window",
    )
    args = parser.parse_args()

    if args.out:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    observations: List[Tuple[np.datetime64, str]] = []
    if args.ctrl_url:
        series = _fetch_series_from_ctrl(
            args.ctrl_url,
            sensor=args.sensor,
            device=args.device,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            limit=args.limit,
        )
        observations = _fetch_observations_from_ctrl(
            args.ctrl_url,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            limit=args.limit,
        )
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")

        db_path = Path(args.db) if args.db else _load_db_path(config_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Missing database: {db_path}")
        series = _fetch_series(db_path)
        observations = _fetch_observations(db_path)
    if not series:
        print("No sensor readings found.")
        return 0

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

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = [t for t, _ in points]
        values = np.array([v for _, v in points], dtype=float)
        ax.plot(times, values, label=name, linewidth=1.2)

        raw_ax = raw_axes[idx]
        raw_ax.plot(times, values, label=name, linewidth=1.2)
        raw_ax.legend()
        raw_ax.set_title(name)
        raw_ax.set_xlabel("Timestamp")
        raw_ax.set_ylabel("Value")
        raw_ax.xaxis.set_major_locator(locator)
        raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        raw_ax.tick_params(axis="x", rotation=30)

    if observations:
        for raw_ax in raw_axes + [ax]:
            for obs_time, _ in observations:
                raw_ax.axvline(obs_time, color="tab:orange", alpha=0.4, linewidth=1)
        for obs_time, note in observations:
            if not note:
                continue
            ax.annotate(
                note,
                xy=(obs_time, 0.99),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="black",
            )

    ax.set_title("Sensor Readings")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
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
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
