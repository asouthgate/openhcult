#!/usr/bin/env python3
"""Plot sensor time series from the configured SQLite database."""

from __future__ import annotations

import argparse
import configparser
import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        "SELECT sr.adjusted_time_ms, d.name, d.address, sr.sensor, sr.measurement "
        "FROM sensor_readings sr "
        "JOIN devices d ON sr.device_id = d.id "
        "ORDER BY sr.adjusted_time_ms ASC, sr.id ASC"
    )
    with sqlite3.connect(db_path) as conn:
        for time_ms, device_name, device_addr, sensor_name, value in conn.execute(query):
            if time_ms is None:
                continue
            timestamp = np.datetime64(int(time_ms), "ms")
            label = device_name or device_addr or "unknown"
            series.setdefault(f"{label}:{sensor_name}", []).append(
                (timestamp, int(value))
            )
    return series


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot sensor time series from the configured SQLite database."
    )
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "openhcult.conf"),
        help="Path to openhcult.conf (default: repo root)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override database path (otherwise read from config)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write PNG to this path instead of showing a window",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    db_path = Path(args.db) if args.db else _load_db_path(config_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Missing database: {db_path}")

    if args.out:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    series = _fetch_series(db_path)
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
        ax.scatter(times, values, label=name, s=18, alpha=0.7)

        raw_ax = raw_axes[idx]
        raw_ax.scatter(times, values, label=name, s=18, alpha=0.8)
        raw_ax.legend()
        raw_ax.set_title(f"{name} raw")
        raw_ax.set_xlabel("Timestamp")
        raw_ax.set_ylabel("Value")
        raw_ax.xaxis.set_major_locator(locator)
        raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        raw_ax.tick_params(axis="x", rotation=30)

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
