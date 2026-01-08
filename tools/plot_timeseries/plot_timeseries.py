#!/usr/bin/env python3
"""Plot sensor time series from the configured SQLite database."""

from __future__ import annotations

import argparse
import configparser
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
        "SELECT adjusted_time_ms, sensor, measurement "
        "FROM sensor_readings "
        "ORDER BY adjusted_time_ms ASC, id ASC"
    )
    with sqlite3.connect(db_path) as conn:
        for time_ms, sensor_name, value in conn.execute(query):
            if time_ms is None:
                continue
            timestamp = np.datetime64(int(time_ms), "ms")
            series.setdefault(sensor_name, []).append((timestamp, int(value)))
    return series


def _gaussian_average(values: np.ndarray, window: int, sigma: float) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    smoothed = np.empty_like(values, dtype=float)
    for i in range(values.size):
        start = max(0, i - half)
        end = min(values.size, i + half + 1)
        window_values = values[start:end]
        window_weights = weights[(start - i + half):(end - i + half)]
        weight_sum = float(window_weights.sum())
        if weight_sum == 0:
            smoothed[i] = float(window_values.mean())
        else:
            smoothed[i] = float((window_values * window_weights).sum() / weight_sum)
    return smoothed


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
    if len(sensor_names) > 2:
        sensor_names = sensor_names[:2]

    locator = mdates.AutoDateLocator()
    has_two = len(sensor_names) == 2
    fig = plt.figure(figsize=(12, 8))
    if has_two:
        raw_axes = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2),
        ]
        ax = fig.add_subplot(2, 1, 2)
    else:
        raw_axes = [fig.add_subplot(2, 1, 1)]
        ax = fig.add_subplot(2, 1, 2)

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = [t for t, _ in points]
        values = np.array([v for _, v in points], dtype=float)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax == vmin:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - vmin) / (vmax - vmin)
        window = min(5, len(normalized))
        sigma = max(1.0, window / 2.0)
        smoothed = _gaussian_average(normalized, window, sigma)
        ax.plot(times, smoothed, label=f"{name} (smoothed)")
        ax.scatter(times, normalized, label=f"{name} (raw)", s=18, alpha=0.7)

        raw_ax = raw_axes[min(idx, len(raw_axes) - 1)]
        raw_window = min(5, len(values))
        raw_sigma = max(1.0, raw_window / 2.0)
        raw_smoothed = _gaussian_average(values, raw_window, raw_sigma)
        raw_ax.plot(times, raw_smoothed, label=f"{name} (smoothed)")
        raw_ax.scatter(times, values, label=f"{name} (raw)", s=18, alpha=0.8)
        raw_ax.legend()
        raw_ax.set_title(f"{name} raw")
        raw_ax.set_xlabel("Timestamp")
        raw_ax.set_ylabel("Value")
        raw_ax.xaxis.set_major_locator(locator)
        raw_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        raw_ax.tick_params(axis="x", rotation=30)

    ax.set_title("Sensor Readings (Normalized)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Normalized (min-max)")
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
