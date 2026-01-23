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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


from hcultinf.inference import compute_ewma, compute_zscore


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


def _fetch_observations(db_path: Path) -> List[Tuple[int, np.datetime64, str]]:
    observations: List[Tuple[int, np.datetime64, str]] = []
    query = "SELECT id, observed_at, note FROM observations ORDER BY observed_at ASC, id ASC"
    with sqlite3.connect(db_path) as conn:
        try:
            for obs_id, observed_at, note in conn.execute(query):
                if observed_at is None:
                    continue
                timestamp = np.datetime64(int(observed_at), "ms")
                observations.append((int(obs_id), timestamp, str(note)))
        except sqlite3.OperationalError:
            return []
    return observations


def _fetch_series_from_ctrl(
    ctrl_url: str,
    *,
    sensor: str | None,
    device: str | None,
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
    start_utc: str | None,
    end_utc: str | None,
    limit: int,
) -> List[Tuple[int, np.datetime64, str]]:
    params: Dict[str, str] = {"limit": str(limit)}
    if start_utc:
        params["start_utc"] = start_utc
    if end_utc:
        params["end_utc"] = end_utc

    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/observations?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)

    observations: List[Tuple[int, np.datetime64, str]] = []
    for row in payload.get("data", []):
        observed_at = row.get("observed_at")
        if observed_at is None:
            continue
        timestamp = np.datetime64(int(observed_at), "ms")
        observations.append((int(row.get("id", 0)), timestamp, str(row.get("note", ""))))
    return observations


def _plot_raw_subsensor_readings(ax, raw_ax, times, values, ewma, name, args, locator):
    ax.plot(times, values, label=name, linewidth=1.2)
    ax.plot(times, ewma, label=f"{name} EWMA", linewidth=1.2, linestyle="--")
    raw_ax.plot(times, values, label=name, linewidth=1.2)
    raw_ax.plot(times, ewma, label="EWMA", linewidth=1.2, linestyle="--")
    raw_ax.legend()
    raw_ax.set_title(name)
    raw_ax.set_xlabel("Timestamp")
    raw_ax.set_ylabel("Value")
    raw_ax.set_ylim(1, 2500)
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


def _fetch_data(args):
    observations: List[Tuple[int, np.datetime64, str]] = []
    if args.ctrl_url:
        series = _fetch_series_from_ctrl(
            args.ctrl_url,
            sensor=args.sensor,
            device=args.device,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            limit=args.limit,
        )
        observations = _fetch_observations_from_ctrl(
            args.ctrl_url,
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
    return series, observations


def main(args) -> int:
    if args.out:
        matplotlib.use("Agg")

    # observations: List[Tuple[int, np.datetime64, str]] = []
    # if args.ctrl_url:
    #     series = _fetch_series_from_ctrl(
    #         args.ctrl_url,
    #         sensor=args.sensor,
    #         device=args.device,
    #         start_utc=args.start_utc,
    #         end_utc=args.end_utc,
    #         limit=args.limit,
    #     )
    #     observations = _fetch_observations_from_ctrl(
    #         args.ctrl_url,
    #         start_utc=args.start_utc,
    #         end_utc=args.end_utc,
    #         limit=args.limit,
    #     )
    # else:
    #     config_path = Path(args.config)
    #     if not config_path.exists():
    #         raise FileNotFoundError(f"Missing config: {config_path}")

    #     db_path = Path(args.db) if args.db else _load_db_path(config_path)
    #     if not db_path.exists():
    #         raise FileNotFoundError(f"Missing database: {db_path}")
    #     series = _fetch_series(db_path)
    #     observations = _fetch_observations(db_path)
    # if not series:
    #     print("No sensor readings found.")
    #     return 0

    series, observations = _fetch_data(args)
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
    values_map: Dict[str, np.ndarray] = {}
    baseline_map: Dict[str, np.ndarray] = {}
    times_map: Dict[str, np.ndarray] = {}

    for idx, name in enumerate(sensor_names):
        points = series[name]
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        ewma = compute_ewma(values, args.ewma_alpha)
        zscores = compute_zscore(
            values, lag=args.diff_lag, window=args.mad_window, c=args.mad_scale
        )
        zscores_map[name] = zscores
        values_map[name] = values
        baseline_map[name] = ewma
        times_map[name] = times
        _plot_raw_subsensor_readings(ax, raw_axes[idx], times, values, ewma, name, args, locator)

    if observations:
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

    ax.set_title("Sensor Readings")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.set_ylim(1, 2500)
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

    rfig = plt.figure(figsize=(14, 4 + 4 * math.ceil(len(sensor_names) / 2)))
    rgrid = rfig.add_gridspec(rows + 1, cols)
    rraw_axes = []
    for i in range(len(sensor_names)):
        r = i // cols
        c = i % cols
        rraw_axes.append(rfig.add_subplot(rgrid[r, c]))

    rax = rfig.add_subplot(rgrid[rows, :])

    for idx, name in enumerate(sensor_names):
        times = times_map[name]
        zscores = zscores_map[name]
        residuals = values_map[name] - baseline_map[name]
        _plot_z_subsensor_readings(zax, zraw_axes[idx], times, zscores, name, args, locator)    
        _plot_residual_subsensor_readings(rax, rraw_axes[idx], times, residuals, name, args, locator)   

    zax.set_title("z(t) = d(t) / (c * MAD)")
    zax.set_xlabel("Timestamp")
    zax.set_ylabel("z(t)")
    zax.set_yscale("symlog", linthresh=1.0)
    zax.xaxis.set_major_locator(locator)
    zax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    zax.legend()
    zfig.autofmt_xdate()
    zfig.tight_layout()

    rax.set_title("Residuals x(t) - B(t)")
    rax.set_xlabel("Timestamp")
    rax.set_ylabel("x(t) - B(t)")
    rax.xaxis.set_major_locator(locator)
    rax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    rax.legend()
    rfig.autofmt_xdate()
    rfig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        z_out = out_path.with_name(f"{out_path.stem}_z{out_path.suffix}")
        zfig.savefig(z_out, dpi=150)
        print(f"Wrote {z_out}")
        r_out = out_path.with_name(f"{out_path.stem}_resid{out_path.suffix}")
        rfig.savefig(r_out, dpi=150)
        print(f"Wrote {r_out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
