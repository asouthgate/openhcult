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
import pickle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "openhcult" / "openhcult.conf"
    return Path.home() / ".config" / "openhcult" / "openhcult.conf"


def _load_db_url(config_path: Path) -> str:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if "database" not in parser or "url" not in parser["database"]:
        raise ValueError(f"Missing database.url in {config_path}")
    url = parser["database"]["url"].strip()
    if not url:
        raise ValueError(f"Empty database.url in {config_path}")
    return url


def fetch_series_from_ctrl(
    ctrl_url: str,
    *,
    sensor: str | None,
    device: str | None,
    plant_name: str | None,
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
    if plant_name:
        params["plant"] = plant_name
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

def fetch_data(args):
    observations: List[Tuple[int, np.datetime64, str]] = []
    if args.ctrl_url:
        series = fetch_series_from_ctrl(
            args.ctrl_url,
            sensor=args.sensor,
            device=args.device,
            plant_name=args.plant_name,
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

        db_url = args.db if args.db else _load_db_url(config_path)
        if db_url.startswith("sqlite:////"):
            db_path = Path(db_url.replace("sqlite:////", "/"))
            if not db_path.exists():
                raise FileNotFoundError(f"Missing database: {db_path}")
        series = _fetch_series(db_url)
        observations = _fetch_observations(db_url)
    if not series:
        print("No sensor readings found.")
        return None
    return series, observations


def main(args) -> int:
    if args.out:
        matplotlib.use("Agg")

    fetched = fetch_data(args)
    print(fetched)
    with open(f"sensor-data-{args.start_utc}_{args.end_utc}.pkl", "wb") as f:
        pickle.dump(fetched, f)


if __name__ == "__main__":
    raise SystemExit(main())
