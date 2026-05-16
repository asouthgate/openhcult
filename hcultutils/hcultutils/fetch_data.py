#!/usr/bin/env python3
"""Plot sensor time series from the configured database."""

from __future__ import annotations

import configparser
import pickle
import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
import numpy as np

from hcultutils.query import request_ctrl


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
) -> Dict[str, List[Tuple[np.datetime64, int, int | None]]]:
    series: Dict[str, List[Tuple[np.datetime64, int, int | None]]] = {}
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
    payload = request_ctrl("GET", url)

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
        key = f"{row['device_address']}:{row['sensor']}"
        voltage_mv = row.get("voltage_mv")
        series.setdefault(key, []).append(
            (
                timestamp,
                int(row.get("measurement", 0)),
                int(voltage_mv) if voltage_mv is not None else None,
            )
        )
    return series


def fetch_observations(
    ctrl_url: str,
    *,
    start_utc: str | None = None,
    end_utc: str | None = None,
    plant_name: str | None = None,
    limit: int = 1000,
) -> List[Tuple[int, np.datetime64, str]]:
    params: Dict[str, str] = {"limit": str(limit)}
    if start_utc:
        params["start_utc"] = start_utc
    if end_utc:
        params["end_utc"] = end_utc
    if plant_name:
        params["plant"] = plant_name

    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/observations?{urllib.parse.urlencode(params)}"
    payload = request_ctrl("GET", url)

    observations: List[Tuple[int, np.datetime64, str]] = []
    for row in payload.get("data", []):
        observed_at = row.get("observed_at")
        if observed_at is None:
            continue
        timestamp = np.datetime64(int(observed_at), "ms")
        observations.append(
            (int(row.get("id", 0)), timestamp, str(row.get("note", "")))
        )
    return observations


def fetch_sensor_data(
    ctrl_url: str,
    *,
    sensor: str | None = None,
    device: str | None = None,
    plant_name: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    limit: int = 1000,
) -> Dict[str, List[Tuple[np.datetime64, int, int | None]]]:
    return fetch_series_from_ctrl(
        ctrl_url,
        sensor=sensor,
        device=device,
        plant_name=plant_name,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=limit,
    )


def fetch_data(
    ctrl_url,
    start_utc,
    end_utc,
    *,
    sensor=None,
    device=None,
    plant_name=None,
    limit=1000,
):
    series = fetch_sensor_data(
        ctrl_url,
        sensor=sensor,
        device=device,
        plant_name=plant_name,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=limit,
    )
    observations = fetch_observations(
        ctrl_url,
        start_utc=start_utc,
        end_utc=end_utc,
        plant_name=plant_name,
        limit=limit,
    )
    if not series:
        print("No data found.")
        return None, None
    return series, observations


def main(args) -> int:
    if args.out:
        matplotlib.use("Agg")

    fetched = fetch_data(
        ctrl_url=args.ctrl_url,
        sensor=args.sensor,
        device=args.device,
        plant_name=args.plant_name,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        limit=args.limit,
    )
    with open(f"sensor-data-{args.start_utc}_{args.end_utc}.pkl", "wb") as f:
        pickle.dump(fetched, f)


def fetch_observations_main(args) -> int:
    observations = fetch_observations(
        ctrl_url=args.ctrl_url,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        plant_name=args.plant_name,
        limit=args.limit,
    )
    if not observations:
        print("No observations found.")
        return 1
    with open(f"observations-{args.start_utc}_{args.end_utc}.pkl", "wb") as f:
        pickle.dump(observations, f)
    print(f"Saved {len(observations)} observations to observations-{args.start_utc}_{args.end_utc}.pkl")
    return 0


def fetch_chords(
    ctrl_url: str,
    *,
    plant_name: str,
    offset_ms: int = 600_000,
    width_ms: int = 3_000_000,
    sensor: str | None = None,
    device_address: str | None = None,
) -> dict:
    params: Dict[str, str] = {
        "plant": plant_name,
        "offset_ms": str(offset_ms),
        "width_ms": str(width_ms),
    }
    if sensor:
        params["sensor"] = sensor
    if device_address:
        params["device_address"] = device_address

    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/chords?{urllib.parse.urlencode(params)}"
    return request_ctrl("GET", url)


def fetch_chords_main(args) -> int:
    result = fetch_chords(
        ctrl_url=args.ctrl_url,
        plant_name=args.plant_name,
        offset_ms=args.offset_ms,
        width_ms=args.width_ms,
        sensor=args.sensor,
        device_address=args.device_address,
    )
    n = len(result.get("chord_times", []))
    if not n:
        print("No chords found.")
        return 1
    start = args.start_utc or "unknown"
    end = args.end_utc or "unknown"
    out_path = f"chords-{args.plant_name}-{start}_{end}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {n} chords to {out_path}")
    return 0


def fetch_prior(
    ctrl_url: str,
) -> dict:
    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/prior"
    return request_ctrl("GET", url)


def fetch_prior_main(args) -> int:
    from datetime import datetime, timezone

    result = fetch_prior(ctrl_url=args.ctrl_url)
    if not result.get("prior_x"):
        print("No prior data found.")
        return 1
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result["downloaded_at"] = ts
    out_path = f"prior-{ts}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved prior data ({len(result['prior_x'])} points) to {out_path}")
    return 0


def fetch_sensor_data_main(args) -> int:
    series = fetch_sensor_data(
        ctrl_url=args.ctrl_url,
        sensor=args.sensor,
        device=args.device,
        plant_name=args.plant_name,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        limit=args.limit,
    )
    if not series:
        print("No sensor data found.")
        return 1
    plant = f"{args.plant_name}-" if args.plant_name else ""
    out_path = f"sensor-data-{plant}{args.start_utc}_{args.end_utc}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(series, f)
    total = sum(len(v) for v in series.values())
    print(f"Saved {total} readings across {len(series)} series to {out_path}")
    return 0
