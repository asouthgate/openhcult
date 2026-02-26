#!/usr/bin/env python3
"""Infer events from recent sensor data and store as observations."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

# from hcultinf.inference import (
#     compute_zscore,
#     detect_hysteresis,
#     detect_z_triggers,
#     merge_events,
# )

from hcultinf.inference import classify_events

def _iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch_series_from_ctrl(
    ctrl_url: str,
    *,
    start_utc: str,
    end_utc: str,
    limit: int,
) -> Dict[str, List[Tuple[int, int]]]:
    series: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    params: Dict[str, str] = {
        "format": "json",
        "limit": str(limit),
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/timeseries?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)
    for row in payload.get("data", []):
        time_ms = row.get("adjusted_time_ms")
        if time_ms is None:
            continue
        device_name = row.get("device_name") or row.get("device_address") or "unknown"
        sensor_name = row.get("sensor") or "sensor"
        key = f"{device_name}:{sensor_name}"
        series[key].append((int(time_ms), int(row.get("measurement", 0))))
    for key in series:
        series[key].sort(key=lambda item: item[0])
    return series


def _post_observation(ctrl_url: str, note: str, observed_at: str) -> None:
    payload = json.dumps({"note": note, "observed_at": observed_at}).encode("utf-8")
    req = urllib.request.Request(
        f"{ctrl_url.rstrip('/')}/observations",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def _fetch_observations_from_ctrl(
    ctrl_url: str,
    *,
    start_utc: str,
    end_utc: str,
    limit: int,
) -> List[Tuple[int, str]]:
    params: Dict[str, str] = {
        "limit": str(limit),
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    base_url = ctrl_url.rstrip("/")
    url = f"{base_url}/observations?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)
    observations: List[Tuple[int, str]] = []
    for row in payload.get("data", []):
        observed_at = row.get("observed_at")
        if observed_at is None:
            continue
        observations.append((int(observed_at), str(row.get("note", ""))))
    return observations


def _parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def run(args: argparse.Namespace) -> int:
    ctrl_url = args.ctrl_url or "http://127.0.0.1:8000"
    if args.start_utc or args.end_utc:
        end = _parse_utc(args.end_utc) if args.end_utc else datetime.now(timezone.utc)
        start = _parse_utc(args.start_utc) if args.start_utc else end - timedelta(hours=args.hours)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=args.hours)
    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")

    series = _fetch_series_from_ctrl(
        ctrl_url, start_utc=start_utc, end_utc=end_utc, limit=args.limit
    )
    if not series:
        print("No sensor readings found.")
        return 0

    obs_start = (start - timedelta(seconds=args.merge_distance_sec)).isoformat().replace(
        "+00:00", "Z"
    )
    obs_end = (end + timedelta(seconds=args.merge_distance_sec)).isoformat().replace(
        "+00:00", "Z"
    )
    observations = _fetch_observations_from_ctrl(
        ctrl_url, start_utc=obs_start, end_utc=obs_end, limit=args.limit
    )
    auto_times = sorted(
        obs_time for obs_time, note in observations if note.startswith("AUTO:")
    )
    auto_times_np = np.array(auto_times, dtype=np.int64)

    inserted = 0
    for key, points in series.items():
        times_ms = np.array([t for t, _ in points], dtype=np.int64)
        values = np.array([v for _, v in points], dtype=float)
        triggers, run_lengths, starts = classify_events(values, args.diff_lag, args.mad_window, args.mad_scale, args.z_pvalue)

        start_indexes = np.where(starts)[0]
        for trigger_idx in start_indexes:
            if trigger_idx < 0 or trigger_idx >= times_ms.size:
                continue
            observed_at = _iso_utc(int(times_ms[trigger_idx]))
            if auto_times_np.size:
                candidate = int(times_ms[trigger_idx])
                pos = int(np.searchsorted(auto_times_np, candidate))
                nearby = []
                if pos > 0:
                    nearby.append(auto_times_np[pos - 1])
                if pos < auto_times_np.size:
                    nearby.append(auto_times_np[pos])
                if any(abs(candidate - t) <= args.merge_distance_sec * 1000 for t in nearby):
                    continue
            note = (
                "AUTO: "
                f"{key} "
                f"z_p={args.z_pvalue} "
                f"lag={args.diff_lag} "
                f"madw={args.mad_window} "
                f"mads={args.mad_scale} "
            )
            _post_observation(ctrl_url, note, observed_at)
            inserted += 1

    print(f"Inserted {inserted} AUTO observations.")
    return 0
