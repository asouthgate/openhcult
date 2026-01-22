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

from hcultutils.inference import (
    compute_ewma,
    compute_zscore,
    detect_hysteresis,
    detect_z_triggers,
    merge_events,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer events from recent sensor data and store observations."
    )
    parser.add_argument(
        "--ctrl-url",
        default="http://127.0.0.1:8000",
        help="hcultctrl base URL",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Lookback window in hours",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit number of rows when querying hcultctrl",
    )
    parser.add_argument(
        "--diff-lag",
        type=int,
        default=1,
        help="Lag (in samples) for d(t) = s(t) - s(t-h)",
    )
    parser.add_argument(
        "--mad-window",
        type=int,
        default=50,
        help="Window size (in samples) for rolling MAD",
    )
    parser.add_argument(
        "--mad-scale",
        type=float,
        default=1.4826,
        help="Scale factor for MAD -> sigma",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.1,
        help="EWMA alpha for baseline (0 < alpha <= 1)",
    )
    parser.add_argument(
        "--z-pvalue",
        type=float,
        default=0.000001,
        help="Two-sided p-value threshold for z(t) triggers",
    )
    parser.add_argument(
        "--resid-threshold",
        type=float,
        default=15.0,
        help="Absolute residual threshold for hysteresis test",
    )
    parser.add_argument(
        "--hyst-window",
        type=int,
        default=20,
        help="Window size (in samples) around trigger for hysteresis",
    )
    parser.add_argument(
        "--hyst-k",
        type=int,
        default=3,
        help="Required count of |r(t)| > threshold within window",
    )
    parser.add_argument(
        "--merge-distance-sec",
        type=int,
        default=240,
        help="Minimum seconds between stored events",
    )
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)
    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")

    series = _fetch_series_from_ctrl(
        args.ctrl_url, start_utc=start_utc, end_utc=end_utc, limit=args.limit
    )
    if not series:
        print("No sensor readings found.")
        return 0

    inserted = 0
    for key, points in series.items():
        times_ms = np.array([t for t, _ in points], dtype=np.int64)
        values = np.array([v for _, v in points], dtype=float)
        baseline = compute_ewma(values, args.ewma_alpha)
        zscores = compute_zscore(
            values, lag=args.diff_lag, window=args.mad_window, c=args.mad_scale
        )
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
        for trigger_idx in merged:
            if trigger_idx < 0 or trigger_idx >= times_ms.size:
                continue
            observed_at = _iso_utc(int(times_ms[trigger_idx]))
            note = (
                "AUTO: "
                f"{key} "
                f"z_p={args.z_pvalue} "
                f"ewma={args.ewma_alpha} "
                f"lag={args.diff_lag} "
                f"madw={args.mad_window} "
                f"mads={args.mad_scale} "
                f"rthr={args.resid_threshold} "
                f"hwin={args.hyst_window} "
                f"hk={args.hyst_k}"
            )
            _post_observation(args.ctrl_url, note, observed_at)
            inserted += 1

    print(f"Inserted {inserted} AUTO observations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
