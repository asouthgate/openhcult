#!/usr/bin/env python3
"""Infer events from recent sensor data and store as observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import urllib

import numpy as np

from hcultinf.inference import classify_events
from hcultutils.fetch_data import fetch_data


def _iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _get_non_duplicate_events(series, prev_event_times, diff_lag, mad_window, mad_scale, z_pvalue, merge_distance_sec): 
    events = [] # (note, observed_at)
    for sensor, points in series.items():
        times_ms = np.array([t for t, _ in points], dtype=np.int64)
        values = np.array([v for _, v in points], dtype=float)
        _, _, starts = classify_events(values, diff_lag, mad_window, mad_scale, z_pvalue)
        starts_t = sorted(times_ms[starts])

        for tt in starts_t:
            candidate_neighbors = np.array(list(prev_event_times) + [tk for tk in starts_t if tt != tk])
            observed_at = _iso_utc(int(tt))
            if candidate_neighbors.size:
                candidate = int(tt)
                pos = int(np.searchsorted(candidate_neighbors, candidate))
                nearby = []
                if pos > 0:
                    nearby.append(candidate_neighbors[pos - 1])
                if pos < prev_event_times.size:
                    nearby.append(candidate_neighbors[pos])
                if any(abs(candidate - t) <= merge_distance_sec * 1000 for t in nearby):
                    continue
            note = (
                "AUTO: "
                f"{sensor} "
                f"z_p={z_pvalue} "
                f"lag={diff_lag} "
                f"madw={mad_window} "
                f"mads={mad_scale} "
            )
            events.append((note, observed_at))
    return events


def run(args: argparse.Namespace) -> int:
    ctrl_url = args.ctrl_url or "http://127.0.0.1:8000"
    print(args)
    if args.start_utc or args.end_utc:
        end = _parse_utc(args.end_utc) if args.end_utc else datetime.now(timezone.utc)
        start = _parse_utc(args.start_utc) if args.start_utc else end - timedelta(hours=args.hours)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=args.hours)
    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")

    series, observations = fetch_data(
        ctrl_url, start_utc, end_utc, limit=args.limit
    )
    if not series:
        print("No sensor readings found.")
        return 0

    auto_times = sorted(
        obs_time for obs_time, note in observations if note.startswith("AUTO:")
    )
    auto_times_np = np.array(auto_times, dtype=np.int64)

    events = _get_non_duplicate_events(
        series,
        auto_times_np,
        args.diff_lag,
        args.mad_window,
        args.mad_scale,
        args.z_pvalue,
        args.merge_distance_sec,
    )

    for note, observed_at in events:
        _post_observation(ctrl_url, note, observed_at)

    print(f"Inserted {len(events)} AUTO observations.")
    return 0
