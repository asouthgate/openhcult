#!/usr/bin/env python3
"""Infer events from recent sensor data and store as observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import urllib

import numpy as np

from hcultutils.fetch_data import fetch_data
from hcultinf.detection import SegmentDetector

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Lowest level to capture everything


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

def _has_close_neighbors(sensor, event_time, prev_sensor_times, merge_distance_sec):
    """
    Checks if any other sensor has already logged an event within the merge distance.
    """
    
    for other_sensor, logged_times in prev_sensor_times:
        # print(sensor, other_sensor)
        if other_sensor != sensor:
            continue
            
        for prev_time in logged_times:
            # Check if the time difference is within our threshold
            if abs(event_time - prev_time).seconds <= merge_distance_sec:
                print("Detected duplicate")
                return True
            else:
                print("No duplicate", abs(event_time - prev_time).seconds)
                
    return False


def _get_non_duplicate_events(series, prev_sensor_times, merge_distance_sec, emwa_tau_minutes, trigger_threshold, release_threshold):

    result = []
    print(prev_sensor_times)
    for sensor, points in series.items():
        times = np.array([t for t, _ in points])
        values = np.array([v for _, v in points], dtype=float)
        ded = SegmentDetector(times, values, emwa_tau_minutes, trigger_threshold, release_threshold)
        ded.debug_plot()
        events = ded.get_watering_events()
        sensor_event_times = [event.start for event in events]

        for event_time in sensor_event_times:

            if not _has_close_neighbors(sensor, event_time, prev_sensor_times, merge_distance_sec):
                note = (
                    "AUTO: "
                    f"{sensor} "
                    f"emwa_tau_minutes={emwa_tau_minutes} "
                )
                iso_string = str(event_time.astype('datetime64[ms]')) + "Z"
                result.append((note, iso_string))
    return result


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

    prev_sensor_times = sorted(
        (sensor, obs_time) for sensor, obs_time, note in observations if note.startswith("AUTO:")
    )

    print(prev_sensor_times)

    events = _get_non_duplicate_events(
        series,
        prev_sensor_times,
        args.merge_distance_seconds,
        args.emwa_tau_minutes,
        args.trigger_threshold,
        args.release_threshold
    )
    
    for note, observed_at in events:
        _post_observation(ctrl_url, note, observed_at)

    print(f"Inserted {len(events)} AUTO observations.")
    return 0
