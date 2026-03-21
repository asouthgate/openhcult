import os
import json
import math
import time
import random
from urllib import request

BASE_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request_json(path, method="GET", payload=None):
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


_RAW_TO_MV = 3900 / 4095


def simulate_moisture(
    n_points, hours, base=2200, watering_at_fraction=0.55, noise_seed=42
):
    """Return list of (timestamp_ms, raw, voltage_mv) simulating a drying curve with one watering event."""
    rng = random.Random(noise_seed)
    noise = 40
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - hours * 3600 * 1000
    step_ms = hours * 3600 * 1000 // n_points
    watering_i = int(n_points * watering_at_fraction)
    value = float(base)
    rows = []
    for i in range(n_points):
        t = start_ms + i * step_ms
        if i == watering_i:
            value = 800 + rng.uniform(-30, 30)
        elif i > watering_i:
            recovery = base * (1 - math.exp(-(i - watering_i) / (n_points * 0.08)))
            value = 800 + recovery + rng.uniform(-noise, noise)
        else:
            value = base - 0.15 * i + rng.uniform(-noise, noise)
        raw = max(600, min(3000, int(value)))
        rows.append((t, raw, int(raw * _RAW_TO_MV)))
    return rows


def simulate_moisture_multi(
    n_events, days_per_cycle=3, base=2200, wet=800, noise_seed=42
):
    """Return (readings, watering_times_ms) with n_events watering events.

    Readings are at 10-minute intervals over n_events * days_per_cycle days.
    Each cycle: sensor dries from wet floor back toward base, then drops at watering.
    """
    rng = random.Random(noise_seed)
    now_ms = int(time.time() * 1000)
    cycle_ms = days_per_cycle * 24 * 3600 * 1000
    total_ms = n_events * cycle_ms
    start_ms = now_ms - total_ms
    step_ms = 30 * 60 * 1000

    jitter = cycle_ms // 4
    watering_times = sorted(
        int(start_ms + i * cycle_ms + cycle_ms // 2 + rng.randint(-jitter, jitter))
        for i in range(n_events)
    )

    dry_tau_h = days_per_cycle * 24 * 0.55
    recovery_tau_h = 3.0

    readings = []
    last_water_ms = start_ms - cycle_ms
    next_idx = 0

    n_points = total_ms // step_ms
    for i in range(n_points):
        t = start_ms + i * step_ms
        if next_idx < len(watering_times) and t >= watering_times[next_idx]:
            last_water_ms = watering_times[next_idx]
            next_idx += 1
        h = (t - last_water_ms) / 3_600_000
        quick = (base - wet) * 0.25 * (1 - math.exp(-h / recovery_tau_h))
        slow = (base - wet) * 0.75 * (1 - math.exp(-h / dry_tau_h))
        value = wet + quick + slow + rng.uniform(-40, 40)
        raw = max(600, min(3000, int(value)))
        readings.append((t, raw, int(raw * _RAW_TO_MV)))

    return readings, watering_times
