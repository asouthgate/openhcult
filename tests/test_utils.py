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


def simulate_moisture(
    n_points, hours, base=2200, watering_at_fraction=0.55, noise_seed=42
):
    """Return list of (timestamp_ms, value) simulating a drying curve with one watering event."""
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
        rows.append((t, max(600, min(3000, int(value)))))
    return rows
