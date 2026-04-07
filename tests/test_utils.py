import os
import json
from urllib import request

BASE_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SMOKE_USERNAME = os.environ.get("OPENHCULT_SMOKE_USERNAME", "admin")
SMOKE_PASSWORD = os.environ.get("OPENHCULT_SMOKE_PASSWORD", "admin")

# Sensor voltage range derived from calibration CSV averages across all sensors
# dry soil (0 ml): mean of sensor1_voltage + sensor2_voltage across A/B/C = 2398 mV
# saturated (75 ml): mean of sensor1_voltage + sensor2_voltage across A/B/C = 942 mV
SENSOR_DRY_MV = 2398
SENSOR_WET_MV = 942

_token = None


def _get_token():
    global _token
    if _token:
        return _token
    data = json.dumps({"username": SMOKE_USERNAME, "password": SMOKE_PASSWORD}).encode()
    req = request.Request(
        f"{BASE_URL}/auth/login",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with request.urlopen(req, timeout=10) as resp:
        _token = json.loads(resp.read().decode())["access_token"]
    return _token


def request_json(path, method="GET", payload=None):
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json", "Authorization": f"Bearer {_get_token()}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)
