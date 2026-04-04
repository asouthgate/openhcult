import os
import json
from urllib import request

BASE_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Sensor voltage range derived from calibration CSV averages across all sensors
# dry soil (0 ml): mean of sensor1_voltage + sensor2_voltage across A/B/C = 2398 mV
# saturated (75 ml): mean of sensor1_voltage + sensor2_voltage across A/B/C = 942 mV
SENSOR_DRY_MV = 2398
SENSOR_WET_MV = 942


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
