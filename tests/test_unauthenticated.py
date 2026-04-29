import os
import pytest
from urllib import request, error

BASE_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

_PROTECTED = [
    ("GET", "/plants"),
    ("GET", "/species"),
    ("GET", "/devices"),
    ("GET", "/observations"),
    ("GET", "/timeseries"),
    ("GET", "/plant_sensors"),
    ("GET", "/water_calibration"),
]


@pytest.mark.parametrize("method,path", _PROTECTED)
def test_requires_auth(method, path):
    req = request.Request(
        f"{BASE_URL}{path}",
        method=method,
        headers={"Accept": "application/json"},
    )
    with pytest.raises(error.HTTPError) as exc:
        request.urlopen(req, timeout=10)
    assert exc.value.code in (401, 503)
