import json
import urllib.error as error
import urllib.request as request

from datetime import datetime, timezone


def format_observed_at(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return value


def pretty_print(value, indent=0):
    spacer = " " * indent
    if isinstance(value, dict):
        print(f"{spacer}{{")
        items = list(value.items())
        for idx, (key, val) in enumerate(items):
            key_str = json.dumps(str(key))
            print(f"{spacer}  {key_str}: ", end="")
            pretty_print(val, indent + 2)
            if idx < len(items) - 1:
                print(",")
            else:
                print()
        print(f"{spacer}}}", end="")
        return
    if isinstance(value, list):
        print(f"{spacer}[")
        for idx, item in enumerate(value):
            pretty_print(item, indent + 2)
            if idx < len(value) - 1:
                print(",")
            else:
                print()
        print(f"{spacer}]", end="")
        return
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        print(f"\"{escaped}\"", end="")
        return
    print(json.dumps(value), end="")


def request_ctrl(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body).get("detail")
        except ValueError:
            detail = None
        message = detail or body or exc.reason
        raise SystemExit(f"hcultutils: {exc.code} {message}") from exc
