import json
import urllib.error as error
import urllib.request as request

from datetime import datetime, timezone


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
