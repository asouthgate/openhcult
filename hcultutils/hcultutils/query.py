import json
import urllib.error as error
import urllib.request as request

from hcultutils.token import load_token


def request_ctrl(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    token = load_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                "hcultutils: not authenticated — run `hcultutils login --ctrl-url <url>`"
            ) from exc
        body = exc.read().decode("utf-8")
        try:
            detail = json.loads(body).get("detail")
        except ValueError:
            detail = None
        message = detail or body or exc.reason
        raise SystemExit(f"hcultutils: {exc.code} {message}") from exc
