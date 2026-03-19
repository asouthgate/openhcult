import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Optional
from hcultutils.query import request_ctrl


def _record_observation(
    base_url: str, plant_name: str, note: str, observed_at: Optional[str] = None
) -> int:
    payload = {"plant_name": plant_name, "note": note, "observed_at": observed_at}
    request_ctrl("POST", f"{base_url}/observations", payload)


def _list_observations(base_url: str, hours: int, limit: int) -> int:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    params = urllib.parse.urlencode(
        {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "limit": limit,
        }
    )
    result = request_ctrl("GET", f"{base_url}/observations?{params}")
    rows = result.get("data", [])
    if not rows:
        print("No observations found.")
        return 0
    for row in rows:
        ts = datetime.fromtimestamp(
            row["observed_at"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        plant = row.get("plant_name") or "-"
        print(f"{row['id']:6}  {ts}  {plant:<30}  {row['note']}")
    return 0


def observations_main(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "record":
        return _record_observation(
            base, args.plant_name, args.note, getattr(args, "observed_at", None)
        )
    if action == "ls":
        return _list_observations(base, args.hours, args.limit)
    return 0
