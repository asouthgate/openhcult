import time
from typing import Optional
from hcultutils.query import request_ctrl


def _record_observation(
    base_url: str, plant_name: str, note: str, observed_at: Optional[str] = None
) -> int:
    payload = {"plant_name": plant_name, "note": note, "observed_at": observed_at}
    request_ctrl("POST", f"{base_url}/observations", payload)


def observations_main(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "record":
        return _record_observation(
            base, args.plant_name, args.note, getattr(args, "observed_at", None)
        )
    return 0
