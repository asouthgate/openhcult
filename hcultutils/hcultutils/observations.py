import time
from typing import Optional
from hcultutils.query import request_ctrl


def _record_observation(
    base_url: str, plant_id: int, note: str, observed_at: Optional[str] = None
) -> int:
    """
    Sends a POST request to create a new observation for a specific plant.
    """
    # Construct the payload based on the ObservationIn schema
    payload = {"plant_id": plant_id, "note": note, "observed_at": observed_at}

    response = request_ctrl("POST", f"{base_url}/observations", payload)


def observations_main(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "record":
        # Mapping args to the function:
        # Assuming args.plant_id, args.note, and optionally args.observed_at
        return _record_observation(
            base, args.plant_id, args.note, getattr(args, "observed_at", None)
        )
    return 0
