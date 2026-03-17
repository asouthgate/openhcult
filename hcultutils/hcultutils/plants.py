import json
from urllib.parse import quote

from hcultutils.query import request_ctrl
from hcultutils.formatting import (
    format_datetime_string,
    format_table,
    pretty_print,
    format_observed_at,
)


def _render_health_payload(payload):
    if isinstance(payload, dict):
        if "recent_observed_at" in payload:
            payload["recent_observed_at"] = format_observed_at(
                payload.get("recent_observed_at")
            )
        if "recent_observations" in payload and isinstance(
            payload["recent_observations"], list
        ):
            for row in payload["recent_observations"]:
                if isinstance(row, dict) and "observed_at" in row:
                    row["observed_at"] = format_observed_at(row.get("observed_at"))
        if "statuses" in payload and isinstance(payload["statuses"], list):
            latest_by_code = {}
            for row in payload["statuses"]:
                if not isinstance(row, dict):
                    continue
                code = row.get("status_code")
                observed_at = int(row.get("observed_at"))
                if not code:
                    continue
                current = latest_by_code.get(code)
                if current is None or (observed_at or 0) > (current[0] or 0):
                    latest_by_code[code] = (observed_at, row)
            deduped = []
            for observed_at, row in sorted(
                latest_by_code.values(),
                key=lambda item: item[0] or 0,
                reverse=True,
            ):
                row["observed_at"] = format_observed_at(row.get("observed_at"))
                if "cleared_at" in row:
                    row["cleared_at"] = format_observed_at(row.get("cleared_at"))
                row["status_code"] = f"\x1b[31m{row.get('status_code')}\x1b[0m"
                deduped.append(row)
            payload["statuses"] = deduped


def _list_plants(base_url: str) -> int:
    payload = request_ctrl("GET", f"{base_url}/plants?include=sensors")
    data = payload.get("data", [])
    headers = ["plant_name", "species_name", "metadata", "sensors"]
    rows = [
        [
            d.get("plant_name"),
            d.get("species_name"),
            d.get("metadata"),
            ",".join(
                [
                    f"{sensd["device_address"]}/{sensd["sensor"]}"
                    for sensd in d.get("sensors")
                ]
            ),
        ]
        for d in data
    ]

    table_output = format_table(headers, rows)
    print(table_output)
    return 0


def _list_plant_sensor_mapping(base_url: str) -> int:
    payload = request_ctrl("GET", f"{base_url}/plant_sensors")
    data = payload.get("data", [])
    headers = ["plant_name", "device_address", "sensor"]
    rows = [
        [
            d.get("plant_name"),
            d.get("device_address"),
            d.get("sensor"),
        ]
        for d in data
    ]

    table_output = format_table(headers, rows)
    print(table_output)
    return 0


def _add_plant(base_url: str, plant_name, species_name, tag, metadata) -> int:
    payload = {
        "plant_name": plant_name,
        "species_name": species_name,
        "tag": tag,
        "metadata": json.loads(metadata) if metadata else None,
    }
    created = request_ctrl("POST", f"{base_url}/plants", payload)
    print(f"Created plant {created.get('id')}")
    return 0


def _update_plant(base_url: str, species_id, tag, metadata, plant_id) -> int:
    payload = {}
    if species_id is not None:
        payload["species_id"] = species_id
    if tag is not None:
        payload["tag"] = tag
    if metadata is not None:
        payload["metadata"] = json.loads(metadata)
    updated = request_ctrl("PATCH", f"{base_url}/plants/{plant_id}", payload)
    print(f"Updated plant {updated.get('id')}")
    return 0


def _delete_plant(base_url: str, plant_name) -> int:
    deleted = request_ctrl("DELETE", f"{base_url}/plants/{plant_name}")
    print(f"Deleted plant {deleted.get('plant_name')}")
    return 0


def _get_health(base_url: str, plant_name) -> int:
    if plant_name is None:
        payload = request_ctrl("GET", f"{base_url}/plants/health")
        if isinstance(payload, dict) and "data" in payload:
            for row in payload.get("data", []):
                _render_health_payload(row)
        pretty_print(payload)
        print()
        return 0
    plant_name = quote(plant_name, safe="")
    payload = request_ctrl("GET", f"{base_url}/plants/{plant_name}/health")
    _render_health_payload(payload)
    pretty_print(payload)
    print()
    return 0


def _assign_plant(base_url: str, plant_name, device, sensor) -> int:
    plant_name = quote(plant_name, safe="")
    payload = {"device": device, "sensor": sensor}
    assigned = request_ctrl("POST", f"{base_url}/plants/{plant_name}/assign", payload)
    print(
        "Assigned plant {plant_name} to {device} sensor {sensor}".format(
            plant_name=assigned.get("plant_name") or plant_name,
            device=assigned.get("device_name")
            or assigned.get("device_address")
            or device,
            sensor=assigned.get("sensor") or sensor,
        )
    )
    return 0


def _set_status(base_url, plant_name, status_code, note) -> int:
    plant_name = quote(plant_name, safe="")
    payload = {"status_code": status_code}
    if note:
        payload["note"] = note
    created = request_ctrl("POST", f"{base_url}/plants/{plant_name}/status", payload)
    print(
        "Added status {status} to {plant_name}".format(
            status=created.get("status_code") or status_code,
            plant_name=created.get("plant_name") or plant_name,
        )
    )
    return 0


def _get_status(base_url, status_action, status_code, note) -> int:
    if status_action == "ls":
        plant_name = quote(plant_name, safe="")
        payload = request_ctrl("GET", f"{base_url}/plants/{plant_name}/status")
        print(json.dumps(payload, indent=2))
        return 0
    if status_action == "set":
        plant_name = quote(plant_name, safe="")
        payload = {"status_code": status_code}
        if note:
            payload["note"] = note
        created = request_ctrl(
            "POST", f"{base_url}/plants/{plant_name}/status", payload
        )
        print(
            "Added status {status} to {plant_name}".format(
                status=created.get("status_code") or status_code,
                plant_name=created.get("plant_name") or plant_name,
            )
        )
        return 0


def plants_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        return _list_plants(base)
    if action == "sensors":
        return _list_plant_sensor_mapping(base)
    if action == "add":
        return _add_plant(
            base, args.plant_name, args.species_name, args.tag, args.metadata
        )
    if action == "update":
        return _update_plant(
            base, args.species_id, args.tag, args.metadata, args.plant_id
        )
    if action == "rm":
        return _delete_plant(base, args.plant_name)
    if action == "health":
        return _get_health(base, args.plant_name)
    if action == "assign":
        return _assign_plant(base, args.plant_name, args.device, args.sensor)
    if action == "set-status":
        return _set_status(base, args.plant_name, args.status_code, args.note)
    if action == "status":
        return _get_status(base, args.status_action, args.status_code, args.note)
    return 0
