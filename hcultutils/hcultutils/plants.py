import json
from urllib.parse import quote

from hcultutils.query import pretty_print, request_ctrl, format_observed_at


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


def _list_plants(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    payload = request_ctrl("GET", f"{base}/plants")
    for row in payload.get("data", []):
        print(
            f"{row.get('id')}\t{row.get('plant_name')}\t{row.get('species_id') or ''}\t{row.get('species_name') or ''}\t{row.get('tag') or ''}\t{row.get('metadata') or ''}"
        )
    return 1


def _add_plant(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    payload = {
        "plant_name": args.plant_name,
        "species_name": args.species_name,
        "tag": args.tag,
        "metadata": json.loads(args.metadata) if args.metadata else None,
    }
    created = request_ctrl("POST", f"{base}/plants", payload)
    print(f"Created plant {created.get('id')}")
    return 0


def _update_plant(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    payload = {}
    if args.species_id is not None:
        payload["species_id"] = args.species_id
    if args.tag is not None:
        payload["tag"] = args.tag
    if args.metadata is not None:
        payload["metadata"] = json.loads(args.metadata)
    updated = request_ctrl("PATCH", f"{base}/plants/{args.id}", payload)
    print(f"Updated plant {updated.get('id')}")
    return 0


def _delete_plant(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    deleted = request_ctrl("DELETE", f"{base}/plants/{args.plant_name}")
    print(f"Deleted plant {deleted.get('plant_name')}")
    return 0


def _get_health(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if args.plant_name is None:
        payload = request_ctrl("GET", f"{base}/plants/health")
        if isinstance(payload, dict) and "data" in payload:
            for row in payload.get("data", []):
                _render_health_payload(row)
        pretty_print(payload)
        print()
        return 0
    plant_name = quote(args.plant_name, safe="")
    payload = request_ctrl("GET", f"{base}/plants/{plant_name}/health")
    _render_health_payload(payload)
    pretty_print(payload)
    print()
    return 0


def _assign_plant(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    plant_name = quote(args.plant_name, safe="")
    payload = {"device": args.device, "sensor": args.sensor}
    assigned = request_ctrl("POST", f"{base}/plants/{plant_name}/assign", payload)
    print(
        "Assigned plant {plant_name} to {device} sensor {sensor}".format(
            plant_name=assigned.get("plant_name") or args.plant_name,
            device=assigned.get("device_name")
            or assigned.get("device_address")
            or args.device,
            sensor=assigned.get("sensor") or args.sensor,
        )
    )
    return 0


def _set_status(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    plant_name = quote(args.plant_name, safe="")
    payload = {"status_code": args.status_code}
    if args.note:
        payload["note"] = args.note
    created = request_ctrl("POST", f"{base}/plants/{plant_name}/status", payload)
    print(
        "Added status {status} to {plant_name}".format(
            status=created.get("status_code") or args.status_code,
            plant_name=created.get("plant_name") or args.plant_name,
        )
    )
    return 0


def _get_status(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if args.status_action == "ls":
        plant_name = quote(args.plant_name, safe="")
        payload = request_ctrl("GET", f"{base}/plants/{plant_name}/status")
        print(json.dumps(payload, indent=2))
        return 0
    if args.status_action == "set":
        plant_name = quote(args.plant_name, safe="")
        payload = {"status_code": args.status_code}
        if args.note:
            payload["note"] = args.note
        created = request_ctrl("POST", f"{base}/plants/{plant_name}/status", payload)
        print(
            "Added status {status} to {plant_name}".format(
                status=created.get("status_code") or args.status_code,
                plant_name=created.get("plant_name") or args.plant_name,
            )
        )
        return 0
    


def plants_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        return _list_plants(ctrl_url, args)
    if action == "add":
        return _add_plant(ctrl_url, args)
    if action == "update":
        return _update_plant(ctrl_url, args)
    if action == "rm":
        return _delete_plant(ctrl_url, args)
    if action == "health":
        return _get_health(ctrl_url, args)
    if action == "assign":
        return _assign_plant(ctrl_url, args)
    if action == "set-status":
        return _set_status(ctrl_url, args)
    if action == "status":
        return _get_status(ctrl_url, args)
    return 1