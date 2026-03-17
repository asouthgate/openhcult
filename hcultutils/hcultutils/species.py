import json

from hcultutils.query import request_ctrl
from hcultutils.formatting import format_table


def _list_species(base_url) -> int:
    payload = request_ctrl("GET", f"{base_url}/species")

    headers = ["Name", "Common Name"]
    rows = [
        [
            d.get("name"),
            d.get("common_name"),
        ]
        for d in payload.get("data", [])
    ]

    # Get the string and print it
    table_output = format_table(headers, rows)
    print(table_output)

    return 0


def _add_species(base_url: str, name, common_name, metadata) -> int:
    payload = {
        "name": name,
        "common_name": common_name,
        "metadata": json.loads(metadata) if metadata else None,
    }
    created = request_ctrl("POST", f"{base_url}/species", payload)
    print(f"Created species {created.get('id')}")
    return 0


def _update_species(base_url: str, name, common_name, metadata) -> int:
    payload = {}
    if name is not None:
        payload["name"] = name
    if common_name is not None:
        payload["common_name"] = common_name
    if metadata is not None:
        payload["metadata"] = json.loads(metadata)
    updated = request_ctrl("PATCH", f"{base_url}/species/{name}", payload)
    print(f"Updated species {updated.get('name')}")
    return 0


def _delete_species(base_url: str, species_name) -> int:
    deleted = request_ctrl("DELETE", f"{base_url}/species/{species_name}")
    print(f"Deleted species {deleted.get('name')}")
    return 0


def species_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        return _list_species(base)
    if action == "add":
        return _add_species(base, args.name, args.common_name, args.metadata)
    if action == "update":
        return _update_species(base, args.name, args.common_name, args.metadata)
    if action == "rm":
        return _delete_species(base, args.name)
