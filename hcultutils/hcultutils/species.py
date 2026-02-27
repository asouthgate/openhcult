
import json

from hcultutils.query import request_ctrl


def species_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        payload = request_ctrl("GET", f"{base}/species")
        for row in payload.get("data", []):
            print(row)
        return 0
    if action == "add":
        payload = {
            "name": args.name,
            "common_name": args.common_name,
            "metadata": json.loads(args.metadata) if args.metadata else None,
        }
        created = request_ctrl("POST", f"{base}/species", payload)
        print(f"Created species {created.get('id')}")
        return 0
    if action == "update":
        payload = {}
        if args.name is not None:
            payload["name"] = args.name
        if args.common_name is not None:
            payload["common_name"] = args.common_name
        if args.metadata is not None:
            payload["metadata"] = json.loads(args.metadata)
        updated = request_ctrl("PATCH", f"{base}/species/{args.name}", payload)
        print(f"Updated species {updated.get('name')}")
        return 0
    if action == "rm":
        deleted = request_ctrl("DELETE", f"{base}/species/{args.species_name}")
        print(f"Deleted species {deleted.get('id')}")
        return 0
    return 1
