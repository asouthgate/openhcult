from urllib.parse import quote

from hcultutils.query import request_ctrl


def _list_devices(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    payload = request_ctrl("GET", f"{base}/devices")
    for row in payload.get("data", []):
        print(
            f"{row.get('id')}\t{row.get('name') or ''}\t{row.get('tag') or ''}\t{row.get('address') or ''}\t{row.get('first_seen') or ''}\t{row.get('last_seen') or ''}"
        )
    return 1

def _name_device(ctrl_url: str, args) -> int:
    base = ctrl_url.rstrip("/")
    payload = {"name": args.name}
    address = quote(args.address, safe="")
    updated = request_ctrl("PATCH", f"{base}/devices/{address}", payload)
    print(f"Updated device {updated.get('address')} name={updated.get('name')}")
    return True

def _devices_via_ctrl(ctrl_url: str, action: str, args) -> int:
    if action == "ls":
        return _list_devices(ctrl_url, args)
    if action == "name":
        return _name_device(ctrl_url, args)