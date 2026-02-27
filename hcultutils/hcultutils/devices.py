from urllib.parse import quote

from hcultutils.query import request_ctrl


def _list_devices(base_url) -> int:
    payload = request_ctrl("GET", f"{base_url}/devices")
    for row in payload.get("data", []):
        print(
            f"{row.get('id')}\t{row.get('name') or ''}\t{row.get('tag') or ''}\t{row.get('address') or ''}\t{row.get('first_seen') or ''}\t{row.get('last_seen') or ''}"
        )
    return 1

def _name_device(base_url, name, address) -> int:
    payload = {"name": name}
    address = quote(address, safe="")
    updated = request_ctrl("PATCH", f"{base_url}/devices/{address}", payload)
    print(f"Updated device {updated.get('address')} name={updated.get('name')}")
    return True

def devices_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        return _list_devices(base)
    if action == "name":
        return _name_device(base, args.name, args.address)