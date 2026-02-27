from urllib.parse import quote

from hcultutils.query import request_ctrl

def devices_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        payload = request_ctrl("GET", f"{base}/devices")
        for row in payload.get("data", []):
            print(
                f"{row.get('id')}\t{row.get('name') or ''}\t{row.get('tag') or ''}\t{row.get('address') or ''}\t{row.get('first_seen') or ''}\t{row.get('last_seen') or ''}"
            )
        return 0
    if action == "name":
        payload = {"name": args.name}
        address = quote(args.address, safe="")
        updated = request_ctrl("PATCH", f"{base}/devices/{address}", payload)
        print(f"Updated device {updated.get('address')} name={updated.get('name')}")
        return 0
    return 1

