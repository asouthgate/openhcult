from urllib.parse import quote

from hcultutils.query import request_ctrl
from hcultutils.formatting import format_datetime_string, format_table


def _list_devices(base_url: str) -> int:
    payload = request_ctrl("GET", f"{base_url}/devices")
    devices = payload.get("data", [])

    if not devices:
        print("No devices found.")
        return 0

    headers = ["Address", "Name", "Tag", "First Seen", "Last Seen"]
    rows = [
        [
            d.get("address"),
            d.get("name"),
            d.get("tag"),
            format_datetime_string(d.get("first_seen")),
            format_datetime_string(d.get("last_seen")),
        ]
        for d in devices
    ]

    # Get the string and print it
    table_output = format_table(headers, rows)
    print(table_output)

    return 0


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
