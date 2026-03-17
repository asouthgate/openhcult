from urllib.parse import quote

from hcultutils.query import request_ctrl


def _format_datetime_string(dtstr):
    return dtstr.split(".")[0].replace("T", " ")


def format_table(headers: list[str], data: list[list[str]]) -> str:
    """Returns a formatted table as a single printable string."""
    if not data and not headers:
        return ""

    # 1. Normalize all data to strings and handle None values
    clean_data = [[str(item or "") for item in row] for row in data]
    clean_headers = [str(h or "") for h in headers]

    # 2. Calculate max width for each column
    # Combine headers and data to find the absolute max width per column
    all_rows = [clean_headers] + clean_data
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(clean_headers))]

    # 3. Build the format string (e.g., "{:<10}  |  {:<20}")
    row_format = "  |  ".join([f"{{:<{w}}}" for w in widths])

    # 4. Create the separator line (e.g., "----------+-----------")
    separator = "-+-".join(["-" * w for w in widths])

    # 5. Construct the final string
    lines = [row_format.format(*clean_headers), separator]
    for row in clean_data:
        lines.append(row_format.format(*row))

    return "\n".join(lines)


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
            _format_datetime_string(d.get("first_seen")),
            _format_datetime_string(d.get("last_seen")),
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
