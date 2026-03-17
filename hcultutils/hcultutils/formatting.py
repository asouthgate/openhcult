from datetime import datetime, timezone
import json


def format_datetime_string(dtstr):
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


def format_observed_at(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return value


def pretty_print(value, indent=0):
    spacer = " " * indent
    if isinstance(value, dict):
        print(f"{spacer}{{")
        items = list(value.items())
        for idx, (key, val) in enumerate(items):
            key_str = json.dumps(str(key))
            print(f"{spacer}  {key_str}: ", end="")
            pretty_print(val, indent + 2)
            if idx < len(items) - 1:
                print(",")
            else:
                print()
        print(f"{spacer}}}", end="")
        return
    if isinstance(value, list):
        print(f"{spacer}[")
        for idx, item in enumerate(value):
            pretty_print(item, indent + 2)
            if idx < len(value) - 1:
                print(",")
            else:
                print()
        print(f"{spacer}]", end="")
        return
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        print(f'"{escaped}"', end="")
        return
    print(json.dumps(value), end="")
