"""Command-line helpers for hcult device management."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse, unquote, quote
import sys

from hcultutils import infer_events, plot_timeseries


class HcultArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _add_base_args(parser):
    parser.add_argument(
        "--start-utc",
        default=None,
        help="Start time in UTC (ISO 8601, e.g. 2026-01-16T12:00:00Z)",
    )
    parser.add_argument(
        "--end-utc",
        default=None,
        help="End time in UTC (ISO 8601, e.g. 2026-01-16T13:00:00Z)",
    )
    parser.add_argument(
        "--ctrl-url",
        default=None,
        help="Query data from hcultctrl instead of SQLite (e.g. http://127.0.0.1:8000)",
    )

def _add_plotter_args(parser):
    parser.add_argument(
        "--sensor",
        default=None,
        help="Filter to a single sensor name (e.g. sensor1)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Filter to a device name or BLE address",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit number of rows when querying hcultctrl",
    )
    parser.add_argument(
        "--diff-lag",
        type=int,
        default=1,
        help="Lag (in samples) for d(t) = s(t) - s(t-h)",
    )
    parser.add_argument(
        "--mad-window",
        type=int,
        default=50,
        help="Window size (in samples) for rolling MAD",
    )
    parser.add_argument(
        "--mad-scale",
        type=float,
        default=1.4826,
        help="Scale factor for MAD -> sigma",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.1,
        help="EWMA alpha for baseline (0 < alpha <= 1)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write PNG to this path instead of showing a window",
    )

def _add_infer_args(parser):
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Lookback window in hours",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit number of rows when querying hcultctrl",
    )
    parser.add_argument(
        "--diff-lag",
        type=int,
        default=1,
        help="Lag (in samples) for d(t) = s(t) - s(t-h)",
    )
    parser.add_argument(
        "--mad-window",
        type=int,
        default=50,
        help="Window size (in samples) for rolling MAD",
    )
    parser.add_argument(
        "--mad-scale",
        type=float,
        default=1.4826,
        help="Scale factor for MAD -> sigma",
    )
    parser.add_argument(
        "--ewma-alpha",
        type=float,
        default=0.1,
        help="EWMA alpha for baseline (0 < alpha <= 1)",
    )
    parser.add_argument(
        "--z-pvalue",
        type=float,
        default=0.000001,
        help="Two-sided p-value threshold for z(t) triggers",
    )
    parser.add_argument(
        "--resid-threshold",
        type=float,
        default=15.0,
        help="Absolute residual threshold for hysteresis test",
    )
    parser.add_argument(
        "--hyst-window",
        type=int,
        default=20,
        help="Window size (in samples) around trigger for hysteresis",
    )
    parser.add_argument(
        "--hyst-k",
        type=int,
        default=3,
        help="Required count of |r(t)| > threshold within window",
    )
    parser.add_argument(
        "--merge-distance-sec",
        type=int,
        default=240,
        help="Minimum seconds between stored events",
    )


def _add_metadata_arg(parser):
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON metadata payload",
    )


def _request_ctrl(method: str, url: str, payload: dict | None = None):
    import json as _json
    import urllib.error as _error
    import urllib.request as _request

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = _json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = _request.Request(url, data=data, method=method, headers=headers)
    try:
        with _request.urlopen(req, timeout=10) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except _error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            detail = _json.loads(body).get("detail")
        except ValueError:
            detail = None
        message = detail or body or exc.reason
        raise SystemExit(f"hcultutils: {exc.code} {message}") from exc


def _species_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        payload = _request_ctrl("GET", f"{base}/species")
        for row in payload.get("data", []):
            print(
                f"{row.get('id')}\t{row.get('name')}\t{row.get('common_name') or ''}\t{row.get('metadata') or ''}"
            )
        return 0
    if action == "add":
        payload = {
            "name": args.name,
            "common_name": args.common_name,
            "metadata": json.loads(args.metadata) if args.metadata else None,
        }
        created = _request_ctrl("POST", f"{base}/species", payload)
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
        updated = _request_ctrl("PATCH", f"{base}/species/{args.id}", payload)
        print(f"Updated species {updated.get('id')}")
        return 0
    if action == "rm":
        deleted = _request_ctrl("DELETE", f"{base}/species/{args.species_name}")
        print(f"Deleted species {deleted.get('id')}")
        return 0
    return 1


def _plants_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        payload = _request_ctrl("GET", f"{base}/plants")
        for row in payload.get("data", []):
            print(
                f"{row.get('id')}\t{row.get('plant_name')}\t{row.get('species_id') or ''}\t{row.get('species_name') or ''}\t{row.get('tag') or ''}\t{row.get('metadata') or ''}"
            )
        return 0
    if action == "add":
        payload = {
            "plant_name": args.plant_name,
            "species_name": args.species_name,
            "tag": args.tag,
            "metadata": json.loads(args.metadata) if args.metadata else None,
        }
        created = _request_ctrl("POST", f"{base}/plants", payload)
        print(f"Created plant {created.get('id')}")
        return 0
    if action == "update":
        payload = {}
        if args.species_id is not None:
            payload["species_id"] = args.species_id
        if args.tag is not None:
            payload["tag"] = args.tag
        if args.metadata is not None:
            payload["metadata"] = json.loads(args.metadata)
        updated = _request_ctrl("PATCH", f"{base}/plants/{args.id}", payload)
        print(f"Updated plant {updated.get('id')}")
        return 0
    if action == "rm":
        deleted = _request_ctrl("DELETE", f"{base}/plants/{args.plant_name}")
        print(f"Deleted plant {deleted.get('plant_name')}")
        return 0
    return 1


def _devices_via_ctrl(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "ls":
        payload = _request_ctrl("GET", f"{base}/devices")
        for row in payload.get("data", []):
            print(
                f"{row.get('id')}\t{row.get('name') or ''}\t{row.get('tag') or ''}\t{row.get('address') or ''}\t{row.get('first_seen') or ''}\t{row.get('last_seen') or ''}"
            )
        return 0
    if action == "name":
        payload = {"name": args.name}
        address = quote(args.address, safe="")
        updated = _request_ctrl("PATCH", f"{base}/devices/{address}", payload)
        print(f"Updated device {updated.get('address')} name={updated.get('name')}")
        return 0
    return 1

def main() -> int:
    parser = HcultArgumentParser(
        prog="hcultutils",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 devices ls\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 devices name AA:BB:CC:DD:EE:FF \"My Device\"\n"
            "  hcultutils plot_timeseries --sensor sensor1 --device AA:BB:CC:DD:EE:FF --start-utc 2026-01-16T12:00:00Z --end-utc 2026-01-16T13:00:00Z\n"
            "  hcultutils infer_events --hours 6\n"
            "  hcultutils species add pothos --common-name \"Golden Pothos\"\n"
            "  hcultutils plants add kitchen-herb --species_name pothos\n"
        ),
    )
    _add_base_args(parser)
    subparsers = parser.add_subparsers(
        help="subcommand help",
        dest="command",
        parser_class=HcultArgumentParser,
    )
    subparsers.required = True
    plot_timeseries_parser = subparsers.add_parser(
        "plot_timeseries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils plot_timeseries --sensor sensor1 --device AA:BB:CC:DD:EE:FF --start-utc 2026-01-16T12:00:00Z --end-utc 2026-01-16T13:00:00Z\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 plot_timeseries --sensor sensor1 --limit 5000\n"
        ),
    )
    _add_base_args(plot_timeseries_parser)
    _add_plotter_args(plot_timeseries_parser)

    infer_events_parser = subparsers.add_parser(
        "infer_events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils infer_events --hours 6\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 infer_events --hours 12 --z-pvalue 0.0001\n"
        ),
    )
    _add_base_args(infer_events_parser)
    _add_infer_args(infer_events_parser)

    species_parser = subparsers.add_parser(
        "species",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils species ls\n"
            "  hcultutils species add pothos --common-name \"Golden Pothos\"\n"
            "  hcultutils species update 1 --name pothos\n"
            "  hcultutils species rm pothos\n"
        ),
    )
    species_sub = species_parser.add_subparsers(dest='action')
    species_sub.required = True
    species_add = species_sub.add_parser('add')
    species_add.add_argument("name")
    species_add.add_argument("--common-name", default=None)
    _add_metadata_arg(species_add)
    species_sub.add_parser('ls')
    species_update = species_sub.add_parser('update')
    species_update.add_argument("id", type=int)
    species_update.add_argument("--name", default=None)
    species_update.add_argument("--common-name", default=None)
    _add_metadata_arg(species_update)
    species_rm = species_sub.add_parser('rm')
    species_rm.add_argument("species_name", type=str)

    plants_parser = subparsers.add_parser(
        "plants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils plants ls\n"
            "  hcultutils plants add kitchen-herb --species_name pothos\n"
            "  hcultutils plants update 1 --tag windowsill\n"
            "  hcultutils plants rm kitchen-herb\n"
        ),
    )
    plants_sub = plants_parser.add_subparsers(dest='action')
    plants_sub.required = True
    plants_add = plants_sub.add_parser('add')
    plants_add.add_argument("plant_name", type=str)
    plants_add.add_argument("--species_name", type=str)
    plants_add.add_argument("--tag", default=None)
    _add_metadata_arg(plants_add)
    plants_sub.add_parser('ls')
    plants_update = plants_sub.add_parser('update')
    plants_update.add_argument("id", type=str)
    plants_update.add_argument("--species-id", type=int, default=None)
    plants_update.add_argument("--tag", default=None)
    _add_metadata_arg(plants_update)
    plants_rm = plants_sub.add_parser('rm')
    plants_rm.add_argument("plant_name", type=str)

    devices_parser = subparsers.add_parser(
        "devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils devices ls\n"
            "  hcultutils devices name AA:BB:CC:DD:EE:FF \"My Device\"\n"
        ),
    )
    devices_sub = devices_parser.add_subparsers(dest='action')
    devices_sub.required = True
    devices_sub.add_parser('ls')
    devices_name = devices_sub.add_parser('name')
    devices_name.add_argument("address", type=str)
    devices_name.add_argument("name", type=str)

    args = parser.parse_args()
    if args.command == 'plot_timeseries':
        plot_timeseries.main(args)
        return 0
    if args.command == 'infer_events':
        return infer_events.run(args)
    if args.command == 'species':
        return _species_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == 'plants':
        return _plants_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == 'devices':
        return _devices_via_ctrl(args.ctrl_url, args.action, args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
