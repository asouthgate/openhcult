"""Command-line helpers for hcult device management."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse, unquote
import sys

from hcultutils import infer_events, plot_timeseries


DEFAULT_CONFIG_NAME = "openhcult.conf"


def _default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "openhcult" / DEFAULT_CONFIG_NAME
    return Path.home() / ".config" / "openhcult" / DEFAULT_CONFIG_NAME


def _load_config():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    parser = ConfigParser()
    parser.read(config_path)
    return parser, config_path, repo_root


def _get_db_url() -> str:
    parser, config_path, _ = _load_config()
    if "database" not in parser or "url" not in parser["database"]:
        raise ValueError(f"Missing database.url in {config_path}")
    url = parser["database"]["url"].strip()
    if not url:
        raise ValueError(f"Empty database.url in {config_path}")
    return url


def _connect_db(db_url: str):
    parsed = urlparse(db_url)
    if parsed.scheme in ("", "file", "sqlite"):
        if parsed.scheme in ("file", "sqlite"):
            db_path = Path(unquote(parsed.path))
        else:
            db_path = Path(db_url)
        return sqlite3.connect(str(db_path))
    if parsed.scheme.startswith("postgres"):
        import psycopg

        return psycopg.connect(db_url)
    raise ValueError(f"Unsupported database URL: {db_url}")


def _placeholder(conn) -> str:
    module = conn.__class__.__module__
    return "%s" if "psycopg" in module or "psycopg2" in module else "?"


def _parse_device_id(value: str) -> tuple[str, str]:
    try:
        device_id = int(value)
    except ValueError:
        return "address", value
    return "id", str(device_id)


def _update_device_tag(device_key: str, device_value: str, new_tag: str) -> int:
    db_url = _get_db_url()
    conn = _connect_db(db_url)
    try:
        cursor = conn.cursor()
        placeholder = _placeholder(conn)
        if device_key == "id":
            cursor.execute(
                f"SELECT id, address, tag FROM devices WHERE id = {placeholder}",
                (device_value,),
            )
        else:
            cursor.execute(
                f"SELECT id, address, tag FROM devices WHERE address = {placeholder}",
                (device_value,),
            )
        row = cursor.fetchone()
        if not row:
            print(f"No device found for {device_key}={device_value}", file=sys.stderr)
            return 1
        device_id, address, old_tag = row
        cursor.execute(
            f"UPDATE devices SET tag = {placeholder}, last_seen = CURRENT_TIMESTAMP WHERE id = {placeholder}",
            (new_tag, device_id),
        )
        conn.commit()
        print(f"Updated device {device_id} ({address}): {old_tag} -> {new_tag}")
        return 0
    finally:
        conn.close()


def _list_devices() -> int:
    db_url = _get_db_url()
    conn = _connect_db(db_url)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, tag, address, last_seen FROM devices ORDER BY id ASC"
        )
        rows = cursor.fetchall()
        if not rows:
            print("No devices found.")
            return 0
        print("id\tname\ttag\taddress\tlast_seen")
        for device_id, name, tag, address, last_seen in rows:
            safe_name = "" if name is None else name
            safe_tag = "" if tag is None else tag
            safe_address = "" if address is None else address
            safe_last_seen = "" if last_seen is None else last_seen
            print(
                f"{device_id}\t{safe_name}\t{safe_tag}\t{safe_address}\t{safe_last_seen}"
            )
        return 0
    finally:
        conn.close()


def _build_parser() -> argparse.ArgumentParser:
    return parser


def _print_usage_examples(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print("\nExamples:")
    print("  hcult sensor ls")
    print("  hcult sensor tag 3 Kitchen_Sink")
    print("  hcult sensor tag AA:BB:CC:DD:EE:FF Garden")
    print("  hcult plot_timeseries --ctrl-url http://127.0.0.1:8000")
    print("  hcult infer_events --hours 1")

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
        "--config",
        default=str(_default_config_path()),
        help="Path to openhcult.conf (default: XDG config)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override database URL (otherwise read from config)",
    )
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

def main() -> int:
    parser = argparse.ArgumentParser(prog='hcultutils')
    subparsers = parser.add_subparsers(help='subcommand help', dest='command')
    subparsers.required = True
    plot_timeseries_parser = subparsers.add_parser('plot_timeseries')
    _add_base_args(plot_timeseries_parser)
    _add_plotter_args(plot_timeseries_parser)

    infer_events_parser = subparsers.add_parser('infer_events')
    _add_base_args(infer_events_parser)
    _add_infer_args(infer_events_parser)

    args = parser.parse_args()
    if args.command == 'plot_timeseries':
        plot_timeseries.main(args)
        return 0
    if args.command == 'infer_events':
        return infer_events.run(args)
    parser.print_help()
    return 1



if __name__ == "__main__":
    raise SystemExit(main())
