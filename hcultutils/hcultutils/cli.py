"""Command-line helpers for hcult device management."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import sys

from hcultutils import infer_events, plot_timeseries, fetch_data
from hcultutils.plants import plants_via_ctrl
from hcultutils.species import species_via_ctrl
from hcultutils.devices import devices_via_ctrl
from hcultutils.calibration import calibration_main


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
    parser.add_argument(
        "--hours",
        default=None,
        type=int,
        help="Last number of hours to process",
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
        "--plant-name",
        default=None,
        help="Filter to a plant name (requires hcultctrl)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write PNG to this path instead of showing a window",
    )


def _add_infer_args(parser):
    parser.add_argument(
        "--emwa_tau_minutes",
        type=int,
        default=30,
        help="EMWA scale (minutes); controls how smooth the smoothed curve is",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit number of rows when querying hcultctrl",
    )


def _add_metadata_arg(parser):
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON metadata payload",
    )


def _add_fetch_data_command(subparsers):
    fetch_data_parsers = subparsers.add_parser(
        "fetch_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils fetch_data --sensor sensor1 --device AA:BB:CC:DD:EE:FF --start-utc 2026-01-16T12:00:00Z --end-utc 2026-01-16T13:00:00Z\n"
        ),
    )
    _add_base_args(fetch_data_parsers)
    _add_plotter_args(fetch_data_parsers)


def _add_plot_timeseries_command(subparsers):
    plot_timeseries_parser = subparsers.add_parser(
        "plot_timeseries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils plot_timeseries --sensor sensor1 --device AA:BB:CC:DD:EE:FF --start-utc 2026-01-16T12:00:00Z --end-utc 2026-01-16T13:00:00Z\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 plot_timeseries --plant-name kitchen-herb --sensor sensor1\n"
            "  hcultutils --ctrl-url http://127.0.0.1:8000 plot_timeseries --sensor sensor1 --limit 5000\n"
        ),
    )
    _add_base_args(plot_timeseries_parser)
    _add_plotter_args(plot_timeseries_parser)
    _add_infer_args(plot_timeseries_parser)


def _add_infer_events_command(subparsers):
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


def _add_species_command(subparsers):
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
    species_add.add_argument("--common_name", default=None)
    _add_metadata_arg(species_add)
    species_sub.add_parser('ls')
    species_update = species_sub.add_parser('update')
    species_update.add_argument("name")
    species_update.add_argument("--common_name", default=None)
    _add_metadata_arg(species_update)
    species_rm = species_sub.add_parser('rm')
    species_rm.add_argument("species_name", type=str)


def _add_plants_command(subparsers):
    plants_parser = subparsers.add_parser(
        "plants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils plants ls\n"
            "  hcultutils plants add kitchen-herb --species_name pothos\n"
            "  hcultutils plants update 1 --tag windowsill\n"
            "  hcultutils plants rm kitchen-herb\n"
            "  hcultutils plants health kitchen-herb\n"
            "  hcultutils plants assign kitchen-herb AA:BB:CC:DD:EE:FF sensor1\n"
            "  hcultutils plants set-status kitchen-herb DROOPING_LEAVES\n"
            "  hcultutils plants status ls kitchen-herb\n"
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

    plants_assign = plants_sub.add_parser("assign")
    plants_assign.add_argument("plant_name", type=str)
    plants_assign.add_argument("device", type=str)
    plants_assign.add_argument("sensor", type=str)

    plants_status = plants_sub.add_parser("set-status")
    plants_status.add_argument("plant_name", type=str)
    plants_status.add_argument("status_code", type=str)
    plants_status.add_argument("--note", default=None)

    plants_status_group = plants_sub.add_parser("status")
    status_sub = plants_status_group.add_subparsers(dest="status_action")
    status_sub.required = True
    status_ls = status_sub.add_parser("ls")
    status_ls.add_argument("plant_name", type=str)
    status_set = status_sub.add_parser("set")
    status_set.add_argument("plant_name", type=str)
    status_set.add_argument("status_code", type=str)
    status_set.add_argument("--note", default=None)

    plants_health = plants_sub.add_parser("health")
    plants_health.add_argument("--plant_name", type=str)


def _add_devices_command(subparsers):
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

def _add_calibration_command(subparsers):
    devices_parser = subparsers.add_parser(
        "calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hcultutils calibration submit_response_curve\n"
        ),
    )
    cal_sub = devices_parser.add_subparsers(dest='action')
    cal_sub.required = True
    submit_parser = cal_sub.add_parser('submit_response_curve')
    submit_parser.add_argument("--csv", type=str)
    submit_parser.add_argument("--version", type=str)
    submit_parser.add_argument("--created_at", type=str)


def _build_parser() -> HcultArgumentParser:
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

    _add_fetch_data_command(subparsers)
    _add_plot_timeseries_command(subparsers)
    _add_infer_events_command(subparsers)
    _add_species_command(subparsers)
    _add_plants_command(subparsers)
    _add_devices_command(subparsers)
    _add_calibration_command(subparsers)

    return parser


def _apply_hours_args(args):
    if args.hours is None:
        return
    now = datetime.now(timezone.utc)
    hours_ago = now - timedelta(hours=args.hours)
    args.start_utc = hours_ago.isoformat()


def _dispatch_command(args) -> int:
    if args.command == 'plot_timeseries':
        plot_timeseries.main(args)
        return 0
    if args.command == 'fetch_data':
        fetch_data.main(args)
        return 0
    if args.command == 'infer_events':
        return infer_events.run(args)
    if args.command == 'species':
        return species_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == 'plants':
        return plants_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == "plant":
        return plants_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == 'devices':
        return devices_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == 'calibration':
        return calibration_main(args.ctrl_url, args.action, args)
    return 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _apply_hours_args(args)
    status = _dispatch_command(args)
    if status == 1:
        parser.print_help()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
