"""Command-line helpers for hcult device management."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import os
import sys

from hcultutils import infer_events, plot_timeseries, fetch_data
from hcultutils.plants import plants_via_ctrl
from hcultutils.species import species_via_ctrl
from hcultutils.devices import devices_via_ctrl
from hcultutils.calibration import calibration_main
from hcultutils.observations import observations_main
from hcultutils.inference_train import inference_train_main


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
        default=os.environ.get("HCULT_CTRL_URL", None),
        help="URL for the CTRL node",
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
        "--merge_distance_seconds",
        type=int,
        default=10,
        help="Events this close together are considered duplicates",
    )
    parser.add_argument(
        "--emwa_tau_minutes",
        type=int,
        default=30,
        help="EMWA scale (minutes); controls how smooth the smoothed curve is",
    )
    parser.add_argument(
        "--trigger_threshold",
        type=float,
        default=-0.75,
        help="Thresholds for triggering candidate events at start of dis-equilibrium state",
    )
    parser.add_argument(
        "--release_threshold",
        type=float,
        default=-0.7,
        help="Threshold for triggering release into equilibrium state",
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
    )
    _add_base_args(fetch_data_parsers)
    _add_plotter_args(fetch_data_parsers)


def _add_plot_timeseries_command(subparsers):
    plot_timeseries_parser = subparsers.add_parser(
        "plot_timeseries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_base_args(plot_timeseries_parser)
    _add_plotter_args(plot_timeseries_parser)
    _add_infer_args(plot_timeseries_parser)


def _add_infer_events_command(subparsers):
    infer_events_parser = subparsers.add_parser(
        "infer_events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_base_args(infer_events_parser)
    _add_infer_args(infer_events_parser)


def _add_species_command(subparsers):
    species_parser = subparsers.add_parser(
        "species",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    species_sub = species_parser.add_subparsers(dest="action")
    species_sub.required = True
    species_add = species_sub.add_parser("add")
    species_add.add_argument("name")
    species_add.add_argument("--common_name", default=None)
    _add_metadata_arg(species_add)
    species_sub.add_parser("ls")
    species_update = species_sub.add_parser("update")
    species_update.add_argument("name")
    species_update.add_argument("--common_name", default=None)
    _add_metadata_arg(species_update)
    species_rm = species_sub.add_parser("rm")
    species_rm.add_argument("species_name", type=str)


def _add_plants_command(subparsers):
    plants_parser = subparsers.add_parser(
        "plants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plants_sub = plants_parser.add_subparsers(dest="action")
    plants_sub.required = True
    plants_add = plants_sub.add_parser("add")
    plants_add.add_argument("plant_name", type=str)
    plants_add.add_argument("--species_name", type=str)
    plants_add.add_argument("--tag", default=None)
    _add_metadata_arg(plants_add)
    plants_sub.add_parser("ls")
    plants_sub.add_parser("sensors")
    plants_update = plants_sub.add_parser("update")
    plants_update.add_argument("id", type=str)
    plants_update.add_argument("--species-id", type=int, default=None)
    plants_update.add_argument("--tag", default=None)
    _add_metadata_arg(plants_update)
    plants_rm = plants_sub.add_parser("rm")
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
    )
    devices_sub = devices_parser.add_subparsers(dest="action")
    devices_sub.required = True
    devices_sub.add_parser("ls")
    devices_name = devices_sub.add_parser("name")
    devices_name.add_argument("address", type=str)
    devices_name.add_argument("name", type=str)


def _add_calibration_command(subparsers):
    devices_parser = subparsers.add_parser(
        "calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cal_sub = devices_parser.add_subparsers(dest="action")
    cal_sub.required = True
    submit_parser = cal_sub.add_parser("submit_response_curve")
    submit_parser.add_argument("--csv", type=str)
    submit_parser.add_argument("--version", type=str)
    submit_parser.add_argument("--created_at", type=str)


def _add_observations_command(subparsers):
    obs_parser = subparsers.add_parser(
        "observations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Manage plant observations",
    )
    obs_sub = obs_parser.add_subparsers(dest="action")
    obs_sub.required = True

    record_parser = obs_sub.add_parser("record", help="Record a new observation")

    record_parser.add_argument(
        "--plant-name", type=str, default=None, help="Name of the plant (optional)"
    )
    record_parser.add_argument(
        "--note", required=True, help="The observation text (must be non-empty)"
    )
    record_parser.add_argument(
        "--observed-at",
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        help="ISO 8601 timestamp. If omitted, server uses current time.",
    )


def _add_inference_train_command(subparsers):
    parser = subparsers.add_parser("inference_train")
    _add_base_args(parser)
    parser.add_argument("--plant-name", default=None, help="Filter to a specific plant")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument(
        "--grid-n",
        type=int,
        default=8,
        help="Grid points per parameter axis (n³ total)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Cycle through detected events with lognormal fits",
    )


def _build_parser() -> HcultArgumentParser:
    parser = HcultArgumentParser(
        prog="hcultutils",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    _add_observations_command(subparsers)
    _add_inference_train_command(subparsers)
    return parser


def _apply_hours_args(args):
    if args.hours is None:
        return
    now = datetime.now(timezone.utc)
    hours_ago = now - timedelta(hours=args.hours)
    args.start_utc = hours_ago.isoformat()


def _dispatch_command(args) -> int:
    if not args.ctrl_url:
        raise ValueError("Must specify --ctrl-url or define HCULT_CTRL_URL")
    if args.command == "plot_timeseries":
        plot_timeseries.main(args)
        return 0
    if args.command == "fetch_data":
        fetch_data.main(args)
        return 0
    if args.command == "infer_events":
        return infer_events.run(args)
    if args.command == "species":
        return species_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == "plants":
        return plants_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == "plant":
        return plants_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == "devices":
        return devices_via_ctrl(args.ctrl_url, args.action, args)
    if args.command == "calibration":
        return calibration_main(args.ctrl_url, args.action, args)
    if args.command == "observations":
        return observations_main(args.ctrl_url, args.action, args)
    if args.command == "inference_train":
        return inference_train_main(args.ctrl_url, args)
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
