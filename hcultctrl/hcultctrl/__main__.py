"""Entrypoint for running the hcultctrl API."""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from . import config


def main():
    parser = argparse.ArgumentParser(description="Run the hcultctrl API.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to openhcult.conf (default: XDG config)",
    )
    parser.add_argument(
        "--host",
        "--hostname",
        dest="host",
        default=None,
        help="Host/IP to bind (default: config or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: config or 8000)",
    )
    args = parser.parse_args()

    if args.config:
        config.set_config_path(Path(args.config))
        os.environ["HCULT_CONFIG_PATH"] = args.config
    host = args.host or config.get_ctrl_host()
    port = args.port or config.get_ctrl_port()
    config.setup_logging()
    uvicorn.run(
        "hcultctrl.api:app",
        host=host,
        port=port,
        log_config=None,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
