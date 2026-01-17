"""Entrypoint for running the hcultctrl API."""

import argparse
import logging
from pathlib import Path

import uvicorn

from . import config


def _setup_logging():
    handlers = []
    if config.get_log_stdout():
        handlers.append(logging.StreamHandler())
    log_path = config.get_log_path("hcultctrl")
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        except PermissionError:
            handlers.append(logging.StreamHandler())
    if not handlers:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


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
    host = args.host or config.get_ctrl_host()
    port = args.port or config.get_ctrl_port()
    _setup_logging()
    uvicorn.run(
        "hcultctrl.api:app",
        host=host,
        port=port,
        reload=False,
        log_config=None,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
