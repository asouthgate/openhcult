import argparse
import asyncio
import logging
from pathlib import Path

from . import ble
from . import config
from hcultdb import queries

def _setup_logging():
    handlers = []
    if config.get_log_stdout():
        handlers.append(logging.StreamHandler())
    log_path = config.get_log_path("hcultmon")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

async def _main():
    db_url = config.get_db_url()
    db_con = queries.connect(db_url)
    await ble.run_monitor(db_con)

def main():
    parser = argparse.ArgumentParser(description="Run the hcultmon BLE monitor.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to openhcult.conf (default: XDG config)",
    )
    args = parser.parse_args()

    if args.config:
        config.set_config_path(Path(args.config))
    _setup_logging()
    asyncio.run(_main())

if __name__ == "__main__":
    main()
