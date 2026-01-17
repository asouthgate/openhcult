"""Entrypoint for running the hcultctrl API."""

import logging

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
    _setup_logging()
    uvicorn.run(
        "hcultctrl.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_config=None,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
