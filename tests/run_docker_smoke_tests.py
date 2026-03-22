#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from urllib import request, error

CTRL_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COMPOSE_CMD = os.environ.get("OPENHCULT_COMPOSE_CMD", "docker-compose")
TIMEOUT_S = int(os.environ.get("OPENHCULT_SMOKE_TIMEOUT_S", "90"))
DB_URL = os.environ.get(
    "OPENHCULT_SMOKE_DB_URL",
    "postgresql://hcult:hcult@postgres:5432/hcult",
)


def _run(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check)


def _wait_for_ctrl():
    start = time.time()
    while True:
        try:
            with request.urlopen(f"{CTRL_URL}/", timeout=5):
                return
        except (error.URLError, ConnectionResetError):
            if time.time() - start > TIMEOUT_S:
                raise RuntimeError("hcultctrl did not become ready in time")
            time.sleep(2)


def _wait_for_postgres():
    start = time.time()
    while True:
        cmd = (
            f"{COMPOSE_CMD} exec -T postgres "
            "pg_isready -U hcult -d hcult >/dev/null 2>&1"
        )
        if subprocess.run(cmd, shell=True).returncode == 0:
            return
        if time.time() - start > TIMEOUT_S:
            raise RuntimeError("postgres did not become ready in time")
        time.sleep(2)


def main() -> int:
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    _run(f"{COMPOSE_CMD} down -v --remove-orphans")
    _run(f"{COMPOSE_CMD} up --build -d")
    try:
        _wait_for_postgres()
        _run(
            f"{COMPOSE_CMD} exec -T hcultctrl "
            'python -c "from hcultdb import setup; '
            f"setup.setup_db('{DB_URL}')\""
        )
        _wait_for_ctrl()
        result = _run(
            "pytest -q tests/test_api.py tests/test_seed_visualisation.py"
        ).returncode
        if interactive:
            print(f"\nStack is up. Frontend: {CTRL_URL}/frontend/app/")
            input("Press Enter to tear down.\n")
        return result
    except Exception:
        _run(f"{COMPOSE_CMD} logs --no-color hcultctrl", check=False)
        _run(f"{COMPOSE_CMD} logs --no-color postgres", check=False)
        raise
    finally:
        _run(f"{COMPOSE_CMD} down -v --remove-orphans", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
