#!/usr/bin/env python3
import argparse as ap
import json
import os
import secrets
import subprocess
import time
from urllib import request, error

CTRL_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COMPOSE_CMD = os.environ.get("OPENHCULT_COMPOSE_CMD", "docker-compose")
TIMEOUT_S = int(os.environ.get("OPENHCULT_SMOKE_TIMEOUT_S", "90"))
DB_URL = os.environ.get(
    "OPENHCULT_SMOKE_DB_URL",
    "postgresql://hcult:hcult@postgres:5432/hcult",
)


def _run(cmd, check=True, extra_env=None):
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(cmd, shell=True, check=check, env=env)


def _wait_for_ctrl():
    start = time.time()
    while True:
        try:
            with request.urlopen(f"{CTRL_URL}/status", timeout=5):
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


def _setup_auth(password: str) -> None:
    data = json.dumps({"username": "admin", "password": password}).encode()
    req = request.Request(
        f"{CTRL_URL}/auth/setup",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with request.urlopen(req, timeout=10) as resp:
        resp.read()


def _dump_logs():
    _run(f"{COMPOSE_CMD} logs --no-color hcultctrl", check=False)


def main() -> int:
    args = ap.ArgumentParser(description="Run smoke tests against hcultctrl in Docker")
    args.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Keep the stack running and stream logs after tests",
    )
    args.add_argument(
        "-p",
        "--password",
        type=str,
        default=secrets.token_urlsafe(12),
        help="Password for the admin user (default: random)",
    )

    args = args.parse_args()
    interactive = args.interactive
    password = args.password

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
        _run("pytest -s -q tests/test_unauthenticated.py")
        _setup_auth(password)
        print(f"\nSmoke test credentials: admin / {password}\n")
        result = _run(
            "pytest -s -q tests/test_api.py tests/test_seed_visualisation.py tests/test_water_calibration.py",
            extra_env={"OPENHCULT_SMOKE_PASSWORD": password},
        ).returncode
        if result != 0:
            _dump_logs()
            return result
        if interactive:
            print(f"\nStack is up. Frontend: {CTRL_URL}/frontend/app/")
            print(f"Login: admin / {password}")
            print("Streaming hcultctrl logs (Ctrl+C to stop)...\n")
            try:
                subprocess.run(
                    f"{COMPOSE_CMD} logs -f --no-color hcultctrl",
                    shell=True,
                )
            except KeyboardInterrupt:
                pass
        return result
    except Exception:
        _dump_logs()
        raise
    finally:
        _run(f"{COMPOSE_CMD} down -v --remove-orphans", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
