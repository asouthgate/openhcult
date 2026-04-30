#!/usr/bin/env python3
import argparse
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_URL = os.environ.get(
    "OPENHCULT_DSN", "postgresql://hcult:hcult@localhost:5432/hcult"
)
CTRL_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COMPOSE_CMD = shlex.split(os.environ.get("OPENHCULT_COMPOSE_CMD", "docker-compose"))
DEV_CONF = os.path.join(REPO_ROOT, "docker", "openhcult.dev.conf")
CALIB_CSV = os.path.join(REPO_ROOT, "calib", "calibration.csv")
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")
VENV_BIN = os.path.join(REPO_ROOT, ".venv", "bin")
TIMEOUT = int(os.environ.get("OPENHCULT_TIMEOUT", "30"))
DEV_PASSWORD = "dev"

DEFAULT_TESTS = [
    "tests/test_api.py",
    "tests/test_seed_visualisation.py",
    "tests/test_water_calibration.py",
]

_proc_ctrl = None
_proc_frontend = None


def _color(text, code):
    return f"\033[{code}m{text}\033[0m"


def red(text):
    return _color(text, 31)


def green(text):
    return _color(text, 32)


def gray(text):
    return _color(text, 90)


def _compose(*args):
    subprocess.run([*COMPOSE_CMD, *args], check=True)


def _stop_children():
    global _proc_ctrl, _proc_frontend
    for p in (_proc_ctrl, _proc_frontend):
        if p is None:
            continue
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _proc_ctrl = None
    _proc_frontend = None


def _shutdown():
    _stop_children()
    _compose("down", "--remove-orphans")
    print(gray("Stopped all services"))


def _wait_for_postgres():
    import psycopg

    print(gray("Waiting for postgres..."))
    for _ in range(TIMEOUT):
        try:
            conn = psycopg.connect(DB_URL, autocommit=True)
            conn.close()
            print(green("Postgres is ready"))
            return
        except Exception:
            time.sleep(1)
    print(red("Postgres did not become ready in time"))
    raise SystemExit(1)


def _wait_for_ctrl():
    print(gray(f"Waiting for ctrl at {CTRL_URL}..."))
    for _ in range(TIMEOUT):
        try:
            urlopen(f"{CTRL_URL}/status", timeout=5)
            print(green("Ctrl is ready"))
            return
        except (URLError, ConnectionResetError, OSError):
            time.sleep(1)
    print(red(f"Ctrl not ready at {CTRL_URL} after {TIMEOUT}s"))
    raise SystemExit(1)


def _ctrl_running():
    try:
        urlopen(f"{CTRL_URL}/status", timeout=3)
        return True
    except (URLError, ConnectionResetError, OSError):
        return False


def _reset_db():
    import psycopg
    from hcultdb import setup

    parsed = urlparse(DB_URL)
    postgres_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port or 5432}/postgres"
    db_name = parsed.path.lstrip("/")
    print(gray("Dropping and recreating database..."))
    conn = psycopg.connect(postgres_url, autocommit=True)
    conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.execute(f"CREATE DATABASE {db_name}")
    conn.close()
    setup.setup_db(DB_URL)
    print(green("Database reset complete"))


def _setup_auth(password):
    try:
        with urlopen(f"{CTRL_URL}/auth/status", timeout=5) as resp:
            status = json.loads(resp.read().decode())
        if status.get("configured"):
            try:
                login_data = json.dumps(
                    {"username": "admin", "password": password}
                ).encode()
                login_req = Request(
                    f"{CTRL_URL}/auth/login",
                    data=login_data,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(login_req, timeout=5):
                    pass
                print(green(f"Auth: admin / {password}"))
                return
            except URLError:
                pass
            creds_path = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "openhcult"
                / "credentials.json"
            )
            if not creds_path.exists():
                from hcultctrl import config

                creds_path = config.get_credentials_path()
            if creds_path.exists():
                creds_path.unlink()
            for _ in range(10):
                try:
                    with urlopen(f"{CTRL_URL}/auth/status", timeout=3) as resp:
                        data = json.loads(resp.read().decode())
                    if not data.get("configured"):
                        break
                except (URLError, ConnectionResetError):
                    time.sleep(1)
    except URLError:
        pass
    data = json.dumps({"username": "admin", "password": password}).encode()
    req = Request(
        f"{CTRL_URL}/auth/setup",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(req, timeout=10):
        pass
    print(green(f"Auth configured: admin / {password}"))


def _run_tests(password, test_paths):
    result = subprocess.run(
        [VENV_PYTHON, "-m", "pytest", "-s", "-q", "tests/test_unauthenticated.py"]
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    _setup_auth(password)
    result = subprocess.run(
        [VENV_PYTHON, "-m", "pytest", "-s", "-q", *test_paths],
        env={
            **os.environ,
            "OPENHCULT_SMOKE_PASSWORD": password,
            "HCULT_CALIBRATION_CSV": CALIB_CSV,
        },
    )
    if result.returncode == 0:
        print(green(f"All tests passed. Frontend: {CTRL_URL}/frontend/app/"))
    raise SystemExit(result.returncode)


def cmd_start(_args):
    _stop_children()

    def _sig_handler(_sig, _frame):
        _shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _compose("up", "-d", "postgres")
    _wait_for_postgres()
    print(gray("Ensuring database schema..."))
    from hcultdb import setup

    setup.setup_db(DB_URL)

    global _proc_ctrl, _proc_frontend

    env = {
        **os.environ,
        "HCULT_CONFIG_PATH": DEV_CONF,
        "HCULT_CALIBRATION_CSV": CALIB_CSV,
    }
    _proc_ctrl = subprocess.Popen(
        [
            os.path.join(VENV_BIN, "uvicorn"),
            "hcultctrl.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
            "--reload-dir",
            os.path.join(REPO_ROOT, "hcultctrl"),
            "--reload-dir",
            os.path.join(REPO_ROOT, "hcultdb", "src"),
        ],
        env=env,
    )
    _wait_for_ctrl()
    _setup_auth(DEV_PASSWORD)

    frontend_dir = os.path.join(REPO_ROOT, "frontend")
    vite = os.path.join(frontend_dir, "node_modules", ".bin", "vite")
    if not os.path.exists(vite):
        vite = "npx"
    _proc_frontend = subprocess.Popen([vite, "run", "dev"], cwd=frontend_dir)

    print()
    print(green("All services running:"))
    print(f"  API:      {CTRL_URL}")
    print(f"  Frontend: http://127.0.0.1:5173")
    print(f"  Login:    admin / {DEV_PASSWORD}")
    print()
    print(gray("Press Ctrl+C to stop all services"))

    try:
        _proc_ctrl.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


def cmd_test(args):
    if not _ctrl_running():
        print(red("Ctrl is not running. Start it with: python3 dev.py start"))
        raise SystemExit(1)
    password = os.environ.get("OPENHCULT_SMOKE_PASSWORD", DEV_PASSWORD)
    test_paths = args.tests or DEFAULT_TESTS
    _reset_db()
    _wait_for_ctrl()
    _run_tests(password, test_paths)


def cmd_ci(args):
    password = os.environ.get("OPENHCULT_SMOKE_PASSWORD", secrets.token_urlsafe(12))
    test_paths = args.tests or DEFAULT_TESTS
    print(gray("Building and starting docker stack..."))
    _compose("down", "-v", "--remove-orphans")
    _compose("up", "--build", "-d")
    try:
        _wait_for_postgres()
        from hcultdb import setup

        setup.setup_db(DB_URL)
        _wait_for_ctrl()
        _run_tests(password, test_paths)
    except Exception:
        subprocess.run([*COMPOSE_CMD, "logs", "--no-color", "hcultctrl"], check=False)
        raise
    finally:
        _compose("down", "-v", "--remove-orphans")


def main():
    parser = argparse.ArgumentParser(
        prog="dev.py",
        description="Local development orchestrator for openhcult",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="Start postgres + ctrl + frontend. Ctrl+C to stop")
    test_p = sub.add_parser("test", help="Run smoke tests (resets DB, auto-auth)")
    test_p.add_argument(
        "tests", nargs="*", help="Test paths (default: all smoke tests)"
    )
    ci_p = sub.add_parser("ci", help="Run full CI: build docker, test, teardown")
    ci_p.add_argument("tests", nargs="*", help="Test paths (default: all smoke tests)")
    args = parser.parse_args()
    {"start": cmd_start, "test": cmd_test, "ci": cmd_ci}[args.command](args)


if __name__ == "__main__":
    main()
