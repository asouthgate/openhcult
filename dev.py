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
PID_DIR = os.path.join(REPO_ROOT, ".dev")

DEFAULT_TESTS = [
    "tests/test_api.py",
    "tests/test_seed_visualisation.py",
    "tests/test_water_calibration.py",
]


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


def _pid_file(name):
    return os.path.join(PID_DIR, f"{name}.pid")


def _read_pid(name):
    path = _pid_file(name)
    try:
        return int(Path(path).read_text().strip())
    except (ValueError, OSError, FileNotFoundError):
        return None


def _write_pid(name, pid):
    os.makedirs(PID_DIR, exist_ok=True)
    Path(_pid_file(name)).write_text(str(pid))


def _kill_pid(name):
    pid = _read_pid(name)
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    Path(_pid_file(name)).unlink(missing_ok=True)


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
    from urllib.parse import urlparse

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


def cmd_up(_args):
    _compose("up", "-d", "postgres")
    _wait_for_postgres()
    print(gray("Ensuring database schema..."))
    from hcultdb import setup

    setup.setup_db(DB_URL)
    print(green("Schema ok"))

    os.makedirs(PID_DIR, exist_ok=True)
    _kill_pid("ctrl")
    env = {
        **os.environ,
        "HCULT_CONFIG_PATH": DEV_CONF,
        "HCULT_CALIBRATION_CSV": CALIB_CSV,
    }
    log = open(os.path.join(PID_DIR, "ctrl.log"), "w")
    proc = subprocess.Popen(
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
        stdout=log,
        stderr=log,
    )
    _write_pid("ctrl", proc.pid)
    print(green(f"Ctrl started (pid {proc.pid}, log: .dev/ctrl.log)"))
    _wait_for_ctrl()
    _setup_auth(DEV_PASSWORD)

    _kill_pid("frontend")
    frontend_dir = os.path.join(REPO_ROOT, "frontend")
    vite = os.path.join(frontend_dir, "node_modules", ".bin", "vite")
    if not os.path.exists(vite):
        vite = "npx"
    fe_log = open(os.path.join(PID_DIR, "frontend.log"), "w")
    fe_proc = subprocess.Popen(
        [vite, "run", "dev"], cwd=frontend_dir, stdout=fe_log, stderr=fe_log
    )
    _write_pid("frontend", fe_proc.pid)
    print(green(f"Frontend started (pid {fe_proc.pid}, log: .dev/frontend.log)"))
    print()
    print(green("All services running:"))
    print(f"  API:      {CTRL_URL}")
    print(f"  Frontend: http://127.0.0.1:5173")
    print(f"  Login:    admin / {DEV_PASSWORD}")
    print(f"  Logs:     {PID_DIR}/")
    print()
    print(gray("Stop everything with: python3 dev.py down"))


def cmd_down(_args):
    _kill_pid("ctrl")
    _kill_pid("frontend")
    _compose("down", "--remove-orphans")
    print(green("Stopped all services"))


def cmd_test(args):
    if not _ctrl_running():
        print(red("Ctrl is not running. Start it with: python3 dev.py up"))
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
    sub.add_parser("up", help="Start postgres + ctrl + frontend, setup auth")
    sub.add_parser("down", help="Stop all services")
    test_p = sub.add_parser("test", help="Run smoke tests (resets DB, auto-auth)")
    test_p.add_argument(
        "tests", nargs="*", help="Test paths (default: all smoke tests)"
    )
    ci_p = sub.add_parser("ci", help="Run full CI: build docker, test, teardown")
    ci_p.add_argument("tests", nargs="*", help="Test paths (default: all smoke tests)")
    args = parser.parse_args()
    {"up": cmd_up, "down": cmd_down, "test": cmd_test, "ci": cmd_ci}[args.command](args)


if __name__ == "__main__":
    main()
