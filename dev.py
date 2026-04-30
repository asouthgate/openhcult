#!/usr/bin/env python3
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_URL = os.environ.get(
    "OPENHCULT_DSN", "postgresql://hcult:hcult@localhost:5432/hcult"
)

CTRL_URL = os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COMPOSE_CMD = os.environ.get("OPENHCULT_COMPOSE_CMD", "docker-compose")
DEV_CONF = os.path.join(REPO_ROOT, "docker", "openhcult.dev.conf")
CALIB_CSV = os.path.join(REPO_ROOT, "calib", "calibration.csv")
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")
TIMEOUT = int(os.environ.get("OPENHCULT_TIMEOUT", "30"))

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
    subprocess.run([COMPOSE_CMD, *args], check=True)


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


def _parse_dsn(dsn):
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    postgres_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port or 5432}/postgres"
    db_name = parsed.path.lstrip("/")
    return postgres_url, db_name


def _reset_db():
    import psycopg
    from hcultdb import setup

    postgres_url, db_name = _parse_dsn(DB_URL)
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
            print(gray("Auth already configured, skipping setup"))
            return
    except URLError:
        pass
    data = json.dumps({"username": "admin", "password": password}).encode()
    req = Request(
        f"{CTRL_URL}/auth/setup",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(req, timeout=10) as resp:
        resp.read()


def _run_tests(password, test_paths):
    result = subprocess.run(
        [VENV_PYTHON, "-m", "pytest", "-s", "-q", "tests/test_unauthenticated.py"]
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    _setup_auth(password)
    print(f"Credentials: admin / {password}")

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
    print(gray("Starting postgres..."))
    _compose("up", "-d", "postgres")
    _wait_for_postgres()
    print(gray("Ensuring database schema..."))
    from hcultdb import setup

    setup.setup_db(DB_URL)
    print(green("Schema ok"))


def cmd_init_db(_args):
    from hcultdb import setup

    print(gray("Reinitializing database schema..."))
    setup.setup_db(DB_URL)
    print(green("Done"))


def cmd_reset_db(_args):
    _reset_db()


def cmd_ctrl(_args):
    env = {
        **os.environ,
        "HCULT_CONFIG_PATH": DEV_CONF,
        "HCULT_CALIBRATION_CSV": CALIB_CSV,
    }
    uvicorn = os.path.join(REPO_ROOT, ".venv", "bin", "uvicorn")
    print(green("Starting ctrl (with --reload)..."))
    os.execle(
        uvicorn,
        "uvicorn",
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
        env,
    )


def cmd_frontend(_args):
    print(green("Starting frontend dev server..."))
    frontend_dir = os.path.join(REPO_ROOT, "frontend")
    vite = os.path.join(frontend_dir, "node_modules", ".bin", "vite")
    if not os.path.exists(vite):
        vite = "npx"
    os.chdir(frontend_dir)
    os.execvp(vite, [vite, "run", "dev"])


def cmd_down(_args):
    print(gray("Stopping services..."))
    _compose("down", "--remove-orphans")


def cmd_status(_args):
    import psycopg

    print(gray("Postgres:"))
    try:
        conn = psycopg.connect(DB_URL, autocommit=True)
        conn.close()
        print(green("  Running"))
    except Exception:
        print(red("  Stopped"))

    print(gray("Ctrl:"))
    if _ctrl_running():
        print(green(f"  Running ({CTRL_URL}/status)"))
    else:
        print(red("  Stopped"))

    print(gray("Frontend:"))
    try:
        urlopen("http://127.0.0.1:5173/", timeout=3)
        print(green("  Running (http://127.0.0.1:5173)"))
    except Exception:
        print(red("  Stopped"))


def cmd_test(args):
    if not _ctrl_running():
        print(red("Ctrl is not running. Start it with: ./dev.py ctrl"))
        raise SystemExit(1)

    password = os.environ.get("OPENHCULT_SMOKE_PASSWORD", secrets.token_urlsafe(12))
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
        subprocess.run([COMPOSE_CMD, "logs", "--no-color", "hcultctrl"], check=False)
        raise
    finally:
        _compose("down", "-v", "--remove-orphans")


def main():
    parser = argparse.ArgumentParser(
        prog="dev.py",
        description="Local development orchestrator for openhcult",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("up", help="Start postgres (persistent), init schema")
    sub.add_parser("down", help="Stop all docker services")
    sub.add_parser("ctrl", help="Start ctrl with uvicorn --reload")
    sub.add_parser("frontend", help="Start frontend dev server")
    sub.add_parser("status", help="Show status of services")
    sub.add_parser("init-db", help="(Re)initialize database schema (idempotent)")
    sub.add_parser("reset-db", help="Drop and recreate the hcult database")

    test_p = sub.add_parser(
        "test", help="Run smoke tests against native ctrl (resets DB)"
    )
    test_p.add_argument(
        "tests", nargs="*", help="Test paths (default: all smoke tests)"
    )

    ci_p = sub.add_parser("ci", help="Run full CI: build docker, test, teardown")
    ci_p.add_argument("tests", nargs="*", help="Test paths (default: all smoke tests)")

    args = parser.parse_args()
    {
        "up": cmd_up,
        "down": cmd_down,
        "ctrl": cmd_ctrl,
        "frontend": cmd_frontend,
        "init-db": cmd_init_db,
        "reset-db": cmd_reset_db,
        "status": cmd_status,
        "test": cmd_test,
        "ci": cmd_ci,
    }[args.command](args)


if __name__ == "__main__":
    main()
