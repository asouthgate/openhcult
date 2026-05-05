#!/usr/bin/env python3
import argparse
import dataclasses
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError

REPO = Path(__file__).resolve().parent


def _detect_compose_cmd():
    for cmd in [["docker", "compose"], ["docker-compose"]]:
        try:
            subprocess.run([*cmd, "version"], capture_output=True, check=True)
            return cmd
        except Exception:
            continue
    raise SystemExit("Neither 'docker compose' nor 'docker-compose' found")


DEFAULT_AUTH_USER = "admin"


def _color(text, code):
    return f"\033[{code}m{text}\033[0m"


def red(text):
    return _color(text, 31)


def green(text):
    return _color(text, 32)


def gray(text):
    return _color(text, 90)


DEFAULT_TESTS = [
    "tests/test_api.py",
    "tests/test_seed_visualisation.py",
    "tests/test_water_calibration.py",
]


@dataclasses.dataclass
class Ctx:
    db_url: str = ""
    ctrl_url: str = ""
    timeout: int = 30
    compose_cmd: list = dataclasses.field(default_factory=list)


def _compose(ctx, *args):
    subprocess.run([*ctx.compose_cmd, *args], check=True)


def _wait_for_postgres(ctx):
    import psycopg

    print(gray("Waiting for postgres..."))
    for _ in range(ctx.timeout):
        try:
            conn = psycopg.connect(ctx.db_url, autocommit=True)
            conn.close()
            print(green("Postgres is ready"))
            return
        except Exception:
            time.sleep(1)
    print(red("Postgres did not become ready in time"))
    raise SystemExit(1)


def _wait_for_ctrl(ctx):
    print(gray(f"Waiting for ctrl at {ctx.ctrl_url}..."))
    for _ in range(ctx.timeout):
        try:
            urlopen(f"{ctx.ctrl_url}/status", timeout=5)
            print(green("Ctrl is ready"))
            return
        except (URLError, ConnectionResetError, OSError):
            time.sleep(1)
    print(red(f"Ctrl not ready at {ctx.ctrl_url} after {ctx.timeout}s"))
    raise SystemExit(1)


def _ctrl_running(ctx):
    try:
        urlopen(f"{ctx.ctrl_url}/status", timeout=3)
        return True
    except (URLError, ConnectionResetError, OSError):
        return False


def _reset_db(ctx):
    import psycopg
    from psycopg import sql
    from hcultdb import setup

    parsed = urlparse(ctx.db_url)
    postgres_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port or 5432}/postgres"
    db_name = parsed.path.lstrip("/")
    print(gray("Dropping and recreating database..."))
    conn = psycopg.connect(postgres_url, autocommit=True)
    conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    conn.close()
    setup.setup_db(ctx.db_url)
    print(green("Database reset complete"))


def _setup_auth(ctx, password):
    try:
        with urlopen(f"{ctx.ctrl_url}/auth/status", timeout=5) as resp:
            status = json.loads(resp.read().decode())
        if status.get("configured"):
            login_data = json.dumps(
                {"username": DEFAULT_AUTH_USER, "password": password}
            ).encode()
            login_req = Request(
                f"{ctx.ctrl_url}/auth/login",
                data=login_data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(login_req, timeout=5):
                pass
            print(green(f"Auth: {DEFAULT_AUTH_USER} / {password}"))
            return
    except URLError:
        pass
    data = json.dumps({"username": DEFAULT_AUTH_USER, "password": password}).encode()
    req = Request(
        f"{ctx.ctrl_url}/auth/setup",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(req, timeout=10):
        pass
    print(green(f"Auth configured: {DEFAULT_AUTH_USER} / {password}"))


def _run_tests(ctx, password, test_paths):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "-q", "tests/test_unauthenticated.py"]
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    _setup_auth(ctx, password)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "-q", *test_paths],
        env={
            **os.environ,
            "OPENHCULT_SMOKE_PASSWORD": password,
            "HCULT_CALIBRATION_CSV": str(REPO / "calib" / "calibration.csv"),
        },
    )
    if result.returncode == 0:
        print(green(f"All tests passed. Frontend: {ctx.ctrl_url}/frontend/app/"))
    raise SystemExit(result.returncode)


_proc_ctrl = None
_proc_frontend = None


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


def _shutdown(ctx):
    _stop_children()
    _compose(ctx, "down", "--remove-orphans")
    print(gray("Stopped all services"))


def cmd_start(ctx, args):
    password = args.password
    parsed_ctrl = urlparse(ctx.ctrl_url)
    ctrl_host = parsed_ctrl.hostname or "127.0.0.1"
    ctrl_port = parsed_ctrl.port or 8000

    _compose(ctx, "up", "-d", "postgres")
    _wait_for_postgres(ctx)
    print(gray("Ensuring database schema..."))
    from hcultdb import setup

    setup.setup_db(ctx.db_url)

    global _proc_ctrl, _proc_frontend
    env = {
        **os.environ,
        "HCULT_CONFIG_PATH": str(REPO / "docker" / "openhcult.dev.conf"),
        "HCULT_CALIBRATION_CSV": str(REPO / "calib" / "calibration.csv"),
    }
    _proc_ctrl = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "hcultctrl.api:app",
            "--host",
            ctrl_host,
            "--port",
            str(ctrl_port),
            "--reload",
            "--reload-dir",
            str(REPO / "hcultctrl"),
            "--reload-dir",
            str(REPO / "hcultdb" / "src"),
        ],
        env=env,
    )
    _wait_for_ctrl(ctx)
    _setup_auth(ctx, password)

    frontend_dir = REPO / "frontend"
    vite = frontend_dir / "node_modules" / ".bin" / "vite"
    _proc_frontend = subprocess.Popen(
        [str(vite), "dev"] if vite.exists() else ["npx", "vite", "dev"],
        cwd=str(frontend_dir),
    )

    print()
    print(green("All services running:"))
    print(f"  API:      {ctx.ctrl_url}")
    print(f"  Frontend: http://{ctrl_host}:5173")
    print(f"  Login:    {DEFAULT_AUTH_USER} / {password}")
    print()
    print(gray("Press Ctrl+C to stop all services"))

    try:
        _proc_ctrl.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(ctx)


def cmd_test(ctx, args):
    if not _ctrl_running(ctx):
        print(red("Ctrl is not running. Start it with: python3 dev.py start"))
        raise SystemExit(1)
    password = os.environ.get("OPENHCULT_SMOKE_PASSWORD", args.password)
    test_paths = args.tests or DEFAULT_TESTS
    _reset_db(ctx)
    _wait_for_ctrl(ctx)
    _run_tests(ctx, password, test_paths)


def cmd_ci(ctx, args):
    password = os.environ.get("OPENHCULT_SMOKE_PASSWORD", secrets.token_urlsafe(12))
    test_paths = args.tests or DEFAULT_TESTS
    print(gray("Building and starting docker stack..."))
    _compose(ctx, "down", "-v", "--remove-orphans")
    _compose(ctx, "up", "--build", "-d")
    try:
        _wait_for_postgres(ctx)
        from hcultdb import setup

        setup.setup_db(ctx.db_url)
        _wait_for_ctrl(ctx)
        _run_tests(ctx, password, test_paths)
    except Exception:
        subprocess.run(
            [*ctx.compose_cmd, "logs", "--no-color", "hcultctrl"], check=False
        )
        raise
    finally:
        _compose(ctx, "down", "-v", "--remove-orphans")


def main():
    parser = argparse.ArgumentParser(
        prog="dev.py",
        description="Local development orchestrator for openhcult",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "OPENHCULT_DSN", "postgresql://hcult:hcult@localhost:5432/hcult"
        ),
        help="Database URL (default: OPENHCULT_DSN env or localhost)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OPENHCULT_BASE_URL", "http://127.0.0.1:8000"),
        help="Ctrl base URL (default: OPENHCULT_BASE_URL env or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("OPENHCULT_TIMEOUT", "30")),
        help="Wait timeout in seconds (default: 30)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start_p = sub.add_parser(
        "start", help="Start postgres + ctrl + frontend. Ctrl+C to stop"
    )
    start_p.add_argument(
        "--password", default="dev", help="Dev auth password (default: dev)"
    )

    test_p = sub.add_parser("test", help="Run smoke tests (resets DB, auto-auth)")
    test_p.add_argument(
        "--password", default="dev", help="Auth password (default: dev)"
    )
    test_p.add_argument(
        "tests", nargs="*", help="Test paths (default: all smoke tests)"
    )

    ci_p = sub.add_parser("ci", help="Run full CI: build docker, test, teardown")
    ci_p.add_argument("tests", nargs="*", help="Test paths (default: all smoke tests)")

    args = parser.parse_args()

    ctx = Ctx(
        db_url=args.dsn,
        ctrl_url=args.url.rstrip("/"),
        timeout=args.timeout,
        compose_cmd=(
            shlex.split(os.environ.get("OPENHCULT_COMPOSE_CMD", ""))
            if os.environ.get("OPENHCULT_COMPOSE_CMD")
            else _detect_compose_cmd()
        ),
    )

    {"start": cmd_start, "test": cmd_test, "ci": cmd_ci}[args.command](ctx, args)


if __name__ == "__main__":
    main()
