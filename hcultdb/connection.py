"""Shared DB connection helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote

import sqlite3


def is_postgres(conn) -> bool:
    module = conn.__class__.__module__
    return "psycopg" in module or "psycopg2" in module


def placeholder(conn) -> str:
    return "%s" if is_postgres(conn) else "?"


def connect(db_url: str):
    parsed = urlparse(db_url)
    if parsed.scheme in ("", "file", "sqlite"):
        if parsed.scheme in ("file", "sqlite"):
            db_path = Path(unquote(parsed.path))
        else:
            db_path = Path(db_url)
        return sqlite3.connect(str(db_path), check_same_thread=False)
    if parsed.scheme.startswith("postgres"):
        import psycopg

        return psycopg.connect(db_url)
    raise ValueError(f"Unsupported database URL: {db_url}")


def fetchall_dicts(cursor) -> list[dict]:
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
