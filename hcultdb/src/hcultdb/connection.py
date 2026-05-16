"""Shared DB connection helpers."""

from __future__ import annotations

from urllib.parse import urlparse

import psycopg


def is_postgres(conn) -> bool:
    module = conn.__class__.__module__
    return "psycopg" in module or "psycopg2" in module


def placeholder(conn) -> str:
    return "%s" if is_postgres(conn) else RuntimeError("Unsupported database connection type")


def connect(db_url: str):
    parsed = urlparse(db_url)
    if parsed.scheme.startswith("postgres"):
        return psycopg.connect(db_url, connect_timeout=10)
    raise ValueError(f"Unsupported database URL: {db_url}")


def fetchall_dicts(cursor) -> list[dict]:
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
