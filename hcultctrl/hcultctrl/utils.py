from datetime import datetime, timezone
import json
from fastapi import HTTPException
from . import config

from hcultdb.connection import connect


def parse_utc_ms(value: str, field: str) -> int:
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value}") from exc


def normalize_metadata(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def get_db_conn():
    db_url = config.get_db_url()
    conn = connect(db_url)
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

