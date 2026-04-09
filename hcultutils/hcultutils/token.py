import os
from pathlib import Path

_TOKEN_PATH = Path.home() / ".hcult" / "token"


def load_token() -> str | None:
    if not _TOKEN_PATH.exists():
        return None
    return _TOKEN_PATH.read_text().strip() or None


def save_token(token: str) -> None:
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_PATH.write_text(token)
    os.chmod(_TOKEN_PATH, 0o600)
