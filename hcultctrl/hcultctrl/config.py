"""Configuration loader for hcultctrl."""

import os
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_NAME = "openhcult.conf"
DEFAULT_LOG_DIR = "/var/log/hcult"
DEFAULT_LOG_STDOUT = True
DEFAULT_CTRL_HOST = "127.0.0.1"
DEFAULT_CTRL_PORT = 8000
_CONFIG_PATH_OVERRIDE: Optional[Path] = None


def _default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "openhcult" / DEFAULT_CONFIG_NAME
    return Path.home() / ".config" / "openhcult" / DEFAULT_CONFIG_NAME


def _load_config():
    """Return the parsed repo-level config and its path."""
    repo_root = Path(__file__).resolve().parents[2]
    env_path = os.environ.get("HCULT_CONFIG_PATH")
    config_path = (
        _CONFIG_PATH_OVERRIDE
        or (Path(env_path) if env_path else None)
        or _default_config_path()
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    parser = ConfigParser()
    parser.read(config_path)
    return parser, config_path, repo_root


def set_config_path(config_path: Path) -> None:
    """Override the config path used by hcultctrl."""
    global _CONFIG_PATH_OVERRIDE
    _CONFIG_PATH_OVERRIDE = config_path


def get_db_url() -> str:
    """Fetch the database URL from config."""
    parser, config_path, _ = _load_config()
    if "database" not in parser or "url" not in parser["database"]:
        raise ValueError(f"Missing database.url in {config_path}")
    url = parser["database"]["url"].strip()
    if not url:
        raise ValueError(f"Empty database.url in {config_path}")
    return url


def get_log_path(service_name: str) -> Path:
    """Fetch the log file path from config, defaulting to /var/log/hcult."""
    parser, config_path, _ = _load_config()
    log_dir = DEFAULT_LOG_DIR
    log_file = None
    if "logging" in parser:
        if "file" in parser["logging"]:
            log_file = parser["logging"]["file"].strip()
        elif "path" in parser["logging"]:
            log_dir = parser["logging"]["path"].strip()
    if log_file is not None:
        if log_file == "" or log_file.lower() in {"none", "null"}:
            return None
        log_path = Path(log_file).expanduser()
    else:
        log_path = Path(log_dir).expanduser() / f"{service_name}.log"
    return log_path


def get_log_stdout() -> bool:
    """Return whether logs should also go to stdout."""
    parser, _, _ = _load_config()
    if "logging" not in parser or "stdout" not in parser["logging"]:
        return DEFAULT_LOG_STDOUT
    value = parser["logging"]["stdout"].strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return DEFAULT_LOG_STDOUT


def get_ctrl_host() -> str:
    """Return the bind host for hcultctrl."""
    parser, _, _ = _load_config()
    if "ctrl" in parser and "host" in parser["ctrl"]:
        value = parser["ctrl"]["host"].strip()
        if value:
            return value
    return DEFAULT_CTRL_HOST


def get_ctrl_port() -> int:
    """Return the bind port for hcultctrl."""
    parser, _, _ = _load_config()
    if "ctrl" in parser and "port" in parser["ctrl"]:
        value = parser["ctrl"]["port"].strip()
        if value.isdigit():
            return int(value)
    return DEFAULT_CTRL_PORT


def get_credentials_path() -> Path:
    config_path = _CONFIG_PATH_OVERRIDE or _default_config_path()
    return config_path.parent / "credentials.json"


def get_auth_token_expiry_hours() -> int:
    parser, _, _ = _load_config()
    if "auth" in parser and "token_expiry_hours" in parser["auth"]:
        value = parser["auth"]["token_expiry_hours"].strip()
        if value.isdigit():
            return int(value)
    return 24
