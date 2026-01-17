"""Configuration loader for hcultctrl."""

import os
from configparser import ConfigParser
from pathlib import Path


DEFAULT_CONFIG_NAME = "openhcult.conf"
DEFAULT_LOG_DIR = "/var/log/hcult"
DEFAULT_LOG_STDOUT = True


def _default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "openhcult" / DEFAULT_CONFIG_NAME
    return Path.home() / ".config" / "openhcult" / DEFAULT_CONFIG_NAME


def _load_config():
    """Return the parsed repo-level config and its path."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    parser = ConfigParser()
    parser.read(config_path)
    return parser, config_path, repo_root


def get_db_path() -> Path:
    """Fetch the database path from the config."""
    parser, config_path, repo_root = _load_config()
    if "database" not in parser or "path" not in parser["database"]:
        raise ValueError(f"Missing database.path in {config_path}")
    configured = parser["database"]["path"].strip()
    if not configured:
        raise ValueError(f"Empty database.path in {config_path}")
    db_path = Path(configured).expanduser()
    if not db_path.is_absolute():
        db_path = repo_root / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


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
