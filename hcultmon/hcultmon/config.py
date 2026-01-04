"""Configuration loader for hcultmon."""

from configparser import ConfigParser
from pathlib import Path


DEFAULT_CONFIG_NAME = "openhcult.conf"


def _load_config():
    """Return the parsed repo-level config and its path."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / DEFAULT_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    parser = ConfigParser()
    parser.read(config_path)
    return parser, config_path, repo_root


def get_ble_characteristic_uuid():
    """Fetch the BLE characteristic UUID from the config."""
    parser, config_path, _ = _load_config()
    if "ble" not in parser or "characteristic_uuid" not in parser["ble"]:
        raise ValueError(f"Missing ble.characteristic_uuid in {config_path}")
    return parser["ble"]["characteristic_uuid"].strip()


def get_db_path():
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
