"""Loading of config.yaml."""

from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and return it as a plain dict.

    Raises FileNotFoundError if the file is missing, so a typo in --config
    fails loudly instead of silently falling back to defaults.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(relative: str | Path) -> Path:
    """Turn a config-relative path into an absolute one under the project root."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
