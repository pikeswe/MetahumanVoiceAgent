"""Configuration loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.json"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration JSON file.

    Parameters
    ----------
    path:
        Optional override path. If omitted the default project config is loaded.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into a base configuration."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_overrides(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["load_config", "merge_overrides"]
