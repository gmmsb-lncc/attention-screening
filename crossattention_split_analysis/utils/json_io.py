"""Lightweight JSON I/O helpers for analysis checkpoints."""

import json
import os
from typing import Any, Optional


def read_json(path: str) -> Optional[Any]:
    """Read a JSON file; returns None if missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, payload: Any) -> None:
    """Write JSON atomically to reduce corruption risk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)
