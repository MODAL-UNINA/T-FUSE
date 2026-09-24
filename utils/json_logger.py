"""JSON logging helpers for agent outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _to_json_safe(value: Any) -> Any:
    """Convert numpy-like scalar values to plain Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return value


def save_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Save a JSON payload with stable formatting."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_json_safe(payload), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


