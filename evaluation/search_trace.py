"""Append-only, test-free trace events for online search decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from utils.influence_mode import SUPPORTED_INFLUENCE_MODES

REQUIRED_EVENT_KEYS = (
    "event_id",
    "event_type",
    "timestamp",
    "iteration",
    "influence_mode",
)


def resolve_influence_mode(state: Mapping[str, Any]) -> str:
    """Return the sole configured online influence mode."""
    mode = str(state.get("influence_mode", "fuzzy")).lower()
    if mode not in SUPPORTED_INFLUENCE_MODES:
        raise ValueError(f"Unsupported influence_mode in trace: {mode}")
    return mode


def _assert_no_test_fields(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized.startswith("test_") or normalized in {
                "test",
                "test_df",
                "test_metrics",
                "test_score",
                "test_predictions",
                "y_true_test",
            }:
                raise ValueError(f"Test field is forbidden in online trace: {path}{key}")
            _assert_no_test_fields(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_test_fields(item, f"{path}{index}.")


def append_search_event(
    state: dict[str, Any],
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_no_test_fields(payload)
    trace = list(state.get("search_trace", []))
    event = {
        "event_id": f"search-event-{len(trace) + 1}",
        "event_type": str(event_type),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": int(state.get("valid_training_count", state.get("iteration", 0))),
        "influence_mode": str(payload.get("influence_mode", resolve_influence_mode(state))),
        **dict(payload),
    }
    missing = [key for key in REQUIRED_EVENT_KEYS if key not in event]
    if missing:
        raise ValueError(f"Invalid search event, missing: {missing}")
    _assert_no_test_fields(event)
    trace.append(event)
    state["search_trace"] = trace
    return event


__all__ = [
    "REQUIRED_EVENT_KEYS",
    "append_search_event",
    "resolve_influence_mode",
]
