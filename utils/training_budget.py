"""Deterministic budget helpers for paired search experiments."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def count_valid_trainings(history: Iterable[Mapping[str, Any]] | None) -> int:
    """Count completed trainings that produced a finite technical MASE."""
    count = 0
    for entry in history or []:
        if not isinstance(entry, Mapping) or entry.get("error"):
            continue
        try:
            mase = float(entry.get("mase"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(mase):
            count += 1
    return count


def adaptive_stop_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return run-level search-efficiency fields from valid trials only."""
    actual = count_valid_trainings(state.get("performance_history", []))
    maximum = int(state.get("maximum_valid_trainings", 0) or 0)
    minimum = int(state.get("minimum_valid_trainings_before_stop", 5) or 5)
    reason = state.get("stop_reason")
    return {
        "maximum_valid_trainings": maximum,
        "minimum_valid_trainings_before_stop": minimum,
        "actual_valid_trainings": actual,
        "early_stop_used": bool(
            maximum > 0
            and actual < maximum
            and reason not in {None, "budget_exhausted"}
        ),
    }
