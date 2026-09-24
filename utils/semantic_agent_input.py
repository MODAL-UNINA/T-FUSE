"""Strict input boundary for SemanticAgent.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from utils.semantic_family_space import derive_canonical_family_space


SEMANTIC_AGENT_ALLOWED_STATE_KEYS = (
    "step",
    "strategy",
    "semantic_docs_train_df",
    "semantic_max_docs",
    "semantic_max_chunks",
    "semantic_chunk_size",
    "semantic_deduplicate_facts",
    "semantic_near_duplicate_threshold",
    "semantic_canonical_families",
    "semantic_family_space_status",
    "semantic_aggregation",
    "semantic_target_resolution",
)

SEMANTIC_AGENT_FORBIDDEN_ANALYTICAL_KEYS = frozenset({
    "data_summary", "dataset_length", "analytical_features",
    "analytical_context", "analytical_summary", "series_profile",
    "forecast_horizon", "planner_guidance",
})


def build_semantic_agent_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete and exclusive state visible to SemanticAgent."""
    scoped: dict[str, Any] = {}
    for key in SEMANTIC_AGENT_ALLOWED_STATE_KEYS:
        if key not in state:
            continue
        value = state[key]
        if key == "semantic_docs_train_df" and hasattr(value, "copy"):
            try:
                scoped[key] = value.copy(deep=True)
                continue
            except TypeError:
                pass
        scoped[key] = copy.deepcopy(value)

    catalog = state.get("model_catalog", {})
    available_models = (
        catalog.get("available_models", [])
        if isinstance(catalog, Mapping) else []
    )
    families, status = derive_canonical_family_space(available_models)
    scoped["semantic_canonical_families"] = families
    scoped["semantic_family_space_status"] = status
    if SEMANTIC_AGENT_FORBIDDEN_ANALYTICAL_KEYS.intersection(scoped):
        raise AssertionError("Analytical state crossed the SemanticAgent boundary.")
    return scoped


__all__ = [
    "SEMANTIC_AGENT_ALLOWED_STATE_KEYS",
    "SEMANTIC_AGENT_FORBIDDEN_ANALYTICAL_KEYS",
    "build_semantic_agent_state",
]
