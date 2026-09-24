"""Canonical semantic-family validation without agent/runtime dependencies."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.model_family_state import (
    CANONICAL_FAMILY_ORDER,
    MODEL_TO_FAMILY,
    canonicalize_family_label,
)


def derive_canonical_family_space(runtime_models: Any) -> tuple[List[str], str]:
    if isinstance(runtime_models, list):
        family_set = {
            MODEL_TO_FAMILY[model] for model in runtime_models if model in MODEL_TO_FAMILY
        }
        families = [
            family for family in CANONICAL_FAMILY_ORDER if family in family_set
        ]
        if families:
            return families, "runtime_available"
    registry_families = set(MODEL_TO_FAMILY.values())
    return (
        [family for family in CANONICAL_FAMILY_ORDER if family in registry_families],
        "fallback_registry",
    )


def normalize_signal_families(
    raw: Dict[str, Any], canonical_families: List[str]
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    allowed = set(canonical_families)
    prior = raw.get("model_family_prior")
    if not isinstance(prior, dict):
        return diagnostics

    def normalize(value: Any) -> str | None:
        original = str(value or "").strip()
        canonical = canonicalize_family_label(original.replace("_", " "))
        if canonical in allowed:
            if canonical != original:
                diagnostics.append({
                    "family_alias_normalized": True,
                    "original_family": original,
                    "canonical_family": canonical,
                })
            return canonical
        diagnostics.append({
            "family_alias_normalized": False,
            "original_family": original,
            "canonical_family": None,
            "status": "unknown_family_excluded",
        })
        return None

    for key in ("preferred_model_families", "families_to_deprioritize"):
        items = prior.get(key, [])
        normalized_items = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                family = normalize(item.get("family"))
                if family:
                    normalized_items.append({**item, "family": family})
        prior[key] = normalized_items

    return diagnostics
