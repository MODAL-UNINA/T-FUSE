"""Reliability-aware analytical/semantic evidence fusion.

"""

from __future__ import annotations

from typing import Any, Mapping

from utils.model_family_state import CANONICAL_FAMILY_ORDER
from utils.semantic_normalization import family_score_discriminability
from .fuzzy_membership import clamp
from .fuzzy_reliability import build_reliability_assessment


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _raw_semantic_support(family: str, semantic_context: Mapping[str, Any]) -> tuple[float, float]:
    support = semantic_context.get("family_support", {})
    item = support.get(family, {}) if isinstance(support, Mapping) else {}
    if not isinstance(item, Mapping):
        return 0.5, 0.0
    score = clamp(
        _finite(item.get("normalized_support", item.get("net", 0.0)))
    )

    available = item.get("evidence_available")
    if available is None:
        available = any(
            key in item
            for key in ("normalized_support", "net", "positive", "negative")
        )
    return score, 1.0 if bool(available) else 0.0


def _raw_cross_modal_conflict(
    family_results: Mapping[str, Any],
    semantic_context: Mapping[str, Any],
) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    covered = 0
    for family in CANONICAL_FAMILY_ORDER:
        analytical = family_results.get(family, {})
        if not isinstance(analytical, Mapping):
            analytical = {}
        analytical_score = clamp(_finite(analytical.get("analytical_compatibility", 0.5), 0.5))
        analytical_rel = clamp(_finite(analytical.get("analytical_reliability", 0.0)))
        semantic_score, semantic_available = _raw_semantic_support(family, semantic_context)
        pair_rel = min(analytical_rel, semantic_available)
        if pair_rel > 0:
            covered += 1
            numerator += pair_rel * abs(semantic_score - analytical_score)
            denominator += pair_rel
    conflict = clamp(numerator / denominator) if denominator > 1e-12 else 0.5
    coverage = covered / max(1, len(CANONICAL_FAMILY_ORDER))
    return conflict, coverage


def fuse_evidence(
    analytical_mapping: Mapping[str, Any],
    semantic_context: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Fuse family evidence after uncertainty-aware semantic trust estimation."""
    mode = str(mode).strip().lower()
    family_results = analytical_mapping.get("family_results", {})
    if not isinstance(family_results, Mapping):
        family_results = {}

    raw_conflict, cross_modal_coverage = _raw_cross_modal_conflict(
        family_results, semantic_context
    )
    assessment = build_reliability_assessment(
        mode,
        semantic_context,
        cross_modal_conflict=raw_conflict,
    )
    semantic_trust = assessment.semantic_trust

    combined: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    analytical_reliabilities: list[float] = []

    for family in CANONICAL_FAMILY_ORDER:
        analytical = family_results.get(family, {})
        if not isinstance(analytical, Mapping):
            analytical = {}
        analytical_score = clamp(_finite(analytical.get("analytical_compatibility", 0.5), 0.5))
        analytical_rel = clamp(_finite(analytical.get("analytical_reliability", 0.0)))
        analytical_reliabilities.append(analytical_rel)

        semantic_score, semantic_available = _raw_semantic_support(family, semantic_context)
        semantic_rel = semantic_trust * semantic_available
        denom = analytical_rel + semantic_rel
        score = (
            clamp((analytical_rel * analytical_score + semantic_rel * semantic_score) / denom)
            if denom > 1e-12
            else 0.5
        )
        combined[family] = score
        details[family] = {
            "combined_compatibility": score,
            "semantic_support": semantic_score,
            "semantic_reliability": semantic_rel,
            "semantic_source": "cross_family_normalized_semantic_support" if semantic_available else "semantic_unavailable",
            "analytical_compatibility": analytical_score,
            "analytical_reliability": analytical_rel,
        }

    analytical_trust = (
        sum(analytical_reliabilities) / len(analytical_reliabilities)
        if analytical_reliabilities
        else 0.0
    )
    alignment = clamp(1.0 - raw_conflict)
    reliability = clamp(
        0.40 * semantic_trust
        + 0.35 * analytical_trust
        + 0.15 * alignment
        + 0.10 * cross_modal_coverage
    )

    ordered = sorted(
        CANONICAL_FAMILY_ORDER,
        key=lambda family: (-combined[family], CANONICAL_FAMILY_ORDER.index(family)),
    )
    margin = combined[ordered[0]] - combined[ordered[1]] if len(ordered) > 1 else 0.0
    semantic_scores = {
        family: _raw_semantic_support(family, semantic_context)[0]
        for family in CANONICAL_FAMILY_ORDER
    }
    semantic_discriminability = family_score_discriminability(semantic_scores)

    return {
        "status": "ready",
        "mode": mode,
        "family_compatibility": combined,
        "family_order": ordered,
        "family_details": details,
        "semantic_trust": semantic_trust,
        "semantic_uncertainty": assessment.uncertainty,
        "semantic_reliability": assessment.to_dict(),
        "analytical_trust": clamp(analytical_trust),
        "fusion_reliability": reliability,
        "cross_source_alignment": alignment,
        "conflict": raw_conflict,
        "coverage": cross_modal_coverage,
        "top_margin": margin,
        "discriminability": semantic_discriminability,
        "semantic_discriminability": semantic_discriminability,
        "semantic_family_scores": semantic_scores,
        "discriminability_method": "semantic_support_cross_family_population_std_scaled_by_0.5",
        "fused_top_margin": margin,
    }


__all__ = ["fuse_evidence"]
