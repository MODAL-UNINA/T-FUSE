"""Fuzzy uncertainty/reliability engine for T-FUSE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .fuzzy_membership import clamp, high_input, low_input, medium_input


@dataclass(frozen=True)
class ReliabilityAssessment:
    mode: str
    semantic_trust: float
    uncertainty: float
    semantic_quality: float
    chunk_agreement: float
    evidence_coverage: float
    cross_modal_conflict: float
    source_quality: float
    rule_activations: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rule_activations"] = [dict(item) for item in self.rule_activations]
        return value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return clamp(default)


def extract_semantic_reliability_inputs(
    semantic_context: Mapping[str, Any],
    *,
    cross_modal_conflict: float,
) -> dict[str, float]:
    aggregation = semantic_context.get("aggregation_summary", {})
    evidence = semantic_context.get("evidence_summary", {})
    if not isinstance(aggregation, Mapping):
        aggregation = {}
    if not isinstance(evidence, Mapping):
        evidence = {}

    semantic_quality = _number(
        aggregation.get(
            "raw_aggregation_reliability",
            aggregation.get("reliability_score", 0.0),
        )
    )
    chunk_agreement = _number(
        aggregation.get("chunk_agreement", aggregation.get("domain_agreement", 0.0))
    )
    evidence_coverage = _number(evidence.get("evidence_coverage", 0.0))

    quality_fields = (
        "source_reliability",
        "evidence_specificity",
        "entity_consistency",
        "domain_target_quantity_coherence",
    )
    available = [
        _number(evidence.get(field))
        for field in quality_fields
        if evidence.get(field) is not None
    ]
    source_quality = sum(available) / len(available) if available else semantic_quality

    return {
        "semantic_quality": semantic_quality,
        "chunk_agreement": chunk_agreement,
        "evidence_coverage": evidence_coverage,
        "cross_modal_conflict": _number(cross_modal_conflict, 0.5),
        "source_quality": clamp(source_quality),
    }


def _weighted(rules: list[tuple[str, float, float]], default: float) -> tuple[float, tuple[dict[str, Any], ...]]:
    active = [(name, max(0.0, strength), value) for name, strength, value in rules if strength > 0.0]
    denominator = sum(strength for _, strength, _ in active)
    if denominator <= 1e-12:
        return clamp(default), ()
    output = clamp(sum(strength * value for _, strength, value in active) / denominator)
    trace = tuple(
        {"rule": name, "strength": float(strength), "consequent": float(value)}
        for name, strength, value in active
    )
    return output, trace


def fuzzy_reliability(inputs: Mapping[str, float]) -> ReliabilityAssessment:
    q = _number(inputs.get("semantic_quality"))
    a = _number(inputs.get("chunk_agreement"))
    c = _number(inputs.get("evidence_coverage"))
    x = _number(inputs.get("cross_modal_conflict"), 0.5)
    s = _number(inputs.get("source_quality"), q)

    q_l, q_m, q_h = low_input(q), medium_input(q), high_input(q)
    a_l, a_m, a_h = low_input(a), medium_input(a), high_input(a)
    c_l, c_m, c_h = low_input(c), medium_input(c), high_input(c)
    x_l, x_m, x_h = low_input(x), medium_input(x), high_input(x)
    s_l, s_m, s_h = low_input(s), medium_input(s), high_input(s)

    rules = [
        ("R1_high_quality_agreement_low_conflict", min(q_h, a_h, x_l), 0.92),
        ("R2_high_quality_medium_agreement", min(q_h, a_m, max(x_l, x_m)), 0.75),
        ("R3_medium_evidence", min(q_m, max(a_m, a_h), max(c_m, c_h)), 0.58),
        ("R4_good_source_quality", min(s_h, max(q_m, q_h), max(c_m, c_h)), 0.68),
        ("R5_low_coverage", c_l, 0.22),
        ("R6_low_agreement", a_l, 0.18),
        ("R7_high_conflict", x_h, 0.16),
        ("R8_low_quality", max(q_l, s_l), 0.12),
    ]
    trust, trace = _weighted(rules, default=0.25)

    structural_uncertainty = clamp(
        0.24 * (1.0 - q)
        + 0.22 * (1.0 - a)
        + 0.16 * (1.0 - c)
        + 0.24 * x
        + 0.14 * (1.0 - s)
    )
    uncertainty = clamp(0.55 * (1.0 - trust) + 0.45 * structural_uncertainty)
    trust = clamp(0.80 * trust + 0.20 * (1.0 - structural_uncertainty))

    return ReliabilityAssessment(
        mode="fuzzy",
        semantic_trust=trust,
        uncertainty=uncertainty,
        semantic_quality=q,
        chunk_agreement=a,
        evidence_coverage=c,
        cross_modal_conflict=x,
        source_quality=s,
        rule_activations=trace,
    )


def build_reliability_assessment(
    mode: str,
    semantic_context: Mapping[str, Any],
    *,
    cross_modal_conflict: float,
) -> ReliabilityAssessment:
    mode = str(mode).strip().lower()
    if mode != "fuzzy":
        raise ValueError("T-FUSE reliability accepts only fuzzy influence")
    inputs = extract_semantic_reliability_inputs(
        semantic_context,
        cross_modal_conflict=cross_modal_conflict,
    )
    return fuzzy_reliability(inputs)


__all__ = [
    "ReliabilityAssessment",
    "build_reliability_assessment",
    "extract_semantic_reliability_inputs",
    "fuzzy_reliability",
]
