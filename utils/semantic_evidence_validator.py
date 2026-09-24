"""Deterministic validation of the Semantic Agent paper output contract."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping


_TOP_LEVEL_FIELDS = {
    "domain_interpretation",
    "inferred_domain",
    "domain_confidence",
    "target_interpretation",
    "model_family_prior",
    "textual_evidence_findings",
}
_DOMAIN_FIELDS = {"domain_label", "domain_parent", "domain_concepts"}
_TARGET_FIELDS = {
    "primary_entity",
    "spatial_scope",
    "measure",
    "physical_quantity",
    "temporal_granularity",
    "target_confidence",
}
_PRIOR_FIELDS = {"preferred_model_families", "families_to_deprioritize"}
_PREFERRED_FIELDS = {"family", "priority", "reason"}
_DEPRIORITIZED_FIELDS = {"family", "reason"}

_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("assumed_sample_sufficiency", re.compile(
        r"\b(?:time\s*series|series|dataset|numerical\s+data|data)\b.{0,45}"
        r"\b(?:short|small|sparse|insufficient\s+observations?|too\s+few)\b", re.I,
    )),
    ("assumed_numerical_missingness", re.compile(
        r"\b(?:time\s*series|series|dataset|numerical\s+data|data)\b.{0,45}"
        r"\b(?:missing\s+values?|many\s+na|incomplete)\b", re.I,
    )),
    ("assumed_sampling_irregularity", re.compile(
        r"\b(?:time\s*series|series|dataset|sampling|numerical\s+data)\b.{0,45}"
        r"\b(?:irregular|uneven|non[- ]regular)\b", re.I,
    )),
    ("assumed_stationarity", re.compile(
        r"\b(?:time\s*series|series|numerical\s+data)\b.{0,45}"
        r"\b(?:stationary|nonstationary|non-stationary|autocorrelat|differenc)", re.I,
    )),
    ("assumed_numerical_dynamics", re.compile(
        r"\b(?:time\s*series|series|numerical\s+data|dataset)\b.{0,45}"
        r"\b(?:volatile|volatility|strong\s+trend|weak\s+trend|seasonality|"
        r"seasonal\s+strength)\b", re.I,
    )),
)


def is_operational_semantic_entry(entry: Any) -> bool:
    """Return whether an entry may contribute to semantic aggregation."""
    if not isinstance(entry, Mapping):
        return True
    return bool(str(entry.get("family", "")).strip())


def _require_exact_fields(
    value: Any, expected: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Semantic Agent {path} must be a JSON object.")
    actual = set(value)
    missing = sorted(expected - actual)
    additional = sorted(actual - expected)
    if missing or additional:
        raise ValueError(
            f"Semantic Agent {path} does not match the paper schema: "
            f"missing={missing}, additional={additional}."
        )
    return copy.deepcopy(dict(value))


def _claim_categories(text: Any) -> list[str]:
    normalized = " ".join(str(text or "").split())
    return [
        category for category, pattern in _CLAIM_PATTERNS if pattern.search(normalized)
    ]


def _sanitize_text(text: Any) -> tuple[str, list[str]]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return "", []
    clauses = [
        part.strip(" ,.")
        for part in re.split(r"\s*;\s*|\s+and\s+", normalized, flags=re.I)
        if part.strip()
    ]
    retained: list[str] = []
    categories: list[str] = []
    for clause in clauses:
        found = _claim_categories(clause)
        if found:
            categories.extend(found)
        else:
            retained.append(clause)
    return "; ".join(retained), list(dict.fromkeys(categories))


def _audit_items(
    categories: list[str], *, source_chunk_index: int, field: str
) -> list[dict[str, Any]]:
    return [
        {
            "source_chunk_index": int(source_chunk_index),
            "field": field,
            "claim_category": category,
            "action": "excluded_from_operational_support",
        }
        for category in categories
    ]


def validate_semantic_output(
    raw: Mapping[str, Any], *, source_chunk_index: int = -1
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and normalize semantic output."""
    output = _require_exact_fields(raw, _TOP_LEVEL_FIELDS, "output")
    domain = _require_exact_fields(
        output["domain_interpretation"], _DOMAIN_FIELDS, "domain_interpretation"
    )
    target = _require_exact_fields(
        output["target_interpretation"], _TARGET_FIELDS, "target_interpretation"
    )
    prior = _require_exact_fields(
        output["model_family_prior"], _PRIOR_FIELDS, "model_family_prior"
    )

    preferred = prior["preferred_model_families"]
    deprioritized = prior["families_to_deprioritize"]
    findings = output["textual_evidence_findings"]
    if not isinstance(preferred, list):
        raise ValueError("Semantic Agent preferred_model_families must be a list.")
    if not isinstance(deprioritized, list):
        raise ValueError("Semantic Agent families_to_deprioritize must be a list.")
    if not isinstance(findings, list):
        raise ValueError("Semantic Agent textual_evidence_findings must be a list.")

    violations: list[dict[str, Any]] = []
    clean_preferred: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(preferred):
        entry = _require_exact_fields(
            raw_entry,
            _PREFERRED_FIELDS,
            f"model_family_prior.preferred_model_families[{index}]",
        )
        entry["reason"], categories = _sanitize_text(entry["reason"])
        violations.extend(_audit_items(
            categories, source_chunk_index=source_chunk_index,
            field="model_family_prior.preferred_model_families",
        ))
        if entry["reason"]:
            clean_preferred.append(entry)

    clean_deprioritized: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(deprioritized):
        entry = _require_exact_fields(
            raw_entry,
            _DEPRIORITIZED_FIELDS,
            f"model_family_prior.families_to_deprioritize[{index}]",
        )
        entry["reason"], categories = _sanitize_text(entry["reason"])
        violations.extend(_audit_items(
            categories, source_chunk_index=source_chunk_index,
            field="model_family_prior.families_to_deprioritize",
        ))
        if entry["reason"]:
            clean_deprioritized.append(entry)

    clean_findings: list[str] = []
    for finding in findings:
        if not isinstance(finding, str):
            raise ValueError("Semantic Agent textual evidence findings must be strings.")
        clean, categories = _sanitize_text(finding)
        violations.extend(_audit_items(
            categories, source_chunk_index=source_chunk_index,
            field="textual_evidence_findings",
        ))
        if clean:
            clean_findings.append(clean)

    return ({
        "domain_interpretation": domain,
        "inferred_domain": output["inferred_domain"],
        "domain_confidence": output["domain_confidence"],
        "target_interpretation": target,
        "model_family_prior": {
            "preferred_model_families": clean_preferred,
            "families_to_deprioritize": clean_deprioritized,
        },
        "textual_evidence_findings": clean_findings,
    }, violations)


__all__ = [
    "is_operational_semantic_entry",
    "validate_semantic_output",
]
