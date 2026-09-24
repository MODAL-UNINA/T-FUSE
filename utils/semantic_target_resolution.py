"""Deterministic quantitative-target resolution from semantic evidence only."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from utils.semantic_normalization import normalize_semantic_text


DEFAULT_TARGET_RESOLUTION_CONFIG: dict[str, float] = {
    "material_alternative_threshold": 0.15,
    "quantity_similarity_threshold": 0.80,
    "resolved_primary_support": 0.75,
    "resolved_margin": 0.40,
    "resolved_agreement": 0.70,
    "resolved_reliability": 0.55,
    "resolved_max_conflict": 0.20,
    "resolved_max_alternative_total": 0.20,
    "alternatives_primary_support": 0.65,
    "alternatives_margin": 0.30,
    "alternatives_agreement": 0.55,
    "alternatives_reliability": 0.40,
    "alternatives_max_conflict": 0.35,
    "alternatives_max_total": 0.35,
    "weak_primary_support": 0.50,
    "weak_margin": 0.15,
    "weak_max_conflict": 0.60,
    "unresolved_min_agreement": 0.35,
}

TARGET_RESOLUTION_SCORES = {
    "resolved": 1.00,
    "resolved_with_material_alternatives": 0.70,
    "weakly_resolved": 0.40,
    "unresolved": 0.10,
    "unavailable": 0.00,
}

TARGET_AMBIGUITY_BY_STATUS = {
    "resolved": "low",
    "resolved_with_material_alternatives": "medium",
    "weakly_resolved": "high",
    "unresolved": "high",
    "unavailable": "high",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return _clamp(number) if number == number else None


def _config(config: Mapping[str, Any] | None) -> dict[str, float]:
    result = dict(DEFAULT_TARGET_RESOLUTION_CONFIG)
    if not isinstance(config, Mapping):
        return result
    for key, default in result.items():
        try:
            result[key] = _clamp(float(config.get(key, default)))
        except (TypeError, ValueError):
            result[key] = default
    return result


def _as_dict(signal: Any) -> dict[str, Any]:
    if isinstance(signal, dict):
        return dict(signal)
    if hasattr(signal, "model_dump"):
        return signal.model_dump()
    value = getattr(signal, "__dict__", {})
    return dict(value) if isinstance(value, dict) else {}


def _known_semantic_label(value: Any) -> str | None:
    label = str(value or "").strip()
    normalized = normalize_semantic_text(label)
    if normalized in {"unknown", "none", "null"}:
        return None
    return label


def _cluster_labels(
    labels: Sequence[str], weights: Sequence[float], *, encoder: Any | None,
    similarity_threshold: float,
) -> list[tuple[list[int], str, float]]:
    """Cluster arbitrary semantic labels without a domain quantity taxonomy."""
    if not labels:
        return []
    if encoder is not None:
        # Imported lazily so the deterministic exact-match fallback remains
        # usable in lightweight environments without numerical dependencies.
        from utils.semantic_clustering import cluster_semantic_labels

        return [
            (
                list(cluster.item_indices),
                str(cluster.representative_label),
                float(cluster.total_weight),
            )
            for cluster in cluster_semantic_labels(
                list(labels), weights=list(weights), encoder=encoder,
                similarity_threshold=similarity_threshold,
            )
        ]

    exact_groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        exact_groups.setdefault(normalize_semantic_text(label), []).append(index)
    clusters: list[tuple[list[int], str, float]] = []
    for normalized, indices in sorted(exact_groups.items()):
        representative = min(
            (str(labels[index]) for index in indices),
            key=lambda label: (normalize_semantic_text(label), label),
        )
        clusters.append((
            indices,
            representative,
            sum(float(weights[index]) for index in indices),
        ))
    return clusters


def _representative_measure(
    candidates: Sequence[Mapping[str, Any]], indices: Sequence[int], *,
    encoder: Any | None, similarity_threshold: float,
) -> str:
    labels: list[str] = []
    weights: list[float] = []
    for index in indices:
        candidate = candidates[index]
        measures = candidate.get("measures", [])
        if not isinstance(measures, list):
            measures = [measures]
        valid = [label for value in measures if (label := _known_semantic_label(value))]
        if not valid:
            continue
        shared_weight = float(candidate.get("weight", 0.0)) / len(valid)
        labels.extend(valid)
        weights.extend(shared_weight for _ in valid)
    clusters = _cluster_labels(
        labels, weights, encoder=encoder,
        similarity_threshold=similarity_threshold,
    )
    if not clusters:
        return "unknown"
    _, representative, _ = min(
        clusters,
        key=lambda cluster: (
            -cluster[2], normalize_semantic_text(cluster[1]), cluster[1]
        ),
    )
    return representative


def classify_target_resolution(
    hypotheses: Sequence[Mapping[str, Any]], *, target_confidence: float,
    target_agreement: float, target_quantity_conflict: float,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify target status using one centralized configurable rule set."""
    cfg = _config(config)
    valid = [
        dict(item) for item in hypotheses
        if isinstance(item, Mapping) and (_number(item.get("support")) or 0.0) > 0.0
    ]
    valid.sort(key=lambda item: (
        -(_number(item.get("support")) or 0.0),
        str(item.get("physical_quantity", "")), str(item.get("measure", "")),
    ))
    confidence = _number(target_confidence) or 0.0
    agreement = _number(target_agreement) or 0.0
    reliability = _clamp(confidence * agreement)
    conflict = _number(target_quantity_conflict) or 0.0
    if not valid:
        status = "unavailable"
        primary = second = alternative_total = margin = 0.0
        material_count = 0
    else:
        primary = _number(valid[0].get("support")) or 0.0
        second = _number(valid[1].get("support")) or 0.0 if len(valid) > 1 else 0.0
        alternative_total = sum(
            _number(item.get("support")) or 0.0 for item in valid[1:]
        )
        margin = primary - second
        material_count = sum(
            (_number(item.get("support")) or 0.0)
            >= cfg["material_alternative_threshold"]
            for item in valid[1:]
        )

        unresolved = (
            primary < cfg["weak_primary_support"]
            or margin < cfg["weak_margin"]
            or conflict >= cfg["weak_max_conflict"]
            or agreement < cfg["unresolved_min_agreement"]
        )
        resolved = (
            primary >= cfg["resolved_primary_support"]
            and margin >= cfg["resolved_margin"]
            and agreement >= cfg["resolved_agreement"]
            and reliability >= cfg["resolved_reliability"]
            and conflict <= cfg["resolved_max_conflict"]
            and alternative_total <= cfg["resolved_max_alternative_total"]
            and material_count == 0
        )
        resolved_with_alternatives = (
            primary >= cfg["alternatives_primary_support"]
            and margin >= cfg["alternatives_margin"]
            and agreement >= cfg["alternatives_agreement"]
            and reliability >= cfg["alternatives_reliability"]
            and conflict <= cfg["alternatives_max_conflict"]
            and alternative_total <= cfg["alternatives_max_total"]
            and material_count >= 1
        )
        if unresolved:
            status = "unresolved"
        elif resolved:
            status = "resolved"
        elif resolved_with_alternatives:
            status = "resolved_with_material_alternatives"
        elif (
            primary >= cfg["weak_primary_support"]
            and margin >= cfg["weak_margin"]
            and conflict < cfg["weak_max_conflict"]
        ):
            status = "weakly_resolved"
        else:
            status = "unresolved"

    return {
        "target_resolution_status": status,
        "target_resolution_score": TARGET_RESOLUTION_SCORES[status],
        "target_ambiguity": TARGET_AMBIGUITY_BY_STATUS[status],
        "primary_hypothesis_support": primary,
        "second_hypothesis_support": second,
        "alternative_total_support": alternative_total,
        "hypothesis_margin": margin,
        "number_of_material_alternatives": int(material_count),
        "material_alternative_threshold": cfg["material_alternative_threshold"],
        "target_confidence": confidence,
        "target_agreement": agreement,
        "target_reliability": reliability,
        "target_quantity_conflict": conflict,
        "classification_config": cfg,
    }


def validate_target_resolution_consistency(
    resolution: Mapping[str, Any], target_ambiguity: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return the canonical ambiguity and an audit item when correction is needed."""
    status = str(resolution.get("target_resolution_status", "unavailable"))
    expected = TARGET_AMBIGUITY_BY_STATUS.get(status, "high")
    if target_ambiguity == expected:
        return expected, None
    return expected, {
        "action": "recalculated_from_central_target_resolution",
        "previous_target_ambiguity": str(target_ambiguity),
        "target_resolution_status": status,
        "corrected_target_ambiguity": expected,
    }


def resolve_target_quantity(
    signals: Sequence[Any], *, target_confidence: float, target_agreement: float,
    config: Mapping[str, Any] | None = None, encoder: Any | None = None,
) -> dict[str, Any]:
    """Resolve free-form physical quantities without a hand-written taxonomy."""
    cfg = _config(config)
    groups: dict[int, list[dict[str, Any]]] = {}
    legacy: list[dict[str, Any]] = []
    for signal in signals:
        item = _as_dict(signal)
        try:
            index = int(item.get("source_chunk_index", -1))
        except (TypeError, ValueError):
            index = -1
        if index >= 0:
            groups.setdefault(index, []).append(item)
        else:
            legacy.append(item)
    legacy.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    for offset, item in enumerate(legacy, start=1):
        groups[-offset] = [item]

    candidates: list[dict[str, Any]] = []
    for index, items in sorted(groups.items()):
        confidences: list[float] = []
        quantities: dict[str, dict[str, Any]] = {}
        for item in items:
            target = item.get("target_interpretation", {})
            if not isinstance(target, Mapping):
                continue
            domain_confidence = _number(item.get("domain_confidence"))
            target_confidence_item = _number(target.get("target_confidence"))
            confidences.extend(
                value for value in (domain_confidence, target_confidence_item)
                if value is not None
            )
            quantity = _known_semantic_label(target.get("physical_quantity"))
            if quantity is None:
                continue
            normalized = normalize_semantic_text(quantity)
            quantity_item = quantities.setdefault(normalized, {
                "physical_quantity": quantity,
                "measures": set(),
                "unit_candidates": set(),
            })
            measure = _known_semantic_label(target.get("measure"))
            if measure:
                quantity_item["measures"].add(measure)
            units = target.get("unit_candidates", [])
            if not isinstance(units, list):
                units = [units]
            quantity_item["unit_candidates"].update(
                unit for value in units
                if (unit := _known_semantic_label(value))
            )
        chunk_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        if not quantities:
            continue
        candidate_weight = chunk_confidence / len(quantities)
        for normalized in sorted(quantities):
            quantity_item = quantities[normalized]
            candidates.append({
                "physical_quantity": quantity_item["physical_quantity"],
                "measures": sorted(quantity_item["measures"]),
                "unit_candidates": sorted(quantity_item["unit_candidates"]),
                "source_chunk_index": index,
                "weight": candidate_weight,
            })

    labels = [str(candidate["physical_quantity"]) for candidate in candidates]
    weights = [float(candidate["weight"]) for candidate in candidates]
    quantity_clusters = _cluster_labels(
        labels, weights, encoder=encoder,
        similarity_threshold=cfg["quantity_similarity_threshold"],
    )
    total = sum(cluster[2] for cluster in quantity_clusters)
    hypotheses: list[dict[str, Any]] = []
    for indices, representative, support_weight in quantity_clusters:
        units = sorted({
            str(unit)
            for candidate_index in indices
            for unit in candidates[candidate_index].get("unit_candidates", [])
        }, key=lambda value: (normalize_semantic_text(value), value))
        hypotheses.append({
            "measure": _representative_measure(
                candidates, indices, encoder=encoder,
                similarity_threshold=cfg["quantity_similarity_threshold"],
            ),
            "physical_quantity": representative,
            "support": support_weight / total if total else 0.0,
            "source_chunk_indices": sorted({
                int(candidates[candidate_index]["source_chunk_index"])
                for candidate_index in indices
            }),
            "unit_candidates": units,
        })
    hypotheses.sort(key=lambda item: (
        -float(item["support"]),
        normalize_semantic_text(str(item["physical_quantity"])),
        normalize_semantic_text(str(item["measure"])),
    ))
    quantity_conflict = (
        _clamp(1.0 - hypotheses[0]["support"]) if hypotheses else 0.0
    )
    classification = classify_target_resolution(
        hypotheses,
        target_confidence=target_confidence,
        target_agreement=target_agreement,
        target_quantity_conflict=quantity_conflict,
        config=cfg,
    )
    evidence_chunks = sorted({
        int(candidate["source_chunk_index"]) for candidate in candidates
    })
    return {
        **classification,
        "target_reliability_source": "confidence_times_agreement",
        "quantity_resolution_method": "semantic_physical_quantity_clustering",
        "quantity_evidence_chunk_count": len(evidence_chunks),
        "quantity_total_chunk_count": len(groups),
        "quantity_evidence_coverage": (
            len(evidence_chunks) / len(groups) if groups else 0.0
        ),
        "primary_hypothesis": hypotheses[0] if hypotheses else None,
        "alternative_hypotheses": hypotheses[1:],
    }


__all__ = [
    "DEFAULT_TARGET_RESOLUTION_CONFIG",
    "TARGET_AMBIGUITY_BY_STATUS",
    "TARGET_RESOLUTION_SCORES",
    "classify_target_resolution",
    "resolve_target_quantity",
    "validate_target_resolution_consistency",
]
