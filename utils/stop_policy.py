"""Centralized, deterministic eligibility policy for adaptive Planner STOP."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from utils.configuration_identity import target_transform_label
from utils.training_budget import count_valid_trainings


def _finite_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for item in state.get("performance_history", []) or []:
        if not isinstance(item, Mapping) or item.get("error"):
            continue
        try:
            mase = float(item.get("mase"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(mase):
            history.append({**dict(item), "mase": mase})
    return history


def _plateau_status(
    history: Sequence[Mapping[str, Any]],
    *,
    patience: int,
    minimum_relative_improvement: float,
) -> tuple[bool, float, float | None]:
    """Return continuous stability from time since meaningful improvement."""
    patience = max(1, int(patience))
    if len(history) < 2:
        return False, 0.0, None
    threshold = max(float(minimum_relative_improvement), 1e-12)
    mase_values = [float(item["mase"]) for item in history]
    incumbent = mase_values[0]
    since_significant_improvement = 0
    for value in mase_values[1:]:
        improvement = max(
            0.0, (incumbent - value) / (abs(incumbent) + 1e-12)
        )
        if improvement >= threshold:
            since_significant_improvement = 0
        else:
            since_significant_improvement += 1
        incumbent = min(incumbent, value)
    previous_best = min(mase_values[:-min(patience, len(mase_values) - 1)])
    current_best = min(mase_values)
    window_improvement = max(
        0.0,
        (previous_best - current_best) / (abs(previous_best) + 1e-12),
    )
    time_component = min(
        1.0, since_significant_improvement / float(patience)
    )
    improvement_component = 1.0 - min(1.0, window_improvement / threshold)
    strength = max(0.0, min(1.0, time_component * improvement_component))
    return strength >= 0.8, strength, window_improvement


def _model_treatments(
    history: Sequence[Mapping[str, Any]], model: str
) -> dict[str, bool]:
    labels = {
        target_transform_label(item.get("preprocessing_transformations", []))
        for item in history
        if str(item.get("model", "")) == model
    }
    return {
        "raw_evaluated": "raw" in labels,
        "log_evaluated": bool(labels.intersection({"log", "log1p"})),
    }


def _distinct_hyperparameters(
    history: Sequence[Mapping[str, Any]], model: str
) -> int:
    return len({
        json.dumps(
            item.get("hyperparameters", {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for item in history
        if str(item.get("model", "")) == model
    })


def stop_eligibility(
    state: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the sole auditable answer to whether Planner STOP is allowed.

    The hard maximum overrides every adaptive condition. Before the hard cap,
    adaptive STOP requires the minimum valid budget and no mandatory
    raw/log/clip/current-best refinement. Plateau is only a fuzzy input.
    """
    cfg = dict(state.get("planner_policy", {}) or {})
    cfg.update(dict(config or {}))
    history = _finite_history(state)
    valid_trainings = count_valid_trainings(history)
    maximum = int(state.get("maximum_valid_trainings", 0) or 0)
    maximum = maximum or int(state.get("max_iterations", 0) or 0)
    minimum = int(state.get("minimum_valid_trainings_before_stop", 5) or 5)
    patience = int(cfg.get("stopping_patience", 5) or 5)
    improvement_threshold = float(
        cfg.get("minimum_relative_improvement", 0.002) or 0.002
    )
    plateau, plateau_strength, relative_improvement = _plateau_status(
        history,
        patience=patience,
        minimum_relative_improvement=improvement_threshold,
    )
    hard_stop = maximum > 0 and valid_trainings >= maximum
    minimum_reached = valid_trainings >= minimum
    remaining = max(0, maximum - valid_trainings) if maximum > 0 else 0

    current_best = min(
        history, key=lambda item: float(item["mase"]), default=None
    )
    current_best_model = (
        str(current_best.get("model", "")) if current_best else None
    )
    treatments = _model_treatments(history, current_best_model) if current_best_model else {
        "raw_evaluated": False,
        "log_evaluated": False,
    }

    any_raw = any(
        target_transform_label(item.get("preprocessing_transformations", []))
        == "raw"
        for item in history
    )
    required_raw_pending = bool(history and not any_raw)
    log_recommended = bool(state.get("log_transform_recommended", False))
    log_supported = bool(state.get("log_transform_supported", False))
    required_log_pending = bool(
        current_best_model
        and log_recommended
        and log_supported
        and not (
            treatments["raw_evaluated"] and treatments["log_evaluated"]
        )
    )
    max_refinements = int(cfg.get("max_refinement_trials_per_model", 4) or 4)
    model_trials = sum(
        str(item.get("model", "")) == current_best_model
        for item in history
    )
    distinct_hyperparameters = (
        _distinct_hyperparameters(history, current_best_model)
        if current_best_model
        else 0
    )
    hyperparameter_refinement_complete = (
        not current_best_model
        or max_refinements <= 1
        or model_trials >= max_refinements
        or distinct_hyperparameters >= 2
    )
    refinement_required = bool(
        state.get(
            "refinement_is_required_by_existing_policy",
            current_best_model is not None
            and minimum_reached
            and max_refinements > 1,
        )
    )
    required_model_pending = bool(
        current_best_model
        and refinement_required
        and not hyperparameter_refinement_complete
    )
    hyperparameter_completeness = (
        1.0 if hyperparameter_refinement_complete else 0.0
    )
    target_treatment_completeness = (
        (
            float(treatments["raw_evaluated"])
            + float(treatments["log_evaluated"])
        ) / 2.0
        if log_recommended and log_supported
        else 1.0
    )
    refinement_completeness = (
        (
            hyperparameter_completeness
            + target_treatment_completeness
        ) / 2.0
        if current_best_model
        else 0.0
    )
    current_best_refinement_complete = bool(
        current_best_model
        and refinement_completeness >= 1.0 - 1e-12
    )

    mandatory_pending = bool(
        required_raw_pending
        or required_log_pending
        or required_model_pending
    )
    blockers: list[str] = []
    if not minimum_reached:
        blockers.append("minimum_valid_trainings_not_reached")
    if required_raw_pending:
        blockers.append("required_raw_candidate_pending")
    if required_log_pending:
        blockers.append("required_log_refinement_pending")
    if required_model_pending:
        blockers.append("current_best_refinement_incomplete")
    eligible = bool(
        hard_stop or (minimum_reached and not mandatory_pending)
    )
    return {
        "eligible": eligible,
        "stop_eligible": eligible,
        "hard_stop_required": hard_stop,
        "reasons_blocking_stop": [] if hard_stop else blockers,
        "valid_trainings": valid_trainings,
        "remaining_budget": remaining,
        "maximum_valid_trainings": maximum,
        "minimum_valid_trainings_before_stop": minimum,
        "minimum_budget_reached": minimum_reached,
        "plateau_detected": plateau,
        "plateau_strength": plateau_strength,
        "relative_improvement_over_patience": relative_improvement,
        "stopping_patience": patience,
        "minimum_relative_improvement": improvement_threshold,
        "mandatory_refinement_pending": mandatory_pending,
        "required_raw_candidate_pending": required_raw_pending,
        "required_log_refinement_pending": required_log_pending,
        "required_model_refinement_pending": required_model_pending,
        "current_best_refinement_complete": current_best_refinement_complete,
        "refinement_completeness": refinement_completeness,
        "hyperparameter_refinement_completeness": (
            hyperparameter_completeness
        ),
        "target_treatment_completeness": target_treatment_completeness,
        "current_best_model": current_best_model,
        "current_best_preprocessing": {
            **treatments,
        },
        "current_best_distinct_hyperparameters": distinct_hyperparameters,
        "current_best_model_trials": model_trials,
    }


__all__ = ["stop_eligibility"]
