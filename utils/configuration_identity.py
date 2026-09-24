"""Canonical, dependency-free identities for technical configurations."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping


TABULAR_FEATURE_MODELS = frozenset({
    "MLP", "XGBoost", "RandomForest", "LightGBM", "CatBoost",
    "ElasticNet", "KNN", "SVR",
})


def target_transform_label(preprocessing: Any) -> str:
    """Return the declarative target treatment used by a candidate."""
    if isinstance(preprocessing, Mapping):
        transformations = preprocessing.get(
            "transformations",
            preprocessing.get("preprocessing_transformations", []),
        )
    else:
        transformations = preprocessing or []
    for step in transformations if isinstance(transformations, list) else []:
        if not isinstance(step, Mapping):
            continue
        name = str(step.get("name", "")).strip().lower()
        params = step.get("parameters", {})
        if name == "log_transform":
            method = (
                params.get("method", "log1p")
                if isinstance(params, Mapping)
                else "log1p"
            )
            return str(method)
        if name == "boxcox_transform":
            return name
    return "raw"


def log_exploration_status(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize raw/log trials without changing their empirical scores."""
    trials = [
        item for item in state.get("performance_history", []) or []
        if isinstance(item, Mapping)
    ]
    labels = [
        target_transform_label(item.get("preprocessing_transformations", []))
        for item in trials
    ]
    evaluated_by_model: Dict[str, Dict[str, bool]] = {}
    for item, label in zip(trials, labels):
        model = str(item.get("model", ""))
        if not model:
            continue
        status = evaluated_by_model.setdefault(
            model, {"raw_evaluated": False, "log_evaluated": False}
        )
        if label == "raw":
            status["raw_evaluated"] = True
        elif label in {"log1p", "log"}:
            status["log_evaluated"] = True
    log_count = sum(label in {"log1p", "log"} for label in labels)
    raw_count = sum(label == "raw" for label in labels)
    required = bool(state.get("log_candidate_required", False))
    supported = bool(state.get("log_transform_supported", False))
    return {
        "log_transform_recommended": bool(
            state.get("log_transform_recommended", False)
        ),
        "log_transform_supported": supported,
        "log_candidate_required": required,
        "log_candidate_evaluated": log_count > 0,
        "raw_candidate_evaluated": raw_count > 0,
        "raw_candidates_evaluated": raw_count,
        "log_candidates_evaluated": log_count,
        "evaluated_preprocessing_by_model": evaluated_by_model,
        "raw_log_evaluated_by_model": evaluated_by_model,
        "raw_exploration_pending": raw_count == 0,
        "log_exploration_pending": required and supported and log_count == 0,
    }


def normalize_hyperparameters(model: str, hyperparameters: Dict[str, Any] | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize current configs while accepting legacy ETS output in read paths."""
    normalized = dict(hyperparameters or {})
    audit: Dict[str, Any] = {}
    if str(model) == "ETS" and "error" in normalized:
        normalized.pop("error", None)
        audit = {"legacy_parameter_ignored": "error", "model_type": "ETS"}
    return normalized, audit


def normalize_candidate_hyperparameters(
    model: str,
    hyperparameters: Dict[str, Any] | None,
    preprocessing: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Remove the obsolete tabular lag only when preprocessing is unambiguous."""
    normalized, audit = normalize_hyperparameters(model, hyperparameters)
    if str(model) not in TABULAR_FEATURE_MODELS or "lag" not in normalized:
        return normalized, audit
    legacy_lag = int(normalized["lag"])
    transformations = (
        preprocessing.get("transformations", [])
        if isinstance(preprocessing, Mapping)
        else preprocessing or []
    )
    declared_lags: list[int] = []
    for step in transformations if isinstance(transformations, list) else []:
        if not isinstance(step, Mapping) or str(step.get("name", "")).lower() != "lag_features":
            continue
        params = step.get("parameters", {})
        if isinstance(params, Mapping):
            declared_lags.extend(int(value) for value in params.get("lags", []) or [])
    if declared_lags and legacy_lag != max(declared_lags):
        raise ValueError(
            f"Conflicting temporal-memory specifications for {model}: obsolete "
            f"hyperparameters.lag={legacy_lag}, but lag_features declares "
            f"lags={declared_lags}. Remove hyperparameters.lag."
        )
    normalized.pop("lag")
    audit.update({
        "obsolete_parameter_removed": "lag",
        "temporal_memory_source": (
            "preprocessing.lag_features"
            if declared_lags
            else "no_declared_lag_features"
        ),
    })
    return normalized, audit


def _declarative_preprocessing(preprocessing: list[Any]) -> list[Any]:
    """Remove audit-only metadata from candidate identity."""
    normalized: list[Any] = []
    fitted_keys = {
        "columns", "stats", "history_tail", "dropped_rows_train", "train_len",
        "shift", "lambda", "lower_limit", "upper_limit", "limits_source",
        "outlier_positions", "original_values", "replacement_values", "center",
        "scale", "n_replaced", "detection_scope",
    }
    for step in preprocessing:
        if not isinstance(step, Mapping):
            normalized.append(step)
            continue
        item = {"name": step.get("name"), "parameters": step.get("parameters", {})}
        parameters = item["parameters"]
        if isinstance(parameters, Mapping):
            item["parameters"] = {
                str(key): value
                for key, value in parameters.items()
                if not str(key).startswith("_") and str(key) not in fitted_keys
            }
        normalized.append(item)
    return normalized


def canonical_configuration_signature(
    model: str,
    hyperparameters: Dict[str, Any] | None,
    preprocessing: Iterable[Any] | Dict[str, Any] | None,
) -> str:
    if isinstance(preprocessing, dict):
        prep = preprocessing.get("transformations", [])
    elif isinstance(preprocessing, str):
        # Normalize legacy memory records without splitting strings into
        # individual characters (e.g. "None" -> ["N", "o", ...]).
        text = preprocessing.strip()
        prep = [] if not text or text.lower() == "none" else [
            item.strip() for item in text.split(",") if item.strip()
        ]
    else:
        prep = list(preprocessing or [])
    hp, _ = normalize_candidate_hyperparameters(model, hyperparameters, prep)
    prep = _declarative_preprocessing(prep)
    return json.dumps(
        {"model": model, "hyperparameters": hp, "preprocessing": prep},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


def deduplicate_failed_combinations(items: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Keep the first occurrence of each operational configuration."""
    result: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        signature = canonical_configuration_signature(
            str(item.get("model", "")), item.get("hyperparameters", {}), item.get("preprocessing", [])
        )
        if signature not in seen:
            seen.add(signature)
            record = dict(item)
            record["configuration_signature"] = signature
            record.setdefault("first_failed_step", item.get("step"))
            record.setdefault("last_failed_step", item.get("step"))
            record.setdefault("occurrence_count", 1)
            reason = item.get("rejection_code") or item.get("error") or item.get("retry_reason")
            record.setdefault("failure_reasons", [reason] if reason else [])
            result.append(record)
        else:
            record = next(r for r in result if r["configuration_signature"] == signature)
            record["occurrence_count"] = int(record.get("occurrence_count", 1)) + 1
            record["last_failed_step"] = item.get("step", record.get("last_failed_step"))
            reason = item.get("rejection_code") or item.get("error") or item.get("retry_reason")
            reasons = record.setdefault("failure_reasons", [])
            if reason and reason not in reasons:
                reasons.append(reason)
    return result
