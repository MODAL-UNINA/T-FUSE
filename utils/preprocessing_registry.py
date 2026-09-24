"""Canonical preprocessing contract shared by prompt, parser and runtime."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


class FrameworkConfigurationError(RuntimeError):
    """Fatal mismatch between the declared and implemented runtime contract."""


CANONICAL_PREPROCESSING_TRANSFORMATIONS = (
    "standard_scaler",
    "minmax_scaler",
    "lag_features",
    "rolling_features",
    "calendar_features",
    "log_transform",
    "boxcox_transform",
    "replace_outliers",
)

LLM_SELECTABLE_PREPROCESSING_TRANSFORMATIONS = tuple(
    name for name in CANONICAL_PREPROCESSING_TRANSFORMATIONS
    if name != "replace_outliers"
)

TRANSFORMATION_ALIASES = {
    "standardscaler": "standard_scaler",
    "standard scaler": "standard_scaler",
    "zscore": "standard_scaler",
    "z_score": "standard_scaler",
    "z-score": "standard_scaler",
    "minmaxscaler": "minmax_scaler",
    "minmax scaler": "minmax_scaler",
    "min_max_scaler": "minmax_scaler",
    "min-max scaler": "minmax_scaler",
}

TARGET_TREATMENTS = frozenset({
    "log_transform",
    "boxcox_transform",
})

RUNTIME_FIT_HANDLERS = {
    "standard_scaler": "_fit_apply_scaler",
    "minmax_scaler": "_fit_apply_scaler",
    "lag_features": "_fit_apply_lag_features",
    "rolling_features": "_fit_apply_rolling_features",
    "calendar_features": "_apply_calendar_features",
    "log_transform": "_fit_apply_log_transform",
    "boxcox_transform": "_fit_apply_boxcox_transform",
    "replace_outliers": "_fit_apply_replace_outliers",
}


def canonicalize_transformation_name(raw_name: Any) -> str:
    if not isinstance(raw_name, str):
        raise ValueError(
            "Each preprocessing transformation must have a string 'name'."
        )
    normalized = raw_name.strip().lower()
    normalized = TRANSFORMATION_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_PREPROCESSING_TRANSFORMATIONS:
        raise ValueError(
            f"Unknown preprocessing transformation: {raw_name}. Allowed names "
            f"are: {list(CANONICAL_PREPROCESSING_TRANSFORMATIONS)}"
        )
    return normalized


def validate_target_treatment_combination(names: Iterable[str]) -> None:
    selected = sorted(set(names).intersection(TARGET_TREATMENTS))
    if len(selected) > 1:
        raise ValueError(
            "Target treatments are mutually exclusive and must be evaluated "
            f"as separate candidates; received: {selected}"
        )


def assert_preprocessing_registry_consistency(
    declared: Iterable[str], implemented: Iterable[str]
) -> None:
    declared_set = set(declared)
    implemented_set = set(implemented)
    if declared_set == implemented_set:
        return
    missing = sorted(declared_set - implemented_set)
    undeclared = sorted(implemented_set - declared_set)
    raise FrameworkConfigurationError(
        "Fatal configuration inconsistency: preprocessing transformations "
        f"declared but not implemented={missing}; implemented but not "
        f"declared={undeclared}."
    )


def runtime_implemented_transformations(runtime_type: type) -> set[str]:
    return {
        name
        for name, handler_name in RUNTIME_FIT_HANDLERS.items()
        if callable(getattr(runtime_type, handler_name, None))
    }


def normalized_error_message(message: Any) -> str:
    text = str(message).strip().lower()
    return re.sub(r"\s+", " ", text)


def canonical_failure_fingerprint(
    model: str,
    hyperparameters: Mapping[str, Any] | None,
    preprocessing: Any,
    error_type: str,
    error_message: Any,
) -> str:
    from utils.configuration_identity import canonical_configuration_signature

    signature = canonical_configuration_signature(
        model, dict(hyperparameters or {}), preprocessing
    )
    return "|".join(
        (signature, str(error_type), normalized_error_message(error_message))
    )


def is_non_retryable_preprocessing_error(error: BaseException | str) -> bool:
    if isinstance(error, FrameworkConfigurationError):
        return True
    message = normalized_error_message(error)
    return any(
        marker in message
        for marker in (
            "unknown preprocessing transformation",
            "unsupported preprocessing transformation",
            "missing runtime implementation",
            "fatal configuration inconsistency",
        )
    )


__all__ = [
    "CANONICAL_PREPROCESSING_TRANSFORMATIONS",
    "FrameworkConfigurationError",
    "RUNTIME_FIT_HANDLERS",
    "TARGET_TREATMENTS",
    "TRANSFORMATION_ALIASES",
    "assert_preprocessing_registry_consistency",
    "canonical_failure_fingerprint",
    "canonicalize_transformation_name",
    "is_non_retryable_preprocessing_error",
    "normalized_error_message",
    "runtime_implemented_transformations",
    "validate_target_treatment_combination",
]
