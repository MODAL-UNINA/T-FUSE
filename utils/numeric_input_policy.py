"""Runtime contract for target-only numerical forecasting inputs."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd


TEMPORAL_COLUMNS = frozenset(
    {"date", "ds", "timestamp", "time", "datetime", "unique_id", "_is_new"}
)
CALENDAR_FEATURE_COLUMNS = frozenset(
    {
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "time_idx",
    }
)
FORBIDDEN_CANDIDATE_FIELDS = frozenset(
    {
        "d0",
        "d1",
        "d2",
        "d3",
        "d4",
        "exog",
        "exogenous_features",
        "external_features",
        "regressors",
        "regressor_columns",
        "covariates",
        "past_covariates",
        "future_covariates",
        "known_covariates",
        "feature_columns",
        "columns",
        "input_columns",
        "input_features",
        "numeric_features",
    }
)


class ExternalNumericalInputError(ValueError):
    """Raised when external numerical information enters model execution."""


def is_allowed_derived_feature(column: Any) -> bool:
    """Return whether a model feature is target- or timestamp-derived."""
    name = str(column).strip().lower()
    if re.fullmatch(r"lag_[1-9]\d*", name) or name in CALENDAR_FEATURE_COLUMNS:
        return True
    rolling = re.fullmatch(r"rolling_(mean|std|min|max)_([1-9]\d*)", name)
    return bool(rolling and int(rolling.group(2)) >= 2)


def validated_model_feature_columns(
    frame: pd.DataFrame,
    *,
    context: str,
) -> list[str]:
    """Return allowed numeric predictors and reject every external one."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} requires a pandas DataFrame.")
    numeric_predictors = [
        str(column)
        for column in frame.columns
        if str(column).strip().lower() not in TEMPORAL_COLUMNS | {"target"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    forbidden = [
        column
        for column in numeric_predictors
        if not is_allowed_derived_feature(column)
    ]
    if forbidden:
        raise ExternalNumericalInputError(
            f"{context} received forbidden external numerical columns: "
            f"{forbidden}. Only lag, rolling, and timestamp-derived calendar "
            "features are allowed."
        )
    return numeric_predictors


def assert_raw_series_contract(frame: pd.DataFrame, *, context: str) -> None:
    """Reject numeric raw-series columns other than the target."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} requires a pandas DataFrame.")
    if "target" not in frame.columns:
        raise ValueError(f"{context} requires a 'target' column.")
    forbidden = [
        str(column)
        for column in frame.columns
        if str(column).strip().lower() not in TEMPORAL_COLUMNS | {"target"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if forbidden:
        raise ExternalNumericalInputError(
            f"{context} received forbidden raw numerical columns: {forbidden}. "
            "Raw model partitions may contain only date and target."
        )


def validate_candidate_numeric_contract(candidate: Mapping[str, Any]) -> None:
    """Reject declarations that could route external values into a candidate."""
    violations: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).strip().lower()
                key_path = f"{path}.{raw_key}" if path else str(raw_key)
                if (
                    key in FORBIDDEN_CANDIDATE_FIELDS
                    or "covariate" in key
                    or "regressor" in key
                    or "exogenous" in key
                ):
                    violations.append(key_path)
                visit(item, key_path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            token = value.strip().lower()
            if re.fullmatch(r"d[0-4]", token) or token in {
                "exog",
                "exogenous_features",
                "regressors",
                "covariates",
            }:
                violations.append(path)

    visit(candidate, "candidate")
    if violations:
        raise ExternalNumericalInputError(
            "Candidate configurations cannot declare external numerical "
            f"regressors or Time-MMD covariates; forbidden fields: {violations}."
        )


def numeric_input_audit(
    feature_columns: list[str] | tuple[str, ...],
    *,
    textual_evidence_used: bool,
) -> dict[str, Any]:
    """Build the public numerical-input provenance record."""
    features = [str(column) for column in feature_columns]
    forbidden = [column for column in features if not is_allowed_derived_feature(column)]
    if forbidden:
        raise ExternalNumericalInputError(
            f"Final model audit contains forbidden feature columns: {forbidden}."
        )
    calendar = [column for column in features if column in CALENDAR_FEATURE_COLUMNS]
    return {
        "external_covariates_used": False,
        "external_covariate_policy": "excluded",
        "allowed_numeric_inputs": [
            "target_history",
            "target_derived_features",
            "timestamp_derived_calendar_features",
        ],
        "textual_evidence_used": bool(textual_evidence_used),
        "final_model_feature_columns": features,
        "calendar_feature_columns": calendar,
        "calendar_feature_source": (
            "timestamp_only" if calendar else "not_used"
        ),
    }


__all__ = [
    "CALENDAR_FEATURE_COLUMNS",
    "ExternalNumericalInputError",
    "FORBIDDEN_CANDIDATE_FIELDS",
    "TEMPORAL_COLUMNS",
    "assert_raw_series_contract",
    "is_allowed_derived_feature",
    "numeric_input_audit",
    "validate_candidate_numeric_contract",
    "validated_model_feature_columns",
]
