"""Canonical preprocessing agent compatible with TechnicalAgent preprocessing contracts.

This agent executes the ordered preprocessing pipeline selected by the TechnicalAgent.

"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from utils.replace_outliers import replace_outliers
from utils.numeric_input_policy import (
    assert_raw_series_contract,
    validated_model_feature_columns,
)
from utils.preprocessing_registry import (
    CANONICAL_PREPROCESSING_TRANSFORMATIONS,
    FrameworkConfigurationError,
    TRANSFORMATION_ALIASES,
    canonicalize_transformation_name,
    validate_target_treatment_combination,
)

logger = logging.getLogger(__name__)




class PreprocessingPlan(BaseModel):
    """Preprocessing plan from TechnicalAgent."""

    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    data_summary_before: Dict[str, Any] = Field(default_factory=dict)
    data_summary_after: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ad_hoc_findings: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class PreprocessingAgent:
    """Apply the canonical preprocessing pipeline selected by TechnicalAgent.

    Supported canonical transformation names:
    - standard_scaler
    - minmax_scaler
    - lag_features
    - rolling_features
    - calendar_features
    - log_transform
    - boxcox_transform
    - replace_outliers (deterministically inserted; not LLM-selectable)
    """

    llm: Any | None = None
    ALLOWED_TRANSFORMATIONS = frozenset(
        CANONICAL_PREPROCESSING_TRANSFORMATIONS
    )
    TRANSFORMATION_ALIASES = TRANSFORMATION_ALIASES

    TARGET_COLUMN = "target"
    INTERNAL_COLUMNS = {"_is_new"}

    @staticmethod
    def _summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Quick data summary for the target column."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data summary requires a pandas DataFrame.")
        if df.empty:
            return {
                "n_obs": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "has_nan": False,
                "n_missing": 0,
                "is_stationary_estimate": None,
                "n_columns": 0,
                "columns": [],
            }
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain a 'target' column.")

        target = df["target"].to_numpy(dtype=float)
        valid = target[~np.isnan(target)]
        if len(valid) == 0:
            mean = std = min_val = max_val = None
            stationary = None
        else:
            mean = float(np.mean(valid))
            std = float(np.std(valid))
            min_val = float(np.min(valid))
            max_val = float(np.max(valid))
            if len(valid) > 2:
                stationary = bool(np.std(np.diff(valid)) < np.std(valid) * 0.5)
            else:
                stationary = None

        return {
            "n_obs": int(len(df)),
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "has_nan": bool(np.isnan(target).any()),
            "n_missing": int(np.isnan(target).sum()),
            "is_stationary_estimate": stationary,
            "n_columns": int(len(df.columns)),
            "columns": [str(c) for c in df.columns],
        }

    def _canonicalize_name(self, raw_name: Any) -> str:
        return canonicalize_transformation_name(raw_name)

    def _validate_plan(self, plan: PreprocessingPlan) -> List[Dict[str, Any]]:
        if not isinstance(plan.transformations, list):
            raise ValueError("PreprocessingPlan.transformations must be a list.")

        validated: List[Dict[str, Any]] = []
        for idx, step in enumerate(plan.transformations):
            if not isinstance(step, dict):
                raise ValueError(f"Transformation at index {idx} must be a dictionary.")
            if "name" not in step:
                raise ValueError(
                    f"Transformation at index {idx} missing required field 'name'."
                )
            if "parameters" not in step:
                raise ValueError(
                    f"Transformation '{step.get('name')}' missing required field 'parameters'."
                )
            if not isinstance(step["parameters"], dict):
                raise ValueError(
                    f"Transformation '{step.get('name')}' must have dictionary parameters."
                )

            name = self._canonicalize_name(step["name"])
            params = dict(step["parameters"])
            reason = str(step.get("reason", ""))
            validation = str(step.get("validation", "passed")).lower()
            if validation in {"skipped", "failed"}:
                raise ValueError(
                    f"Transformation '{name}' has validation='{validation}'. "
                    "The preprocessing pipeline must not contain pre-failed/skipped steps."
                )
            if validation != "passed":
                raise ValueError(
                    f"Transformation '{name}' has invalid validation value '{validation}'."
                )

            self._validate_step_parameters(name, params)
            validated.append(
                {
                    "name": name,
                    "parameters": params,
                    "reason": reason,
                    "validation": "passed",
                }
            )

        outlier_positions = [
            index for index, step in enumerate(validated)
            if step["name"] == "replace_outliers"
        ]
        if len(outlier_positions) > 1:
            raise ValueError("replace_outliers must appear exactly once when present.")
        if outlier_positions and outlier_positions[0] != 0:
            raise ValueError("replace_outliers must be the first transformation.")

        validate_target_treatment_combination(
            step["name"] for step in validated
        )
        return validated

    def _validate_step_parameters(self, name: str, params: Dict[str, Any]) -> None:
        if name in {"standard_scaler", "minmax_scaler"}:
            scope = str(params.get("scope", "features")).strip().lower()
            if scope not in {"features", "target"}:
                raise ValueError(
                    f"{name}.parameters.scope must be 'features' or 'target'."
                )
            params["scope"] = scope
            return

        elif name == "lag_features":
            lags = params.get("lags")
            if not isinstance(lags, list) or not lags:
                raise ValueError(
                    "lag_features requires a non-empty parameters.lags list."
                )
            for lag in lags:
                if int(lag) <= 0:
                    raise ValueError("All lag_features lags must be positive integers.")

        elif name == "rolling_features":
            windows = params.get("windows")
            if not isinstance(windows, list) or not windows:
                raise ValueError(
                    "rolling_features requires a non-empty parameters.windows list."
                )
            for window in windows:
                if int(window) <= 1:
                    raise ValueError(
                        "All rolling_features windows must be integers greater than 1."
                    )
            functions = params.get("functions", ["mean"])
            if not isinstance(functions, list) or not functions:
                raise ValueError(
                    "rolling_features.parameters.functions must be a non-empty list when provided."
                )
            allowed = {"mean", "std", "min", "max"}
            invalid = [f for f in functions if f not in allowed]
            if invalid:
                raise ValueError(
                    f"Unsupported rolling feature functions: {invalid}. Allowed: {sorted(allowed)}"
                )

        elif name == "calendar_features":
            # No required params; datetime source is validated at execution.
            return

        elif name == "log_transform":
            method = params.get("method", "log1p")
            if method not in {"log1p", "log"}:
                raise ValueError(
                    "log_transform.parameters.method must be 'log1p' or 'log'."
                )
            if float(params.get("shift", 0.0)) != 0.0 or bool(
                params.get("allow_shift", False)
            ):
                raise ValueError("log_transform does not support target shifting.")

        elif name == "boxcox_transform":
            # Lambda is learned on train if absent.
            return

        elif name == "replace_outliers" and params:
            raise ValueError(
                "replace_outliers has fixed framework parameters and accepts no "
                "candidate parameters."
            )

    @staticmethod
    def _target_array(df: pd.DataFrame) -> np.ndarray:
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain a 'target' column.")
        return df["target"].to_numpy(dtype=float)

    @classmethod
    def _numeric_feature_columns(cls, df: pd.DataFrame) -> List[str]:
        return validated_model_feature_columns(
            df, context="PreprocessingAgent feature scaling"
        )

    @staticmethod
    def _as_int_list(values: Any, field_name: str) -> List[int]:
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field_name} must be a non-empty list.")
        result = [int(v) for v in values]
        return result

    def _fit_apply_lag_features(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        lags = self._as_int_list(params.get("lags"), "lag_features.parameters.lags")
        out = df.copy()
        target = self._target_array(out)
        history_tail = target[-max(lags) :].tolist() if len(target) else []

        for lag in lags:
            out[f"lag_{lag}"] = out["target"].shift(lag)

        feature_cols = [f"lag_{lag}" for lag in lags]
        before = len(out)
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        dropped = before - len(out)
        if out.empty:
            raise ValueError(
                f"lag_features with lags={lags} removed all rows. Dataset is too short."
            )

        log = {
            "name": "lag_features",
            "parameters": {
                "lags": lags,
                "history_tail": history_tail,
                "dropped_rows_train": int(dropped),
            },
            "reason": reason,
            "validation": "passed",
        }
        return out, log

    def _fit_apply_rolling_features(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        windows = self._as_int_list(
            params.get("windows"), "rolling_features.parameters.windows"
        )
        functions = params.get("functions", ["mean"])
        if not isinstance(functions, list) or not functions:
            raise ValueError(
                "rolling_features.parameters.functions must be a non-empty list."
            )

        out = df.copy()
        target = self._target_array(out)
        history_tail = target[-max(windows) :].tolist() if len(target) else []
        shifted = out["target"].shift(1)
        feature_cols: List[str] = []

        for window in windows:
            rolling = shifted.rolling(window=window, min_periods=window)
            for func in functions:
                if func == "mean":
                    col = f"rolling_mean_{window}"
                    out[col] = rolling.mean()
                elif func == "std":
                    col = f"rolling_std_{window}"
                    out[col] = rolling.std(ddof=0)
                elif func == "min":
                    col = f"rolling_min_{window}"
                    out[col] = rolling.min()
                elif func == "max":
                    col = f"rolling_max_{window}"
                    out[col] = rolling.max()
                else:
                    raise ValueError(f"Unsupported rolling function: {func}")
                feature_cols.append(col)

        before = len(out)
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        dropped = before - len(out)
        if out.empty:
            raise ValueError(
                f"rolling_features with windows={windows} removed all rows. Dataset is too short."
            )

        log = {
            "name": "rolling_features",
            "parameters": {
                "windows": windows,
                "functions": functions,
                "history_tail": history_tail,
                "dropped_rows_train": int(dropped),
            },
            "reason": reason,
            "validation": "passed",
        }
        return out, log

    def _extract_datetime(self, df: pd.DataFrame) -> pd.Series:
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(df.index, index=df.index)

        for col in ("date", "ds", "timestamp", "time", "datetime"):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().all():
                    return parsed

        raise ValueError(
            "calendar_features requires a DatetimeIndex or a parseable date/ds/timestamp/time/datetime column."
        )

    def _apply_calendar_features(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
        training: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, Any] | None]:
        out = df.copy()
        dt = self._extract_datetime(out)

        include_month = bool(params.get("include_month", True))
        include_quarter = bool(params.get("include_quarter", True))
        include_dayofweek = bool(params.get("include_dayofweek", False))
        include_time_idx = bool(params.get("include_time_idx", True))

        columns: List[str] = []
        if include_month:
            month = dt.dt.month.astype(float)
            out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
            out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
            columns.extend(["month_sin", "month_cos"])

        if include_quarter:
            quarter = dt.dt.quarter.astype(float)
            out["quarter_sin"] = np.sin(2.0 * np.pi * quarter / 4.0)
            out["quarter_cos"] = np.cos(2.0 * np.pi * quarter / 4.0)
            columns.extend(["quarter_sin", "quarter_cos"])

        if include_dayofweek:
            dow = dt.dt.dayofweek.astype(float)
            out["dayofweek_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            out["dayofweek_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
            columns.extend(["dayofweek_sin", "dayofweek_cos"])

        if include_time_idx:
            out["time_idx"] = np.arange(len(out), dtype=float)
            columns.append("time_idx")

        if training:
            log = {
                "name": "calendar_features",
                "parameters": {
                    "include_month": include_month,
                    "include_quarter": include_quarter,
                    "include_dayofweek": include_dayofweek,
                    "include_time_idx": include_time_idx,
                    "columns": columns,
                    "train_len": int(len(out)),
                },
                "reason": reason,
                "validation": "passed",
            }
            return out, log
        return out, None

    @staticmethod
    def _fit_standard(values: np.ndarray) -> Dict[str, Any]:
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return {"mean": mean.tolist(), "std": std.tolist()}

    @staticmethod
    def _apply_standard(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        mean = np.asarray(stats["mean"], dtype=float)
        std = np.asarray(stats["std"], dtype=float)
        return (values - mean) / std

    @staticmethod
    def _fit_minmax(values: np.ndarray) -> Dict[str, Any]:
        min_val = np.nanmin(values, axis=0)
        max_val = np.nanmax(values, axis=0)
        denom = np.where(np.abs(max_val - min_val) < 1e-12, 1.0, max_val - min_val)
        return {
            "min": min_val.tolist(),
            "max": max_val.tolist(),
            "denom": denom.tolist(),
        }

    @staticmethod
    def _apply_minmax(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        min_val = np.asarray(stats["min"], dtype=float)
        denom = np.asarray(stats["denom"], dtype=float)
        return (values - min_val) / denom

    def _fit_apply_scaler(
        self,
        df: pd.DataFrame,
        name: str,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit one explicitly scoped scaler on training data only."""
        out = df.copy()
        scope = str(params.get("scope", "features")).strip().lower()
        columns = (
            [self.TARGET_COLUMN]
            if scope == "target"
            else self._numeric_feature_columns(out)
        )
        if not columns:
            raise ValueError(
                f"{name} with scope='features' requires at least one numeric "
                "predictor column. Add target-derived lag/rolling features or "
                "timestamp-derived calendar features."
            )

        values = out[columns].to_numpy(dtype=float)

        if name == "standard_scaler":
            stats = self._fit_standard(values)
            scaled = self._apply_standard(values, stats)
        elif name == "minmax_scaler":
            stats = self._fit_minmax(values)
            scaled = self._apply_minmax(values, stats)
        else:
            raise ValueError(f"Unsupported scaler: {name}")

        out.loc[:, columns] = scaled

        log = {
            "name": name,
            "parameters": {
                **params,
                "scope": scope,
                "columns": columns,
                "stats": stats,
            },
            "reason": reason,
            "validation": "passed",
        }
        return out, log


    def _fit_apply_log_transform(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        out = df.copy()
        method = params.get("method", "log1p")
        target = self._target_array(out)
        min_val = float(np.nanmin(target))
        shift = float(params.get("shift", 0.0))
        if shift != 0.0 or bool(params.get("allow_shift", False)):
            raise ValueError("log_transform does not support target shifting.")

        if method == "log1p":
            if min_val <= -1.0:
                raise ValueError("log1p transform requires training target > -1.")
            out["target"] = np.log1p(target)
        elif method == "log":
            if min_val <= 0.0:
                raise ValueError("log transform requires positive training target.")
            out["target"] = np.log(target)
        else:
            raise ValueError("log_transform method must be 'log1p' or 'log'.")

        log = {
            "name": "log_transform",
            "parameters": {"method": method, "shift": shift},
            "reason": reason,
            "validation": "passed",
        }
        return out, log

    def _fit_apply_boxcox_transform(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        out = df.copy()
        target = self._target_array(out)
        min_val = float(np.nanmin(target))
        shift = float(params.get("shift", 0.0))
        if min_val + shift <= 0.0:
            shift = -min_val + 1.0

        from scipy import stats

        transformed, lam = stats.boxcox(target + shift)
        out["target"] = transformed
        log = {
            "name": "boxcox_transform",
            "parameters": {"lambda": float(lam), "shift": shift},
            "reason": reason,
            "validation": "passed",
        }
        return out, log

    def _fit_apply_replace_outliers(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        reason: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit the fixed robust-MAD replacement on this training history."""
        if params:
            raise ValueError("replace_outliers accepts no candidate parameters.")
        out, log = replace_outliers(df)
        if reason:
            log["reason"] = reason
        return out, log

    @staticmethod
    def specifications_for_refit(
        transformations_log: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return the winning pipeline structure without fitted parameters.

        Candidate search fits on train.  Once the winner is fixed, every
        learned preprocessing parameter must instead be fitted on the original
        train+validation observations.
        """
        specifications = copy.deepcopy(transformations_log)
        fitted_keys = {
            "lag_features": {"history_tail", "dropped_rows_train"},
            "rolling_features": {"history_tail", "dropped_rows_train"},
            "calendar_features": {"columns", "train_len"},
            "standard_scaler": {"columns", "stats"},
            "minmax_scaler": {"columns", "stats"},
            "log_transform": {"shift"},
            "boxcox_transform": {"lambda", "shift"},
            "replace_outliers": {
                "outlier_positions", "original_values", "replacement_values",
                "detection_scope", "robust_center", "mad", "robust_scale",
                "detected_candidate_count", "replaced_count",
            },
        }
        for step in specifications:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name", ""))
            params = step.get("parameters")
            if isinstance(params, dict):
                for key in fitted_keys.get(name, set()):
                    params.pop(key, None)
        return specifications

    def _apply_transformations(
        self,
        df: pd.DataFrame,
        plan: PreprocessingPlan,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Fit and apply the ordered preprocessing pipeline on training data."""
        if "target" not in df.columns:
            raise ValueError(
                "PreprocessingAgent requires train_df to contain a 'target' column."
            )
        assert_raw_series_contract(
            df, context="PreprocessingAgent raw training input"
        )

        transformations = self._validate_plan(plan)
        preprocessed = df.copy()
        transformations_log: List[Dict[str, Any]] = []

        for step in transformations:
            name = step["name"]
            params = dict(step.get("parameters", {}))
            reason = step.get("reason", "")

            if name == "lag_features":
                preprocessed, log = self._fit_apply_lag_features(
                    preprocessed, params, reason
                )
            elif name == "rolling_features":
                preprocessed, log = self._fit_apply_rolling_features(
                    preprocessed, params, reason
                )
            elif name == "calendar_features":
                preprocessed, log = self._apply_calendar_features(
                    preprocessed, params, reason, training=True
                )
            elif name in {"standard_scaler", "minmax_scaler"}:
                preprocessed, log = self._fit_apply_scaler(
                    preprocessed, name, params, reason
                )
            elif name == "log_transform":
                preprocessed, log = self._fit_apply_log_transform(
                    preprocessed, params, reason
                )
            elif name == "boxcox_transform":
                preprocessed, log = self._fit_apply_boxcox_transform(
                    preprocessed, params, reason
                )
            elif name == "replace_outliers":
                preprocessed, log = self._fit_apply_replace_outliers(
                    preprocessed, params, reason
                )
            else:
                raise FrameworkConfigurationError(
                    "Fatal configuration inconsistency: transformation "
                    f"{name!r} is declared as supported but has no runtime "
                    "implementation."
                )

            transformations_log.append(log)

        return preprocessed, transformations_log

    def _apply_lag_features_from_log(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        out = df.copy()
        lags = self._as_int_list(params.get("lags"), "lag_features.parameters.lags")
        for lag in lags:
            out[f"lag_{lag}"] = out["target"].shift(lag)
        return out

    def _apply_rolling_features_from_log(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        out = df.copy()
        windows = self._as_int_list(
            params.get("windows"), "rolling_features.parameters.windows"
        )
        functions = params.get("functions", ["mean"])
        shifted = out["target"].shift(1)
        for window in windows:
            rolling = shifted.rolling(window=window, min_periods=window)
            for func in functions:
                if func == "mean":
                    out[f"rolling_mean_{window}"] = rolling.mean()
                elif func == "std":
                    out[f"rolling_std_{window}"] = rolling.std(ddof=0)
                elif func == "min":
                    out[f"rolling_min_{window}"] = rolling.min()
                elif func == "max":
                    out[f"rolling_max_{window}"] = rolling.max()
                else:
                    raise ValueError(f"Unsupported rolling function: {func}")
        return out

    def _apply_scaler_from_log(
        self, df: pd.DataFrame, name: str, params: Dict[str, Any]
    ) -> pd.DataFrame:
        out = df.copy()
        columns = params.get("columns")
        stats = params.get("stats")
        if not isinstance(columns, list) or not columns:
            raise ValueError(
                f"Logged scaler '{name}' missing non-empty parameters.columns."
            )
        if not isinstance(stats, dict):
            raise ValueError(f"Logged scaler '{name}' missing parameters.stats.")
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise ValueError(
                f"Cannot apply scaler '{name}'; missing columns: {missing}"
            )

        values = out[columns].to_numpy(dtype=float)
        if name == "standard_scaler" or (params.get("scaler") == "standard_scaler"):
            scaled = self._apply_standard(values, stats)
        elif name == "minmax_scaler" or (params.get("scaler") == "minmax_scaler"):
            scaled = self._apply_minmax(values, stats)
        else:
            raise ValueError(f"Unsupported logged scaler: {name}")
        out.loc[:, columns] = scaled
        return out

    def _apply_log_transform_from_log(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        out = df.copy()
        method = params.get("method", "log1p")
        shift = float(params.get("shift", 0.0))
        if shift != 0.0:
            raise ValueError("Logged log_transform must have shift=0.")
        target = self._target_array(out)
        if method == "log1p":
            if np.nanmin(target) <= -1.0:
                raise ValueError(
                    "Cannot apply logged log1p transform: target <= -1."
                )
            out["target"] = np.log1p(target)
        elif method == "log":
            if np.nanmin(target) <= 0.0:
                raise ValueError(
                    "Cannot apply logged log transform: target <= 0."
                )
            out["target"] = np.log(target)
        else:
            raise ValueError(f"Unsupported logged log_transform method: {method}")
        return out

    def _apply_boxcox_transform_from_log(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        out = df.copy()
        lam = float(params.get("lambda"))
        shift = float(params.get("shift", 0.0))
        target = self._target_array(out)
        shifted = target + shift
        if np.nanmin(shifted) <= 0.0:
            raise ValueError(
                "Cannot apply logged boxcox_transform: target + shift <= 0."
            )
        if abs(lam) < 1e-12:
            out["target"] = np.log(shifted)
        else:
            out["target"] = (np.power(shifted, lam) - 1.0) / lam
        return out

    def _apply_replace_outliers_from_log(
        self, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Replay fitted replacements on history rows, never on new rows."""
        out = df.copy()
        out["target"] = pd.to_numeric(out["target"], errors="coerce").astype(float)
        positions = params.get("outlier_positions", [])
        replacements = params.get("replacement_values", [])
        if not isinstance(positions, list) or not isinstance(replacements, list):
            raise ValueError(
                "Logged replace_outliers positions and replacements must be lists."
            )
        if len(positions) != len(replacements):
            raise ValueError(
                "Logged replace_outliers positions/replacements length mismatch."
            )

        is_new = (
            out["_is_new"].to_numpy(dtype=bool)
            if "_is_new" in out.columns
            else np.zeros(len(out), dtype=bool)
        )
        target_col = out.columns.get_loc("target")
        for raw_position, raw_replacement in zip(positions, replacements):
            position = int(raw_position)
            replacement = float(raw_replacement)
            if position < 0 or position >= len(out):
                raise ValueError(
                    "Logged replace_outliers position is outside history."
                )
            if not is_new[position]:
                out.iloc[position, target_col] = replacement
        return out

    def apply_transformations_from_log(
        self,
        df: pd.DataFrame,
        transformations_log: List[Dict[str, Any]],
        start_idx: int = 0,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Apply training-fitted transformations to validation/test data.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "apply_transformations_from_log requires a pandas DataFrame."
            )
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain a 'target' column.")
        if not isinstance(transformations_log, list):
            raise ValueError("transformations_log must be a list.")
        assert_raw_series_contract(
            df, context="PreprocessingAgent raw evaluation input"
        )

        if history_df is not None:
            if not isinstance(history_df, pd.DataFrame):
                raise ValueError("history_df must be a pandas DataFrame when provided.")
            if "target" not in history_df.columns:
                raise ValueError("history_df must contain a 'target' column.")
            assert_raw_series_contract(
                history_df, context="PreprocessingAgent raw history input"
            )
            combined = pd.concat(
                [
                    history_df.copy().assign(_is_new=False),
                    df.copy().assign(_is_new=True),
                ],
                axis=0,
            )
        else:
            combined = df.copy().assign(_is_new=True)

        for step in transformations_log:
            if not isinstance(step, dict):
                raise ValueError("Each logged transformation must be a dictionary.")
            if step.get("validation") != "passed":
                raise ValueError(
                    f"Logged transformation {step.get('name')} is not passed; cannot apply pipeline."
                )
            name = self._canonicalize_name(step.get("name"))
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                raise ValueError(
                    f"Logged transformation '{name}' has invalid parameters."
                )

            if name == "lag_features":
                combined = self._apply_lag_features_from_log(combined, params)
            elif name == "rolling_features":
                combined = self._apply_rolling_features_from_log(combined, params)
            elif name == "calendar_features":
                combined, _ = self._apply_calendar_features(
                    combined, params, "", training=False
                )
            elif name in {"standard_scaler", "minmax_scaler"}:
                combined = self._apply_scaler_from_log(combined, name, params)
            elif name == "log_transform":
                combined = self._apply_log_transform_from_log(combined, params)
            elif name == "boxcox_transform":
                combined = self._apply_boxcox_transform_from_log(combined, params)
            elif name == "replace_outliers":
                combined = self._apply_replace_outliers_from_log(
                    combined, params
                )
            elif name == "inverse_transform_before_metrics":
                # Execution marker only. Actual inverse happens in inverse_transform().
                continue
            else:
                raise FrameworkConfigurationError(
                    "Fatal configuration inconsistency: transformation "
                    f"{name!r} is declared as supported but has no replay "
                    "implementation."
                )

        result = combined[combined["_is_new"]].copy()
        result = result.drop(columns=["_is_new"])

        feature_cols = validated_model_feature_columns(
            result, context="PreprocessingAgent transformed evaluation output"
        )
        if feature_cols and result[feature_cols].isna().any().any():
            bad_cols = (
                result[feature_cols].columns[result[feature_cols].isna().any()].tolist()
            )
            raise ValueError(
                "Validation/test preprocessing produced NaN feature values. "
                f"Likely insufficient history for lag/rolling features. Columns: {bad_cols}"
            )

        return result.reset_index(drop=True)

    def inverse_transform(
        self,
        predictions: np.ndarray,
        transformations_log: List[Dict[str, Any]],
        original_train_target: np.ndarray,
    ) -> np.ndarray:
        """Inverse target transformations for predictions.
        """
        result = np.asarray(predictions, dtype=float).copy()

        if not isinstance(transformations_log, list):
            raise ValueError("transformations_log must be a list.")

        for step in reversed(transformations_log):
            if not isinstance(step, dict):
                raise ValueError("Each logged transformation must be a dictionary.")
            if step.get("validation") != "passed":
                raise ValueError(
                    f"Logged transformation {step.get('name')} is not passed; cannot inverse-transform."
                )

            name = self._canonicalize_name(step.get("name"))
            params = step.get("parameters", {})
            if not isinstance(params, dict):
                raise ValueError(
                    f"Logged transformation '{name}' has invalid parameters."
                )

            if name in {
                "lag_features",
                "rolling_features",
                "calendar_features",
                "replace_outliers",
            }:
                continue

            if name in {"standard_scaler", "minmax_scaler"}:
                if str(params.get("scope", "features")) != "target":
                    continue
                stats = params.get("stats")
                if not isinstance(stats, dict):
                    raise ValueError(f"Logged target scaler '{name}' has no fitted stats.")
                if name == "standard_scaler":
                    mean = float(np.asarray(stats["mean"], dtype=float).reshape(-1)[0])
                    std = float(np.asarray(stats["std"], dtype=float).reshape(-1)[0])
                    result = result * std + mean
                else:
                    minimum = float(np.asarray(stats["min"], dtype=float).reshape(-1)[0])
                    denom = float(np.asarray(stats["denom"], dtype=float).reshape(-1)[0])
                    result = result * denom + minimum
                continue

            if name == "log_transform":
                method = params.get("method", "log1p")
                shift = float(params.get("shift", 0.0))
                if shift != 0.0:
                    raise ValueError("Logged log_transform must have shift=0.")
                if method == "log1p":
                    result = np.expm1(result)
                elif method == "log":
                    result = np.exp(result)
                else:
                    raise ValueError(f"Unsupported log_transform method: {method}")

            elif name == "boxcox_transform":
                lam = float(params.get("lambda"))
                shift = float(params.get("shift", 0.0))
                if abs(lam) < 1e-12:
                    result = np.exp(result) - shift
                else:
                    base = lam * result + 1.0
                    result = np.power(np.clip(base, 1e-12, None), 1.0 / lam) - shift

            else:
                raise ValueError(
                    f"Unsupported logged transformation for inverse_transform: {name}"
                )

        return result

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing selected by TechnicalAgent."""
        train_df = state.get("train_df")
        if not isinstance(train_df, pd.DataFrame):
            raise ValueError("state['train_df'] must be a pandas DataFrame")
        if train_df.empty:
            raise ValueError("state['train_df'] must not be empty")
        if "target" not in train_df.columns:
            raise ValueError("state['train_df'] must contain a 'target' column")

        step = int(state.get("step", 0))
        candidate = state.get("current_candidate_config")
        if not isinstance(candidate, dict):
            raise ValueError("state['current_candidate_config'] must be a dictionary")

        preprocessing_config = candidate.get("preprocessing")
        if not isinstance(preprocessing_config, dict):
            raise ValueError(
                "current_candidate_config.preprocessing must be a dictionary"
            )
        if "transformations" not in preprocessing_config:
            raise ValueError(
                "current_candidate_config.preprocessing missing required field 'transformations'"
            )
        transformations = preprocessing_config.get("transformations")
        if not isinstance(transformations, list):
            raise ValueError(
                "current_candidate_config.preprocessing.transformations must be a list"
            )

        logger.info(
            "PreprocessingAgent: applying TechnicalAgent preprocessing pipeline..."
        )
        data_before = self._summarize_data(train_df)
        raw_train_df = state.get("raw_train_df", train_df)
        if not isinstance(raw_train_df, pd.DataFrame):
            raise ValueError("state[raw_train_df] must be a pandas DataFrame")
        raw_train_snapshot = raw_train_df.copy(deep=True)
        plan = PreprocessingPlan(
            transformations=transformations,
            notes="Extracted from TechnicalAgent candidate configuration.",
            confidence=1.0,
        )

        preprocessed_df, transformations_log = self._apply_transformations(
            raw_train_snapshot, plan
        )
        if len(preprocessed_df) > len(raw_train_snapshot):
            raise RuntimeError("Preprocessing unexpectedly increased row count")
        if not raw_train_df.equals(raw_train_snapshot):
            raise RuntimeError("Immutable raw_train_df was modified")
        outlier_log = next(
            (
                copy.deepcopy(item)
                for item in transformations_log
                if item.get("name") == "replace_outliers"
            ),
            {},
        )
        data_after = self._summarize_data(preprocessed_df)

        result = {
            "model": candidate.get("model_type", None),
            "data_summary_before": data_before,
            "data_summary_after": data_after,
            "transformations": transformations_log,
            "notes": plan.notes,
            "confidence": plan.confidence,
            "n_transformations": len(transformations_log),
            "replace_outliers": outlier_log,
            "ad_hoc_findings": plan.ad_hoc_findings,
        }

        state["processed_train_df"] = preprocessed_df
        state["replace_outliers_log"] = outlier_log
        state["preprocessed_train_df"] = preprocessed_df
        state["preprocessing_log"] = result

        val_df = state.get("val_df")
        raw_validation_df = state.get("raw_validation_df", val_df)
        if isinstance(val_df, pd.DataFrame) and isinstance(raw_validation_df, pd.DataFrame):
            if not raw_validation_df["target"].equals(val_df["target"]):
                raise RuntimeError("Validation ground truth was modified before preprocessing")
        if isinstance(val_df, pd.DataFrame) and not val_df.empty:
            state["preprocessed_val_df"] = self.apply_transformations_from_log(
                val_df,
                transformations_log,
                start_idx=len(train_df),
                history_df=raw_train_snapshot,
            )

        if plan.ad_hoc_findings:
            state.setdefault("analysis_history", []).append(
                {
                    "agent": "preprocessing",
                    "task": state.get("planner_guidance", "unknown task"),
                    "result": plan.ad_hoc_findings,
                }
            )

        logger.info(
            "PreprocessingAgent: applied %d transformations (confidence: %.2f)",
            len(transformations_log),
            plan.confidence,
        )
        return state
