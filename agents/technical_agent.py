from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from models.defaults import (
    DEFAULT_HYPERPARAMETERS,
    HYPERPARAMETER_CONSTRAINTS,
    MODEL_HYPERPARAMETER_CONTRACT,
)
from models.trainer import Trainer
from utils.json_logger import _to_json_safe
from llm.prompts import (
    TECHNICAL_AGENT_SYSTEM_PROMPT,
    TECHNICAL_AGENT_USER_PROMPT,
)
from utils.model_family_state import MODEL_TO_FAMILY, model_to_family
from utils.numeric_input_policy import validate_candidate_numeric_contract
from utils.configuration_identity import (
    canonical_configuration_signature,
    normalize_candidate_hyperparameters,
)
from utils.preprocessing_registry import (
    CANONICAL_PREPROCESSING_TRANSFORMATIONS,
    LLM_SELECTABLE_PREPROCESSING_TRANSFORMATIONS,
    TRANSFORMATION_ALIASES,
    canonical_failure_fingerprint,
    canonicalize_transformation_name,
    validate_target_treatment_combination,
)


import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalAgent:
    trainer: Trainer
    llm: Any | None = None
    finalize_mase_threshold: float = 0.85

    MODELS = tuple(MODEL_TO_FAMILY)

    # Canonical preprocessing names expected by the prompt and validators.
    ALLOWED_TRANSFORMATIONS = frozenset(
        CANONICAL_PREPROCESSING_TRANSFORMATIONS
    )
    TRANSFORMATION_ALIASES = TRANSFORMATION_ALIASES

    SCALERS = {"standard_scaler", "minmax_scaler"}
    FEATURE_GENERATORS = {"lag_features", "rolling_features", "calendar_features"}
    TRANSFORMATION_PARAMETER_CONTRACT = {
        "standard_scaler": {"scope"},
        "minmax_scaler": {"scope"},
        "lag_features": {"lags", "_auto_completed", "_auto_completion_reason"},
        "rolling_features": {
            "windows", "functions", "_auto_completed", "_auto_completion_reason"
        },
        "calendar_features": {
            "include_month", "include_quarter", "include_dayofweek",
            "include_time_idx",
        },
        "log_transform": {"method", "shift", "allow_shift"},
        "boxcox_transform": set(),
        "replace_outliers": set(),
    }


    FEATURE_SCALING_REQUIRED_MODELS = {
        "svr",
        "knn",
        "elasticnet",
        "mlp",
    }
    TARGET_SCALING_REQUIRED_MODELS = {
        "lstm", "n-beats", "dlinear", "reformer",
        "autoformer", "informer", "fedformer", "n-hits", "patchtst",
        "timesnet", "tide",
    }

    EXTRA_ALLOWED_HYPERPARAMETERS = {
        "SVR": {
            "C",
            "epsilon",
            "kernel",
            "gamma",
            "degree",
            "coef0",
            "shrinking",
            "tol",
            "cache_size",
            "max_iter",
        },
        "KNN": {
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size",
            "p",
            "metric",
        },
        "ElasticNet": {
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "max_iter",
            "tol",
            "selection",
            "positive",
        },
    }

    
    def _build_system_prompt(self, state: Dict[str, Any]) -> str:
        return (
            TECHNICAL_AGENT_SYSTEM_PROMPT.replace(
                "{supported_preprocessing_transformations}",
                ", ".join(
                    f"`{name}`"
                    for name in LLM_SELECTABLE_PREPROCESSING_TRANSFORMATIONS
                ),
            )
        )


    @staticmethod
    def _build_target_value_constraints(train_df: Any) -> Dict[str, Any]:
        """Derive prompt-level model constraints from the observed training target."""
        if not isinstance(train_df, pd.DataFrame) or "target" not in train_df.columns:
            return {"status": "unavailable"}

        target = pd.to_numeric(train_df["target"], errors="coerce")
        target = target[np.isfinite(target)]
        if target.empty:
            return {"status": "unavailable"}

        minimum = float(target.min())
        maximum = float(target.max())
        strictly_positive = bool((target > 0).all())
        constraints: Dict[str, Any] = {
            "status": "ready",
            "minimum": minimum,
            "maximum": maximum,
            "contains_negative": bool((target < 0).any()),
            "contains_zero": bool((target == 0).any()),
            "strictly_positive": strictly_positive,
            "log1p_supported": bool(minimum > -1.0),
            "log_transform_supported": bool(minimum > -1.0),
        }
        if not strictly_positive:
            constraints["ETS"] = {
                "multiplicative_components_allowed": False,
                "allowed_trend": ["add", None],
                "allowed_seasonal": ["add", None],
                "instruction": (
                    "Do not select trend='mul' or seasonal='mul'. "
                    "Multiplicative ETS components require a strictly positive target."
                ),
            }
        return constraints

    @staticmethod
    def _candidate_request_contract(
        state: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str | None, str | None]:
        """Resolve the single executable action/model pair.

        CandidateRequest is mandatory and is authored only by PlannerAgent.
        """
        request = state.get("candidate_request", {})
        if not isinstance(request, dict) or not request.get("model"):
            raise ValueError(
                "TechnicalAgent requires a Planner candidate_request"
            )
        action = str(request.get("action", "")).strip().lower()
        if action not in {"explore", "refine", "retry"}:
            raise ValueError(
                "CandidateRequest.action must be explore, refine or retry"
            )
        return request, action, str(request["model"])



    def _build_user_prompt(self, state: Dict[str, Any]) -> str:
        """Build comprehensive user prompt with diagnostic and memory context."""
        planner_signals = state.get("planner_signals", {})
        (
            candidate_request,
            candidate_request_action,
            candidate_request_model,
        ) = self._candidate_request_contract(state)
        planner_strategy = (
            candidate_request_action
            or planner_signals.get("technical_mode", "explore")
        )

        train_df = state.get("train_df")
        dataset_length = len(train_df) if isinstance(train_df, pd.DataFrame) else 0
        target_value_constraints = self._build_target_value_constraints(train_df)


        # Extract best model info.
        best_model_data = state.get("best_model") or {}
        current_best_model = best_model_data.get("selected_model", "None")

        current_best_hyperparameters = json.dumps(
            _to_json_safe(best_model_data.get("hyperparameters", {})), ensure_ascii=True
        )
        current_best_score = best_model_data.get("mase", "None")

        best_prep = self._extract_best_transformations(best_model_data)
        current_best_preprocessing = json.dumps(
            _to_json_safe(best_prep), ensure_ascii=True
        )

        # Extract last trained model info.
        perf_hist = state.get("performance_history", [])
        requested_refinement_target = (
            candidate_request_model
            if candidate_request_action == "refine"
            else planner_signals.get("refinement_target_model")
            if not candidate_request
            else None
        )
        refinement_target_model = requested_refinement_target or current_best_model
        target_attempts = [
            item for item in perf_hist
            if isinstance(item, dict) and item.get("model") == refinement_target_model
        ]
        refinement_target_data = min(
            target_attempts,
            key=lambda item: float(item.get("mase", float("inf"))),
            default=best_model_data if refinement_target_model == current_best_model else {},
        )
        refinement_target_hyperparameters = json.dumps(
            _to_json_safe(refinement_target_data.get("hyperparameters", {})),
            ensure_ascii=True,
        )
        refinement_target_preprocessing = json.dumps(
            _to_json_safe(self._extract_attempt_transformations(refinement_target_data)),
            ensure_ascii=True,
        )
        refinement_target_score = refinement_target_data.get("mase", "None")
        refinement_target_block = ""
        if planner_strategy == "refine":
            refinement_target_family = (
                planner_signals.get("refinement_target_family")
                or model_to_family(refinement_target_model)
                or "None"
            )
            refinement_target_block = (
                '"refinement_target":\n'
                f"- model: {refinement_target_model}\n"
                f"- family: {refinement_target_family}\n"
                f"- source: {planner_signals.get('refinement_target_source', 'global_best')}\n"
                f"- best_tested_hyperparameters: {refinement_target_hyperparameters}\n"
                f"- best_tested_preprocessing: {refinement_target_preprocessing}\n"
                f"- best_observed_mase: {refinement_target_score}\n"
            )
        if perf_hist:
            last_attempt = perf_hist[-1]
            last_model_type = last_attempt.get("model", "None")
            last_hyperparameters = json.dumps(
                _to_json_safe(last_attempt.get("hyperparameters", {})),
                ensure_ascii=True,
            )
            last_score = last_attempt.get("mase", "None")
            last_prep = self._extract_attempt_transformations(last_attempt)
            last_preprocessing = json.dumps(_to_json_safe(last_prep), ensure_ascii=True)
        else:
            last_model_type = "None"
            last_hyperparameters = "{}"
            last_score = "None"
            last_preprocessing = "[]"

        # Block every configuration that has already been completed, failed,
        # rejected, duplicated, or explicitly stored in memory.  The TechnicalAgent
        # must never regenerate one of these configurations on retry.
        blocked_configuration_signatures: List[str] = []
        blocked_configuration_signatures_set: set[str] = set()

        def _add_blocked_signature(signature: Any) -> None:
            if not signature:
                return
            value = str(signature)
            if value not in blocked_configuration_signatures_set:
                blocked_configuration_signatures_set.add(value)
                blocked_configuration_signatures.append(value)

        for attempt in perf_hist:
            if not isinstance(attempt, dict):
                continue
            _add_blocked_signature(
                attempt.get("configuration_signature")
                or canonical_configuration_signature(
                    str(attempt.get("model", "")),
                    attempt.get("hyperparameters", {}),
                    self._extract_attempt_transformations(attempt),
                )
            )

        for record in state.get("candidate_ledger", []) or []:
            if not isinstance(record, dict):
                continue
            if record.get("status") not in {"completed", "failed", "rejected", "duplicate", "pending"}:
                continue
            _add_blocked_signature(
                record.get("configuration_signature")
                or canonical_configuration_signature(
                    str(record.get("model", "")),
                    record.get("hyperparameters", {}),
                    record.get("preprocessing_signature", []),
                )
            )

        for record in state.get("rejected_technical_candidates", []) or []:
            if not isinstance(record, dict):
                continue
            _add_blocked_signature(
                record.get("configuration_signature")
                or canonical_configuration_signature(
                    str(record.get("model_type", "")),
                    record.get("hyperparameters", {}),
                    record.get("preprocessing", []),
                )
            )

        memory = state.get("memory", {})
        if not isinstance(memory, dict):
            memory = {}
        failed_combinations: List[Dict[str, Any]] = list(
            memory.get("failed_combinations", []) or []
        )
        for record in failed_combinations:
            if not isinstance(record, dict):
                continue
            _add_blocked_signature(
                record.get("configuration_signature")
                or canonical_configuration_signature(
                    str(record.get("model", "")),
                    record.get("hyperparameters", {}),
                    record.get("preprocessing", []),
                )
            )

        blocked_configuration_signatures_json = json.dumps(
            blocked_configuration_signatures,
            ensure_ascii=True,
            indent=2,
        )
        data_summary = state.get("data_summary", {})
        base_metrics = data_summary.get("base_metrics", {})

        seasonality_detected = bool(
            data_summary.get(
                "seasonality_detected",
                base_metrics.get("seasonality_detected", False),
            )
        )
        seasonality_period = data_summary.get("seasonal_periods", 0)
        frequency = base_metrics.get("inferred_freq", "unknown")
        suggested_lag = data_summary.get("suggested_lag", 0)



        failed_combination = failed_combinations[-1] if failed_combinations else None

        catalog = state.get("model_catalog", {})
        base_models = (
            catalog.get("available_models", [])
            if isinstance(catalog, dict) else []
        )
        available_list = list(base_models)

        if not available_list:
            raise ValueError(
                "No AVAILABLE models remain after applying memory constraints."
            )

        available_models = ", ".join(available_list)

        technical_retry_count = int(state.get("technical_retry_count", 0))
        last_technical_error = state.get("last_technical_error")

        is_technical_retry = (
            bool(state.get("technical_error", False)) or technical_retry_count > 0
        )

        if is_technical_retry and last_technical_error:
            last_failed_combination = state.get("memory", {}).get("last_failed_candidate") or failed_combination or {}

            if planner_strategy == "refine":
                fallback_policy = (
                    "[RETRY POLICY]\n"
                    f"You are in refine mode. You must keep target model '{refinement_target_model}'.\n"
                    "Try a different valid unblocked configuration for the refinement target.\n"
                    "If no valid unblocked configuration exists for the refinement target, "
                    "return NO_VALID_NEW_CONFIGURATION. Do not switch models in refine mode.\n\n"
                )
            else:
                fallback_policy = (
                    "[RETRY POLICY]\n"
                    "Keep the Planner-requested model and try a different valid unblocked configuration.\n"
                    "If the requested model has no valid unblocked configuration left, return NO_VALID_NEW_CONFIGURATION. "
                    "Do not substitute another model: model switching belongs to the Planner.\n\n"
                )

            
            retry_block = (
                "\n\n[MANDATORY TECHNICAL RETRY]\n"
                f"This is technical retry #{technical_retry_count} using the same TechnicalAgent node.\n"
                f"Previous technical error: {last_technical_error}\n"
                "The previous candidate was invalid and did not reach preprocessing or training.\n\n"
                "[GOAL]\n"
                "Output either:\n"
                "1. one NEW runtime-valid forecasting configuration, or\n"
                "2. NO_VALID_NEW_CONFIGURATION if no valid new configuration exists under the current constraints.\n\n"
                "[NOVELTY RULE]\n"
                "A configuration is NEW only if it is not present in tested_combinations.\n"
                f"You MUST not repeat this failed configuration:\n{last_failed_combination}\n"
                "You must not repeat any tested, failed, invalid, current_best, or last_trained configuration.\n"
                "Changing only rationale, JSON key order, whitespace, comments, or equivalent empty preprocessing does not count as a new configuration.\n\n"
                + fallback_policy + 
                "[ESCALATION OUTPUT]\n"
                "When no valid new configuration exists, output exactly this JSON shape:\n"
                "{\n"
                '  "status": "NO_VALID_NEW_CONFIGURATION",\n'
                '  "reason": "No runtime-valid unblocked configuration exists under the current constraints.",\n'
                '  "request_planner_replan": true,\n'
                '  "exhausted_scope": "same_model",\n'
                '  "blocked_by": ["tested_configuration", "failed_configuration", "model_contract", "preprocessing_contract"]\n'
                "}\n\n"
                "[ESCALATION FIELD RULES]\n"
                'The field "exhausted_scope" must be exactly one of: "same_model", "same_family", "all_available_models".\n'
                'The field "blocked_by" may include only relevant reasons from: '
                '"tested_configuration", "failed_configuration", "invalid_configuration", '
                '"model_contract", "preprocessing_contract", "planner_constraints".\n\n'
                "[PRIORITY ORDER]\n"
                "1. Runtime validity.\n"
                "2. Novel combination.\n"
                "3. Model and preprocessing contract compliance.\n"
                "4. Planner guidance.\n"
                "5. Expected performance.\n\n"
                "If novelty and validity cannot both be satisfied, return NO_VALID_NEW_CONFIGURATION.\n"
            ) 
        else:
            retry_block = ""


        if planner_strategy == "refine":
            constraint_block = (
                f"REFINE MODE: you MUST keep target model '{refinement_target_model}', but you may change "
                "hyperparameters and/or preprocessing.\n"
                "You MUST NOT repeat failed configurations.\n" + retry_block
            )
        else:
            constraint_block = (
                "EXPLORE MODE: choose a high-value configuration from an untested or under-tested "
                "plausible family when possible. Do not repeat already proposed configurations.\n"
                + retry_block
            )
        if candidate_request_model:
            constraint_block += (
                "\n[DETERMINISTIC CANDIDATE REQUEST]\n"
                f"request_id={candidate_request.get('request_id')}\n"
                f"You MUST configure model_type='{candidate_request_model}'. "
                "This is the Planner-selected trial; do not substitute another model.\n"
                f"Structured request={json.dumps(_to_json_safe(candidate_request), ensure_ascii=True)}\n"
            )
            required_target_transform = candidate_request.get(
                "required_target_transform"
            )
            if required_target_transform:
                constraint_block += (
                    "This trial MUST use target_transform="
                    f"{required_target_transform!r}; this is one targeted "
                    "exploration trial, not a global preprocessing rule.\n"
                )
            if (
                candidate_request_model.lower()
                in self.TARGET_SCALING_REQUIRED_MODELS
            ):
                constraint_block += (
                    "\n[TARGET-SEQUENCE PREPROCESSING CONTRACT]\n"
                    f"{candidate_request_model} consumes only the target sequence.\n"
                    "Include exactly one standard_scaler or minmax_scaler with "
                    "parameters.scope='target'.\n"
                    "Do not use scope='features'.\n"
                    "Do not add lag_features, rolling_features, or "
                    "calendar_features: this model does not consume tabular "
                    "predictor columns.\n"
                )

        allowed_lag_feature_values = self._default_lags(state)
        constraint_block += (
            "\n[DETERMINISTIC LAG-FEATURE CONTRACT]\n"
            "lag_features.parameters.lags may contain only values from "
            f"{allowed_lag_feature_values}.\n"
            "A suggested/model lag such as 12 is one lag value; NEVER expand "
            "it into the full range [1, 2, ..., 12]. Use a non-empty subset "
            "of the allowed values exactly as listed.\n"
        )

        successful = memory.get("successful_patterns", [])
        regime = state.get("context_regime", "normal")
        best_success = None
        if isinstance(successful, list):
            best_success = next(
                (p for p in successful if p.get("regime") == regime), None
            )
        best_success_str = (
            json.dumps(_to_json_safe(best_success), ensure_ascii=True)
            if best_success
            else "None"
        )

        recent_winners = (
            json.dumps(_to_json_safe(successful[:3]), ensure_ascii=True)
            if isinstance(successful, list)
            else "None"
        )

        inferred_domain = "not_exposed"

        model_hyperparameter_contract = json.dumps(
            _to_json_safe(MODEL_HYPERPARAMETER_CONTRACT),
            ensure_ascii=True,
        )


        from core.candidate_catalog import DEFAULT_CANDIDATE_CATALOG

        catalog_constraints = DEFAULT_CANDIDATE_CATALOG.constraints_dict()
        catalog_suggestions = DEFAULT_CANDIDATE_CATALOG.suggestions_dict()
        if candidate_request_model and candidate_request_model in catalog_constraints:
            prompt_catalog_constraints = {
                candidate_request_model: catalog_constraints[candidate_request_model]
            }
            prompt_catalog_suggestions = {
                candidate_request_model: catalog_suggestions.get(candidate_request_model, {})
            }
        else:
            prompt_catalog_constraints = {
                model: catalog_constraints[model]
                for model in available_list
                if model in catalog_constraints
            }
            prompt_catalog_suggestions = {
                model: catalog_suggestions.get(model, {})
                for model in available_list
                if model in catalog_constraints
            }
        candidate_catalog_constraints = json.dumps(
            _to_json_safe(prompt_catalog_constraints),
            ensure_ascii=True,
        )
        candidate_catalog_suggestions = json.dumps(
            _to_json_safe(prompt_catalog_suggestions),
            ensure_ascii=True,
        )

        step = state.get("step", 0)
        max_iterations = state.get("max_iterations", 100)
        remaining_iterations = max(0, max_iterations - step)

        analysis_history = state.get("analysis_history", [])

        if not analysis_history:
            analysis_history_summary = "None yet."
        else:
            lines = []
            for item in analysis_history[-3:]:
                lines.append(
                    f"- Task: {item.get('task', 'unknown')} | Result: {json.dumps(item.get('result', {}))}"
                )
            analysis_history_summary = "\n".join(lines)

        from utils.model_family_state import CANONICAL_FAMILY_ORDER, MODEL_TO_FAMILY
        family_model_map = {
            family: [model for model, mapped in MODEL_TO_FAMILY.items() if mapped == family]
            for family in CANONICAL_FAMILY_ORDER
            if any(mapped == family for mapped in MODEL_TO_FAMILY.values())
        }

        user_prompt = TECHNICAL_AGENT_USER_PROMPT.format(
            constraint_block=constraint_block,
            available_models=available_models,
            planner_strategy=planner_strategy,
            family_model_map=json.dumps(family_model_map, ensure_ascii=True, indent=2),
            step=step,
            max_iterations=max_iterations,
            remaining_iterations=remaining_iterations,
            dataset_length=dataset_length,
            seasonality_detected=seasonality_detected,
            seasonality_period=seasonality_period,
            frequency=frequency,
            suggested_lag=suggested_lag,
            target_value_constraints=json.dumps(
                _to_json_safe(target_value_constraints), ensure_ascii=True
            ),
            forecast_horizon=state.get("forecast_horizon", 10),
            analysis_history_summary=analysis_history_summary,
            best_success=best_success_str,
            inferred_domain=inferred_domain,
            recent_winners=recent_winners,
            current_best_model=current_best_model,
            current_best_hyperparameters=current_best_hyperparameters,
            current_best_preprocessing=current_best_preprocessing,
            current_best_score=current_best_score,
            refinement_target_block=refinement_target_block,
            last_model_type=last_model_type,
            last_hyperparameters=last_hyperparameters,
            last_preprocessing=last_preprocessing,
            last_score=last_score,
            planner_technical_guidance=(
                str(state.get("planner_guidance"))
                if state.get("planner_guidance")
                else "None (no specific guidance)."
            ),
            model_hyperparameter_contract=model_hyperparameter_contract,
            candidate_catalog_constraints=candidate_catalog_constraints,
            candidate_catalog_suggestions=candidate_catalog_suggestions,
            blocked_configuration_signatures=blocked_configuration_signatures_json,
            supported_preprocessing_transformations=json.dumps(
                list(LLM_SELECTABLE_PREPROCESSING_TRANSFORMATIONS),
                ensure_ascii=True,
            ),
        )

        return user_prompt

    def _canonicalize_transformation_name(self, raw_name: Any) -> str:
        return canonicalize_transformation_name(raw_name)

    @staticmethod
    def _coerce_positive_int_list(
        value: Any, field_name: str, *, minimum: int = 1
    ) -> List[int]:
        """Coerce a scalar/list-like value into a sorted list of unique positive ints."""
        if value is None:
            return []
        raw_values = value if isinstance(value, list) else [value]
        out: List[int] = []
        for raw in raw_values:
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{field_name} must contain integers. Invalid value: {raw!r}"
                )
            if parsed < minimum:
                raise ValueError(
                    f"{field_name} values must be >= {minimum}. Invalid value: {parsed}"
                )
            if parsed not in out:
                out.append(parsed)
        return out

    @staticmethod
    def _state_int(state: Dict[str, Any], *paths: str) -> int | None:
        """Read an int from common state locations using dotted paths."""
        for path in paths:
            cur: Any = state
            ok = True
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if not ok:
                continue
            try:
                if cur is not None:
                    return int(cur)
            except (TypeError, ValueError):
                continue
        return None

    def _default_lags(self, state: Dict[str, Any]) -> List[int]:
        """Deterministic, explicit completion for missing lag_features.lags.

        This is not a silent fallback: the caller records _auto_completed metadata
        in the candidate parameters. It prevents repeated LLM retries caused by
        incomplete but otherwise clear feature-generation intent.
        """
        dataset_length = 0
        train_df = state.get("train_df")
        if isinstance(train_df, pd.DataFrame):
            dataset_length = len(train_df)

        suggested_lag = self._state_int(state, "data_summary.suggested_lag")
        seasonal_period = self._state_int(
            state,
            "data_summary.seasonal_periods",
        )
        horizon = self._state_int(state, "forecast_horizon") or 1

        candidates = [1]
        if horizon > 1:
            candidates.append(min(horizon, 3))
        if suggested_lag and suggested_lag > 1:
            candidates.append(suggested_lag)
        if seasonal_period and seasonal_period > 1:
            candidates.append(seasonal_period)

        # Keep lags feasible for short training windows.
        max_lag = max(1, dataset_length // 3) if dataset_length else 60
        out: List[int] = []
        for lag in candidates:
            if 1 <= int(lag) <= max_lag and int(lag) not in out:
                out.append(int(lag))
        return out or [1]

    def _default_rolling_windows(self, state: Dict[str, Any]) -> List[int]:
        """Deterministic, explicit completion for missing rolling_features.windows."""
        dataset_length = 0
        train_df = state.get("train_df")
        if isinstance(train_df, pd.DataFrame):
            dataset_length = len(train_df)

        suggested_lag = self._state_int(state, "data_summary.suggested_lag")
        seasonal_period = self._state_int(
            state,
            "data_summary.seasonal_periods",
        )
        horizon = self._state_int(state, "forecast_horizon") or 1
        frequency = str(
            state.get("data_summary", {})
            .get("base_metrics", {})
            .get("inferred_freq", "")
        ).upper()

        candidates: List[int] = []
        if frequency.startswith("D"):
            candidates.extend([7, 14, 30])
        elif frequency.startswith("W"):
            candidates.extend([4, 13, 26])
        elif frequency.startswith("M") or frequency in {"MS", "ME"}:
            candidates.extend([3, 6, 12])
        elif frequency.startswith("Q"):
            candidates.extend([2, 4, 8])
        else:
            candidates.extend([3, 6, 12])

        if horizon > 1:
            candidates.insert(0, max(2, min(horizon, 6)))
        if suggested_lag and suggested_lag > 1:
            candidates.insert(0, suggested_lag)
        if seasonal_period and seasonal_period > 1:
            candidates.insert(0, seasonal_period)

        max_window = max(2, dataset_length // 3) if dataset_length else 60
        out: List[int] = []
        for window in candidates:
            w = int(window)
            if 2 <= w <= max_window and w not in out:
                out.append(w)
            if len(out) >= 3:
                break
        return out or [3]

    def _complete_feature_generation_params(
        self,
        name: str,
        params: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize common aliases and explicitly complete feature-generation params.

        The LLM sometimes emits `rolling_features` with `window` or no windows at all.
        Instead of letting this reach PreprocessingAgent and burn retries, the
        TechnicalAgent converts clear aliases and deterministically fills missing
        feature-generation parameters using dataset diagnostics. The completion is
        recorded in parameters so it is auditable, not silent.
        """
        params = dict(params)

        if name == "lag_features":
            raw_lags = (
                params.get("lags")
                if "lags" in params
                else params.get("lag", params.get("lookback", params.get("lookbacks")))
            )
            lags = self._coerce_positive_int_list(
                raw_lags, "lag_features.parameters.lags", minimum=1
            )
            allowed_lags = self._default_lags(state)
            unsupported_lags = [lag for lag in lags if lag not in allowed_lags]
            if unsupported_lags:
                suggested_lag = self._state_int(
                    state, "data_summary.suggested_lag"
                )
                expanded_suggested_range = bool(
                    lags
                    and suggested_lag
                    and suggested_lag > 1
                    and lags == list(range(1, suggested_lag + 1))
                    and suggested_lag in allowed_lags
                )
                if expanded_suggested_range:
                    lags = [lag for lag in allowed_lags if lag in lags]
                    params["_auto_completed"] = True
                    params["_auto_completion_reason"] = (
                        "TechnicalAgent collapsed an expanded [1..suggested_lag] "
                        "sequence to the deterministic allowed lag values."
                    )
                else:
                    raise ValueError(
                        "lag_features contains lags unsupported by deterministic "
                        f"diagnostics: {unsupported_lags}. Allowed lags: {allowed_lags}."
                    )
            if not lags:
                lags = allowed_lags
                params["_auto_completed"] = True
                params["_auto_completion_reason"] = (
                    "TechnicalAgent filled missing lag_features.parameters.lags "
                    "from forecast horizon, suggested lag, and seasonal period."
                )
            params["lags"] = lags
            for alias in ["lag", "lookback", "lookbacks"]:
                params.pop(alias, None)

        elif name == "rolling_features":
            raw_windows = (
                params.get("windows")
                if "windows" in params
                else params.get(
                    "window",
                    params.get("rolling_window", params.get("rolling_windows")),
                )
            )
            windows = self._coerce_positive_int_list(
                raw_windows, "rolling_features.parameters.windows", minimum=2
            )
            if not windows:
                windows = self._default_rolling_windows(state)
                params["_auto_completed"] = True
                params["_auto_completion_reason"] = (
                    "TechnicalAgent filled missing rolling_features.parameters.windows "
                    "from frequency, forecast horizon, suggested lag, and seasonal period."
                )
            params["windows"] = windows
            for alias in ["window", "rolling_window", "rolling_windows"]:
                params.pop(alias, None)

            raw_functions = params.get("functions", params.get("function", ["mean"]))
            functions = (
                raw_functions if isinstance(raw_functions, list) else [raw_functions]
            )
            functions = [str(f).strip().lower() for f in functions if str(f).strip()]
            if not functions:
                functions = ["mean"]
            allowed = {"mean", "std", "min", "max"}
            invalid = [f for f in functions if f not in allowed]
            if invalid:
                raise ValueError(
                    f"Unsupported rolling feature functions: {invalid}. Allowed: {sorted(allowed)}"
                )
            params["functions"] = functions
            params.pop("function", None)

        return params

    def _validate_transformations(
        self,
        model: str,
        transformations: Any,
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(transformations, list):
            raise ValueError("Candidate preprocessing.transformations must be a list.")

        validated: List[Dict[str, Any]] = []
        key = model.strip().lower()

        for idx, t in enumerate(transformations):
            if not isinstance(t, dict):
                raise ValueError(
                    f"Preprocessing transformation at index {idx} must be a dictionary."
                )
            if "name" not in t:
                raise ValueError(
                    f"Preprocessing transformation at index {idx} missing required field 'name'."
                )
            if "parameters" not in t:
                raise ValueError(
                    f"Preprocessing transformation '{t.get('name')}' missing required field 'parameters'."
                )
            if not isinstance(t.get("parameters"), dict):
                raise ValueError(
                    f"Preprocessing transformation '{t.get('name')}' must have dictionary parameters."
                )

            name = self._canonicalize_transformation_name(t["name"])
            params = dict(t["parameters"])
            params = self._complete_feature_generation_params(name, params, state)
            unknown_parameters = sorted(
                set(params) - self.TRANSFORMATION_PARAMETER_CONTRACT[name]
            )
            if unknown_parameters:
                raise ValueError(
                    f"Unsupported parameters for {name}: {unknown_parameters}. "
                    "External numerical columns cannot be declared through "
                    "preprocessing parameters."
                )

            if name == "log_transform":
                method = str(params.get("method", "log1p"))
                if method not in {"log1p", "log"}:
                    raise ValueError("log_transform method must be log1p or log")
                if float(params.get("shift", 0.0)) != 0.0 or bool(
                    params.get("allow_shift", False)
                ):
                    raise ValueError(
                        "log_transform does not permit target shifting"
                    )
                train_df = state.get("train_df")
                if isinstance(train_df, pd.DataFrame) and "target" in train_df:
                    target = pd.to_numeric(train_df["target"], errors="coerce")
                    target = target[np.isfinite(target)]
                    if method == "log1p" and bool((target <= -1.0).any()):
                        raise ValueError(
                            "log1p is incompatible with training values <= -1"
                        )
                    if method == "log" and bool((target <= 0.0).any()):
                        raise ValueError(
                            "log is incompatible with non-positive training values"
                        )
                params = {"method": method}

            if name == "replace_outliers" and params:
                raise ValueError(
                    "replace_outliers uses fixed framework parameters and accepts "
                    "no candidate parameters"
                )

            if name in self.SCALERS:
                scope = str(params.get("scope", "features")).strip().lower()
                if scope not in {"features", "target"}:
                    raise ValueError(
                        f"{name}.parameters.scope must be 'features' or 'target'."
                    )
                params = {"scope": scope}

            if name == "lag_features":
                lags = self._coerce_positive_int_list(
                    params.get("lags"), "lag_features.parameters.lags", minimum=1
                )
                if not lags:
                    raise ValueError(
                        "lag_features requires a non-empty parameters.lags list."
                    )
                params["lags"] = lags

            if name == "rolling_features":
                windows = self._coerce_positive_int_list(
                    params.get("windows"),
                    "rolling_features.parameters.windows",
                    minimum=2,
                )
                if not windows:
                    raise ValueError(
                        "rolling_features requires a non-empty parameters.windows list."
                    )
                params["windows"] = windows
                functions = params.get("functions", ["mean"])
                if not isinstance(functions, list) or not functions:
                    raise ValueError(
                        "rolling_features.parameters.functions must be a non-empty list when provided."
                    )
                allowed = {"mean", "std", "min", "max"}
                invalid = [
                    str(f).strip().lower()
                    for f in functions
                    if str(f).strip().lower() not in allowed
                ]
                if invalid:
                    raise ValueError(
                        f"Unsupported rolling feature functions: {invalid}. Allowed: {sorted(allowed)}"
                    )
                params["functions"] = [str(f).strip().lower() for f in functions]

            validated.append({"name": name, "parameters": params})

        # Model-specific invalid combinations.
        for t in validated:
            name = t["name"]
            if key == "sarima" and name in {
                "calendar_features",
                "lag_features",
                "rolling_features",
            }:
                raise ValueError(
                    f"Transformation '{name}' is not valid for SARIMA; SARIMA "
                    "consumes target history only."
                )
            if key in {"arima", "ets"} and name in {
                "lag_features",
                "rolling_features",
                "calendar_features",
            }:
                raise ValueError(
                    f"Transformation '{name}' is not valid for {model} in this framework; {model} handles the time series directly."
                )

        names = [t["name"] for t in validated]
        validate_target_treatment_combination(names)

        feature_scalers = [
            t for t in validated
            if t["name"] in self.SCALERS
            and t["parameters"].get("scope", "features") == "features"
        ]
        target_scalers = [
            t for t in validated
            if t["name"] in self.SCALERS
            and t["parameters"].get("scope", "features") == "target"
        ]
        has_feature_scaler = bool(feature_scalers)
        has_feature_generator = any(name in self.FEATURE_GENERATORS for name in names)

        if key in self.TARGET_SCALING_REQUIRED_MODELS:
            forbidden_generators = [
                name for name in names if name in self.FEATURE_GENERATORS
            ]
            if forbidden_generators:
                raise ValueError(
                    f"{model} consumes only target history; target-derived "
                    "tabular feature generators are not consumed: "
                    f"{forbidden_generators}."
                )
            if feature_scalers:
                raise ValueError(
                    f"{model} does not consume tabular predictor columns; "
                    "scope='features' is not allowed."
                )
            if not target_scalers:
                raise ValueError(
                    f"{model} consumes the target sequence and requires a "
                    "standard_scaler or minmax_scaler with scope='target'."
                )

        if has_feature_scaler and not has_feature_generator:
            raise ValueError(
                f"{model}: standard_scaler/minmax_scaler are feature-only and require "
                "numeric predictor columns. Add lag_features, rolling_features, or "
                "calendar_features before the scaler."
            )
        if feature_scalers:
            first_feature_scaler = next(
                index for index, item in enumerate(validated)
                if item in feature_scalers
            )
            generated_before_scaling = any(
                item["name"] in self.FEATURE_GENERATORS
                for item in validated[:first_feature_scaler]
            )
            if has_feature_generator and not generated_before_scaling:
                raise ValueError(
                    "Feature scaling must follow feature generation so the fitted "
                    "scaler acts on the predictors consumed by the model."
                )

        if len(feature_scalers) > 1 or len(target_scalers) > 1:
            raise ValueError("At most one scaler is allowed for each scope.")

        if key in self.FEATURE_SCALING_REQUIRED_MODELS:
            if not has_feature_generator:
                raise ValueError(
                    f"{model} requires predictor features in this framework. Add at least one "
                    "of lag_features, rolling_features, or calendar_features before scaling."
                )
            if not has_feature_scaler:
                raise ValueError(
                    f"{model} requires feature normalization/scaling. Add standard_scaler or minmax_scaler."
                )

        return validated

    @staticmethod
    def _extract_attempt_transformations(
        attempt: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prep = attempt.get("preprocessing")
        if isinstance(prep, dict):
            transforms = prep.get("transformations", [])
        elif isinstance(prep, list):
            transforms = prep
        else:
            transforms = attempt.get("preprocessing_transformations", [])
        return [
            item for item in transforms
            if not (isinstance(item, dict) and item.get("name") == "replace_outliers")
        ] if isinstance(transforms, list) else []

    @staticmethod
    def _extract_best_transformations(
        best_model_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prep_log = best_model_data.get("preprocessing_log")
        if isinstance(prep_log, dict):
            transforms = prep_log.get("transformations", [])
        elif isinstance(prep_log, list):
            transforms = prep_log
        else:
            prep = best_model_data.get("preprocessing")
            if isinstance(prep, dict):
                transforms = prep.get("transformations", [])
            elif isinstance(prep, list):
                transforms = prep
            else:
                transforms = []
        return [
            item for item in transforms
            if not (isinstance(item, dict) and item.get("name") == "replace_outliers")
        ] if isinstance(transforms, list) else []

    def _technical_contract_view(
        self, candidate: Dict[str, Any], *, canonicalize_aliases: bool
    ) -> Dict[str, Any]:
        """Return contract-bearing fields, optionally normalizing harmless aliases."""
        preprocessing = candidate.get("preprocessing")
        transformations = (
            preprocessing.get("transformations")
            if isinstance(preprocessing, dict)
            else preprocessing
        )
        normalized_transformations: Any = copy.deepcopy(transformations)
        if canonicalize_aliases and isinstance(transformations, list):
            normalized_transformations = []
            for transformation in transformations:
                item = copy.deepcopy(transformation)
                if isinstance(item, dict) and "name" in item:
                    item["name"] = self._canonicalize_transformation_name(
                        item["name"]
                    )
                    if item["name"] == "replace_outliers":
                        continue
                normalized_transformations.append(item)
        elif isinstance(normalized_transformations, list):
            normalized_transformations = [
                item for item in normalized_transformations
                if not (isinstance(item, dict) and item.get("name") == "replace_outliers")
            ]
        return {
            "model_type": candidate.get("model_type"),
            "hyperparameters": copy.deepcopy(candidate.get("hyperparameters")),
            "preprocessing": normalized_transformations,
        }

    def _record_technical_compliance(
        self,
        state: Dict[str, Any],
        *,
        raw_decision: Dict[str, Any],
        raw_candidate: Dict[str, Any],
        final_candidate: Dict[str, Any] | None,
        error: Exception | None = None,
    ) -> None:
        """Passively audit one parsed proposal against the existing contract."""
        repair_required = False
        violations: List[Dict[str, Any]] = []
        compliant = error is None and final_candidate is not None
        if compliant:
            before = self._technical_contract_view(
                raw_candidate, canonicalize_aliases=True
            )
            after = self._technical_contract_view(
                final_candidate, canonicalize_aliases=False
            )
            repair_required = before != after
            if repair_required:
                compliant = False
                violations.append({
                    "type": "corrective_repair",
                    "field": "candidate_configuration",
                    "reason": (
                        "Accepted candidate differs from the parsed LLM proposal "
                        "after harmless name-alias canonicalization."
                    ),
                })
        elif error is not None:
            violations.append({
                "type": "contract_validation_error",
                "field": "candidate_configuration",
                "reason": str(error),
            })

        trace = state.setdefault("technical_compliance_trace", [])
        trace.append({
            "proposal_id": f"technical-proposal-{len(trace) + 1}",
            "step": int(state.get("step", 0)),
            "raw_parsed_proposal": copy.deepcopy(raw_decision),
            "contract_compliant": bool(compliant),
            "violations": violations,
            "repair_required": bool(repair_required),
            "final_candidate": copy.deepcopy(final_candidate),
        })

    def _canonical_config_key(
        self,
        model: str,
        hyperparameters: Dict[str, Any],
        transformations: List[Dict[str, Any]],
    ) -> str:
        from utils.configuration_identity import canonical_configuration_signature
        return canonical_configuration_signature(model, hyperparameters, transformations)

    @staticmethod
    def _requires_replace_outliers(state: Dict[str, Any]) -> bool:
        diagnostics = state.get("extreme_outliers_detected", {})
        return bool(
            isinstance(diagnostics, dict)
            and diagnostics.get("extreme_outlier_detected", False)
        )

    @classmethod
    def _enforce_replace_outliers_requirement(
        cls,
        transformations: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Insert the deterministic, train-only outlier step when required."""
        without_outliers = [
            copy.deepcopy(step)
            for step in transformations
            if not (
                isinstance(step, dict)
                and str(step.get("name", "")).strip().lower()
                == "replace_outliers"
            )
        ]
        if cls._requires_replace_outliers(state):
            return [{"name": "replace_outliers", "parameters": {}}] + without_outliers
        return without_outliers

    def _validate_model(self, model: str, state: Dict[str, Any]) -> str | None:
        catalog = state.get("model_catalog", {})
        base_models = (
            catalog.get("available_models", [])
            if isinstance(catalog, dict) else []
        )
        return model if model in base_models else None

    def _record_failed_candidate(
        self, candidate: Dict[str, Any], err_msg: str, state: Dict[str, Any]
    ):
        failed_record = {
                "model": candidate.get("model_type"),
                "hyperparameters": candidate.get("hyperparameters", {}),
                "preprocessing": candidate.get(
                    "preprocessing", {"transformations": []}
                ),
                "mase": float("inf"),
                "rmse": float("inf"),
                "error": err_msg,
                "retry_reason": f"Technical validation failed: {err_msg}",
                "failure_type": "validation_error",
            }
        from utils.configuration_identity import canonical_configuration_signature
        failed_record["configuration_signature"] = canonical_configuration_signature(
            str(candidate.get("model_type", "")), candidate.get("hyperparameters", {}), candidate.get("preprocessing", {})
        )
        failed_record["failure_fingerprint"] = canonical_failure_fingerprint(
            str(candidate.get("model_type", "")),
            candidate.get("hyperparameters", {}),
            candidate.get("preprocessing", {}),
            "invalid_configuration",
            err_msg,
        )
        reason_lower = err_msg.lower()
        if (
            "already tested" in reason_lower
            or "exact failed" in reason_lower
            or "signature" in reason_lower
            or "cannot be retried" in reason_lower
        ):
            rejection_code = "duplicate_candidate"
        elif "preprocessing" in reason_lower or "transformation" in reason_lower:
            rejection_code = "invalid_preprocessing_contract"
        elif "invalid model" in reason_lower:
            rejection_code = "ineligible_model"
        elif "refine mode" in reason_lower:
            rejection_code = "wrong_model_in_refine"
        elif "hyperparameter" in reason_lower:
            rejection_code = "invalid_model_contract"
        else:
            rejection_code = "no_valid_configuration"
        rejected = state.setdefault("rejected_technical_candidates", [])
        rejected.append({
            "rejected_candidate_id": f"rejected-{len(rejected) + 1}",
            "step": int(state.get("step", 0)),
            "attempt": int(state.get("technical_retry_count", 0)) + 1,
            "technical_decision_id": candidate.get("technical_decision_id", f"decision-{int(state.get('step', 0))}"),
            "model_type": candidate.get("model_type"),
            "hyperparameters": candidate.get("hyperparameters", {}),
            "preprocessing": candidate.get("preprocessing", {}).get("transformations", []) if isinstance(candidate.get("preprocessing"), dict) else candidate.get("preprocessing", []),
            "configuration_signature": failed_record["configuration_signature"],
            "failure_fingerprint": failed_record["failure_fingerprint"],
            "rejection_stage": "post_parse_validation",
            "rejection_code": rejection_code,
            "reason": err_msg,
            "was_sent_to_training": False,
        })

    def _validate_candidate_config(
        self, candidate: Dict[str, Any], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize the candidate configuration atomically."""
        try:
            validate_candidate_numeric_contract(candidate)
        except ValueError as exc:
            self._record_failed_candidate(candidate, str(exc), state)
            raise
        model_type = candidate.get("model_type")
        if not model_type or not isinstance(model_type, str):
            err = "Candidate configuration missing valid 'model_type'"
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)

        model = self._validate_model(model_type, state)
        if not model:
            err = f"Invalid model_type: {model_type}"
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)

        candidate["model_type"] = model
        candidate_request = state.get("candidate_request", {})
        requested_model = (
            candidate_request.get("model")
            if isinstance(candidate_request, dict)
            else None
        )
        required_refinement_model = (
            candidate_request.get("required_refinement_model")
            if isinstance(candidate_request, dict)
            else None
        )
        if required_refinement_model and (
            candidate_request.get("action") != "refine"
            or model != required_refinement_model
        ):
            err = (
                "Local preprocessing refinement must remain scoped to "
                f"model={required_refinement_model!r}; action="
                f"{candidate_request.get('action')!r}, model={model!r}."
            )
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)
        if requested_model and model != requested_model:
            err = (
                f"CandidateRequest requires model_type={requested_model!r}; "
                f"received {model!r}."
            )
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)

        hp = candidate.get("hyperparameters")
        try:
            hp, normalization_audit = normalize_candidate_hyperparameters(
                model, hp, candidate.get("preprocessing")
            )
            candidate["hyperparameters"] = self._validate_hyperparams(model, hp)
            if normalization_audit:
                candidate["configuration_normalization"] = normalization_audit
            memory_key = {
                "N-BEATS": "lookback",
                "LSTM": "lag",
                "DLinear": "lag",
                "Reformer": "lag",
            }.get(model)
            train_df = state.get("raw_train_df", state.get("train_df"))
            if memory_key and isinstance(train_df, pd.DataFrame):
                memory = int(candidate["hyperparameters"][memory_key])
                horizon = int(state.get("forecast_horizon", 1))
                if len(train_df) < memory + horizon:
                    raise ValueError(
                        f"{model} {memory_key}={memory} requires at least "
                        f"{memory + horizon} raw training points for horizon={horizon}; "
                        f"received {len(train_df)}. The runtime will not shrink "
                        "declared temporal memory."
                    )
        except ValueError as exc:
            err = str(exc)
            self._record_failed_candidate(candidate, err, state)
            raise


        if model == "ETS" and any(
            candidate["hyperparameters"].get(component) == "mul"
            for component in ("trend", "seasonal")
        ):
            train_df = state.get("train_df")
            if isinstance(train_df, pd.DataFrame) and "target" in train_df.columns:
                target = pd.to_numeric(train_df["target"], errors="coerce")
                if bool((target.dropna() <= 0).any()):
                    err = (
                        "ETS multiplicative trend/seasonal components require a "
                        "strictly positive training target; zero or negative values "
                        "were found. Use additive or null components."
                    )
                    self._record_failed_candidate(candidate, err, state)
                    raise ValueError(err)

        preprocessing = candidate.get("preprocessing")
        if not isinstance(preprocessing, dict):
            err = "Candidate preprocessing must be a dictionary with a 'transformations' list."
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)
        if "transformations" not in preprocessing:
            err = "Candidate preprocessing missing required field: transformations."
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)

        proposed_transformations = list(preprocessing["transformations"])
        required_target_transform = (
            candidate_request.get("required_target_transform")
            if isinstance(candidate_request, dict)
            else None
        )
        proposed_names = {
            str(step.get("name", "")).strip().lower()
            for step in proposed_transformations
            if isinstance(step, dict)
        }
        if required_target_transform == "log1p":
            if not bool(state.get("log_transform_supported", False)):
                raise ValueError("Planner requested log1p for an incompatible target")
            incompatible = proposed_names.intersection({"boxcox_transform"})
            if incompatible:
                raise ValueError(
                    "Required log1p trial cannot include alternative target treatments"
                )
            if "log_transform" not in proposed_names:
                raise ValueError(
                    "Planner requested a log1p candidate; TechnicalAgent must "
                    "propose log_transform explicitly"
                )
        elif required_target_transform == "raw" and proposed_names.intersection(
            {"log_transform", "boxcox_transform"}
        ):
            raise ValueError("Planner requested a raw-target candidate")

        proposed_transformations = self._enforce_replace_outliers_requirement(
            proposed_transformations, state
        )

        try:
            transformations = self._validate_transformations(
                model, proposed_transformations, state
            )
        except ValueError as exc:
            err = str(exc)
            self._record_failed_candidate(candidate, err, state)
            raise

        preprocessing = dict(preprocessing)
        preprocessing["transformations"] = transformations
        if "reasoning" not in preprocessing:
            preprocessing["reasoning"] = ""
        candidate["preprocessing"] = preprocessing
        candidate["status"] = "validated"
        return candidate


    def _allowed_hyperparameter_keys(self, model: str) -> set[str]:
        default_schema = DEFAULT_HYPERPARAMETERS.get(model, {})
        constraint_schema = HYPERPARAMETER_CONSTRAINTS.get(model, {})

        allowed_keys = set()
        if isinstance(default_schema, dict):
            allowed_keys.update(default_schema.keys())
        if isinstance(constraint_schema, dict):
            allowed_keys.update(constraint_schema.keys())
        allowed_keys.update(self.EXTRA_ALLOWED_HYPERPARAMETERS.get(model, set()))
        return allowed_keys

    def _validate_hyperparams(self, model: str, hp: Any) -> Dict[str, Any]:
        """Validate LLM-proposed hyperparameters without silently replacing them.

        The LLM must provide explicit hyperparameters. Unknown keys are rejected.
        Defaults are not injected here, except for data-driven additions handled explicitly
        later in run() such as ETS seasonal_periods.

        The allowed-key contract is the union of:
        - DEFAULT_HYPERPARAMETERS[model],
        - HYPERPARAMETER_CONSTRAINTS[model],
        - EXTRA_ALLOWED_HYPERPARAMETERS[model] for known implementation-supported keys.
        """
        if not isinstance(hp, dict):
            raise ValueError(f"{model} hyperparameters must be a dictionary.")
        if not hp:
            raise ValueError(f"{model} requires explicit non-empty hyperparameters.")
        from core.candidate_catalog import DEFAULT_CANDIDATE_CATALOG

        DEFAULT_CANDIDATE_CATALOG.validate(model, hp)

        allowed_keys = self._allowed_hyperparameter_keys(model)
        if allowed_keys:
            unknown = sorted(set(hp.keys()) - allowed_keys)
            if unknown:
                raise ValueError(
                    f"Unknown hyperparameters for {model}: {unknown}. "
                    f"Allowed keys: {sorted(allowed_keys)}"
                )

        if model == "FEDformer" and hp.get("n_head") != 8:
            raise ValueError(
                "FEDformer hyperparameter n_head must be 8 for the installed "
                "NeuralForecast implementation."
            )

        return dict(hp)

    def _enforce_refine_mode(
        self, candidate: Dict[str, Any], state: Dict[str, Any]
    ) -> None:
        planner_signals = state.get("planner_signals", {})
        (
            candidate_request,
            candidate_request_action,
            candidate_request_model,
        ) = self._candidate_request_contract(state)
        strategy = (
            candidate_request_action
            or str(planner_signals.get("technical_mode", "explore")).lower()
        )

        best_model_data = state.get("best_model") or {}
        current_best_model = best_model_data.get("selected_model")
        target_model = (
            candidate_request_model
            if candidate_request
            else planner_signals.get("refinement_target_model")
            or current_best_model
        )
        perf_hist = state.get("performance_history", [])
        target_attempts = [
            item for item in perf_hist
            if isinstance(item, dict) and item.get("model") == target_model
        ]
        target_data = min(
            target_attempts,
            key=lambda item: float(item.get("mase", float("inf"))),
            default=best_model_data if target_model == current_best_model else {},
        )
        target_hyperparameters = target_data.get("hyperparameters", {})
        target_transformations = self._extract_attempt_transformations(target_data)

        is_refine = strategy == "refine" and target_model is not None
        if not is_refine:
            return

        if candidate["model_type"] != target_model:
            err = f"You are in refine mode. You must keep model_type = '{target_model}'."
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)

        candidate_key = self._canonical_config_key(
            candidate["model_type"],
            candidate["hyperparameters"],
            candidate["preprocessing"]["transformations"],
        )
        current_best_key = self._canonical_config_key(
            target_model,
            target_hyperparameters,
            target_transformations,
        )

        if candidate_key == current_best_key:
            err = "Refine mode requires changing at least one meaningful hyperparameter or preprocessing step."
            self._record_failed_candidate(candidate, err, state)
            raise ValueError(err)


    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError(
                "TechnicalAgent requires an LLM instance; no fallback decision is allowed."
            )

        try:
            decision = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            raise RuntimeError(f"TechnicalAgent LLM call failed: {exc}") from exc

        if not isinstance(decision, dict) or not decision:
            raise RuntimeError(
                "TechnicalAgent LLM returned empty or invalid JSON decision."
            )

        return decision

    @staticmethod
    def _validate_llm_output_schema(decision: Dict[str, Any]) -> None:
        """Enforce exactly the two Technical Agent outputs shown in the paper."""
        no_candidate_fields = {
            "status",
            "reason",
            "request_planner_replan",
            "exhausted_scope",
            "blocked_by",
        }
        candidate_fields = {
            "model_type",
            "hyperparameters",
            "preprocessing",
            "ad_hoc_findings",
            "rationale",
        }
        if decision.get("status") == "NO_VALID_NEW_CONFIGURATION":
            expected = no_candidate_fields
        else:
            expected = candidate_fields
        actual = set(decision)
        missing = sorted(expected - actual)
        additional = sorted(actual - expected)
        if missing or additional:
            raise ValueError(
                "Technical Agent output does not match the paper schema: "
                f"missing={missing}, additional={additional}."
            )
        if expected is no_candidate_fields:
            if not isinstance(decision["blocked_by"], list):
                raise ValueError(
                    "Technical Agent blocked_by must be a list in the paper schema."
                )
            return

        if not isinstance(decision["model_type"], str):
            raise ValueError("Technical Agent model_type must be a string.")
        if not isinstance(decision["hyperparameters"], dict):
            raise ValueError("Technical Agent hyperparameters must be an object.")
        if not isinstance(decision["ad_hoc_findings"], dict):
            raise ValueError("Technical Agent ad_hoc_findings must be an object.")
        if not isinstance(decision["rationale"], str):
            raise ValueError("Technical Agent rationale must be a string.")
        preprocessing = decision["preprocessing"]
        if not isinstance(preprocessing, dict) or set(preprocessing) != {
            "transformations",
            "reasoning",
        }:
            raise ValueError(
                "Technical Agent preprocessing must contain exactly "
                "transformations and reasoning."
            )
        if not isinstance(preprocessing["transformations"], list):
            raise ValueError(
                "Technical Agent preprocessing.transformations must be a list."
            )
        if not isinstance(preprocessing["reasoning"], str):
            raise ValueError("Technical Agent preprocessing.reasoning must be a string.")
        for index, transformation in enumerate(preprocessing["transformations"]):
            if not isinstance(transformation, dict) or set(transformation) != {
                "name",
                "parameters",
            }:
                raise ValueError(
                    "Technical Agent transformation at index "
                    f"{index} must contain exactly name and parameters."
                )
            if not isinstance(transformation["name"], str) or not isinstance(
                transformation["parameters"], dict
            ):
                raise ValueError(
                    "Technical Agent transformation name must be a string and "
                    "parameters must be an object."
                )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select best model, hyperparameters and preprocessing.

        Training happens in TrainingAgent.
        """
        step = int(state.get("step", 0))
        state.pop("current_candidate_config", None)

        train_df = state.get("train_df")
        val_df = state.get("val_df")

        if not isinstance(train_df, pd.DataFrame):
            raise ValueError("train_df missing")
        if not isinstance(val_df, pd.DataFrame):
            raise ValueError("val_df missing")

        # 1. PROMPTS
        system_prompt = self._build_system_prompt(state)
        user_prompt = self._build_user_prompt(state)


        # 2. LLM DECISION
        decision = self._call_llm(system_prompt, user_prompt)
        self._validate_llm_output_schema(decision)

        decision_status = decision.get("status", "VALID_CANDIDATE")
        if decision_status == "NO_VALID_NEW_CONFIGURATION":
            technical_escalation = {
                "status": "NO_VALID_NEW_CONFIGURATION",
                "reason": decision.get(
                    "reason",
                    "No valid unblocked configuration exists under current guidance.",
                ),
                "request_planner_replan": bool(
                    decision.get("request_planner_replan", True)
                ),
                "exhausted_scope": decision.get("exhausted_scope", "unknown"),
                "blocked_by": decision.get("blocked_by", []),
                "step": step,
                "source": "technical_agent",
            }

            logger.warning(
                "TechnicalAgent requested planner replan: %s",
                technical_escalation,
            )

            return {
                "technical_status": "NO_VALID_NEW_CONFIGURATION",
                "request_planner_replan": True,
                "technical_escalation": technical_escalation,

                "technical_error": False,
                "last_technical_error": None,
                "technical_retry_count": state.get("technical_retry_count", 0),
            }

        candidate = {
            "model_type": decision.get("model_type"),
            "hyperparameters": decision.get("hyperparameters"),
            "preprocessing": decision.get("preprocessing"),
            "source": "technical_agent",
            "step": step,
            "technical_decision_id": f"decision-{step}",
            "status": "selected",
            "validation_errors": [],
            "reasoning": decision.get("rationale", ""),
        }
        print("TechnicalAgent: candidate parsed: %s" % candidate["model_type"])
        logger.info("TechnicalAgent: candidate parsed: %s", candidate)

        # 3. VALIDATE BASIC CANDIDATE CONFIGURATION
        logger.info("TechnicalAgent: validating candidate config...")
        raw_candidate = copy.deepcopy(candidate)
        try:
            candidate = self._validate_candidate_config(candidate, state)
        except Exception as exc:
            self._record_technical_compliance(
                state,
                raw_decision=decision,
                raw_candidate=raw_candidate,
                final_candidate=None,
                error=exc,
            )
            raise

        # 4. DATA-DRIVEN HYPERPARAMETER INJECTION WITH EXPLICIT VALIDATION
        data_summary = state.get("data_summary", {})
        seasonal_periods_computed = data_summary.get("seasonal_periods")

        hp = candidate["hyperparameters"]
        if isinstance(hp, dict):
            hp = dict(hp)
            if candidate["model_type"] == "ETS" and hp.get("seasonal") is not None:
                if "seasonal_periods" not in hp or not hp.get("seasonal_periods"):
                    if seasonal_periods_computed and int(seasonal_periods_computed) > 1:
                        hp["seasonal_periods"] = int(seasonal_periods_computed)
                        logger.info(
                            "TechnicalAgent: injecting AnalyticalAgent seasonal_periods=%d into ETS hp.",
                            int(seasonal_periods_computed),
                        )
                        candidate["hyperparameters"] = hp
                        from core.candidate_catalog import DEFAULT_CANDIDATE_CATALOG

                        try:
                            DEFAULT_CANDIDATE_CATALOG.validate(
                                candidate["model_type"],
                                candidate["hyperparameters"],
                            )
                        except ValueError as exc:
                            self._record_failed_candidate(
                                candidate, str(exc), state
                            )
                            self._record_technical_compliance(
                                state,
                                raw_decision=decision,
                                raw_candidate=raw_candidate,
                                final_candidate=None,
                                error=exc,
                            )
                            raise
                    else:
                        err_msg = (
                            "ETS with seasonal component requires a valid seasonal_periods computed by analytical agent. "
                            "No such period was found."
                        )
                        logger.warning(
                            "TechnicalAgent: ETS validation failed. %s", err_msg
                        )
                        self._record_failed_candidate(candidate, err_msg, state)
                        exc = ValueError(err_msg)
                        self._record_technical_compliance(
                            state,
                            raw_decision=decision,
                            raw_candidate=raw_candidate,
                            final_candidate=None,
                            error=exc,
                        )
                        raise exc

        # 5. STRICT MODE AND MEMORY VALIDATION
        try:
            self._enforce_refine_mode(candidate, state)
        except Exception as exc:
            self._record_technical_compliance(
                state,
                raw_decision=decision,
                raw_candidate=raw_candidate,
                final_candidate=None,
                error=exc,
            )
            raise
        self._record_technical_compliance(
            state,
            raw_decision=decision,
            raw_candidate=raw_candidate,
            final_candidate=candidate,
        )

        logger.info("TechnicalAgent: candidate accepted.")
        logger.info(
            "TechnicalAgent: final candidate config before training: %s", candidate
        )

        # 6. SET STATE FOR DOWNSTREAM AGENTS
        state["current_candidate_config"] = candidate
        state["model_rationale"] = decision.get("rationale", "")
        state["model_confidence"] = 0.5

        ad_hoc = decision.get("ad_hoc_findings", {})
        if ad_hoc:
            if "analysis_history" not in state:
                state["analysis_history"] = []
            state["analysis_history"].append(
                {
                    "task": "technical",
                    "result": ad_hoc,
                }
            )

        return state
