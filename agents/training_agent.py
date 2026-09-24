"""Training agent compatible with the canonical technical -> preprocessing -> training chain.

"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from models.trainer import Trainer
logger = logging.getLogger(__name__)


def select_validation_winner(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return only the finite validation-MASE argmin; no prior is accepted."""
    finite = []
    for item in history:
        try:
            mase = float(item.get("mase"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(mase):
            finite.append(dict(item))
    if not finite:
        raise ValueError("No finite validation MASE is available")
    return min(finite, key=lambda item: float(item["mase"]))



@dataclass
class TrainingAgent:
    """Train and evaluate selected models on data prepared by PreprocessingAgent."""

    trainer: Trainer
    finalize_mase_threshold: float = 0.85  # quality threshold only, not automatic END

    @staticmethod
    def _require_df(state: Dict[str, Any], key: str) -> pd.DataFrame:
        df = state.get(key)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"state['{key}'] must be a pandas DataFrame.")
        if df.empty:
            raise ValueError(f"state['{key}'] must not be empty.")
        if "target" not in df.columns:
            raise ValueError(f"state['{key}'] must contain a 'target' column.")
        return df

    @staticmethod
    def _candidate_preprocessing_transformations(candidate: Dict[str, Any]) -> list[Dict[str, Any]]:
        preprocessing = candidate.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            return []
        transformations = preprocessing.get("transformations", [])
        return transformations if isinstance(transformations, list) else []

    @staticmethod
    def _record_failed_attempt(
        state: Dict[str, Any],
        model: str | None,
        hp: Dict[str, Any] | None,
        preprocessing: list[Dict[str, Any]],
        err_msg: str,
        forecast_horizon: int | None = None,
        val_size: int | None = None,
    ) -> None:
        if not model:
            return
        failed_record = {
            "model": model,
            "hyperparameters": hp or {},
            "preprocessing": preprocessing,
            "validation_protocol": "preprocessed_holdout",
            "val_size": val_size if val_size is not None else 0,
            "forecast_horizon": forecast_horizon if forecast_horizon is not None else state.get("forecast_horizon"),
            "evaluated_points": 0,
            "mase": float("inf"),
            "rmse": float("inf"),
            "error": err_msg,
            "retry_reason": f"Training failed: {err_msg}",
        }
        state["last_candidate_failure"] = failed_record

    def _resolve_prepared_datasets(
        self, state: Dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return preprocessed and raw datasets.

        Because the graph is technical -> preprocessing -> training, preprocessed
        train and validation data are required. Raw train/validation are also
        required so metrics can be computed on the original target scale by the
        trainer using the preprocessing log.
        """
        actual_train_df = self._require_df(state, "preprocessed_train_df")
        actual_val_df = self._require_df(state, "preprocessed_val_df")

        raw_train_df = self._require_df(state, "raw_train_df")
        raw_val_df = self._require_df(state, "raw_validation_df")
        current_val_df = self._require_df(state, "val_df")
        if not np.array_equal(
            raw_val_df["target"].to_numpy(dtype=float),
            current_val_df["target"].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise AssertionError("Validation target was modified before evaluation")

        return actual_train_df, actual_val_df, raw_train_df, raw_val_df

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate the selected model on preprocessed data."""
        step = int(state.get("step", 0))
        candidate = state.get("current_candidate_config")
        if not isinstance(candidate, dict):
            raise ValueError(
                "state['current_candidate_config'] must be a dictionary. TechnicalAgent must run before TrainingAgent."
            )

        model = candidate.get("model_type")
        hp = candidate.get("hyperparameters")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "state['current_candidate_config']['model_type'] is missing. TechnicalAgent must run before TrainingAgent."
            )
        if not isinstance(hp, dict) or not hp:
            raise ValueError(
                f"state['current_candidate_config']['hyperparameters'] must be a non-empty dictionary for model '{model}'."
            )

        preprocessing_log = state.get("preprocessing_log")
        if not isinstance(preprocessing_log, dict):
            raise ValueError("state['preprocessing_log'] must be a dictionary. PreprocessingNode must run before TrainingAgent.")
        if "transformations" not in preprocessing_log:
            raise ValueError("state['preprocessing_log'] missing required field 'transformations'.")
        if not isinstance(preprocessing_log.get("transformations"), list):
            raise ValueError("state['preprocessing_log']['transformations'] must be a list.")

        preprocessing_transformations = preprocessing_log.get("transformations", [])
        candidate_record = state.get("candidate_record")
        if isinstance(candidate_record, dict):
            expected_candidate_id = candidate_record.get("candidate_id")
            if candidate_record.get("status") != "pending":
                raise ValueError(
                    "Training requires a pending candidate transaction"
                )
            if (
                preprocessing_log.get("candidate_id") != expected_candidate_id
                or state.get("preprocessing_candidate_id")
                != expected_candidate_id
            ):
                raise ValueError(
                    "Preprocessing artifacts do not belong to the pending "
                    "candidate transaction"
                )
            if preprocessing_log.get("configuration_signature") != (
                candidate_record.get("configuration_signature")
            ):
                raise ValueError(
                    "Preprocessing signature does not match the pending candidate"
                )
        forecast_horizon = int(state.get("forecast_horizon", 10))
        candidate_seed = int(state.get("candidate_seed", 0))
        state["determinism_audit"] = self.trainer.prepare_candidate_training(
            candidate_seed,
            training_phase="VALIDATION",
            candidate_signature=str(
                candidate_record.get("configuration_signature", "")
                if isinstance(candidate_record, dict)
                else ""
            ),
        )

        try:
            (
                actual_train_df,
                actual_val_df,
                raw_train_df,
                raw_val_df,
            ) = self._resolve_prepared_datasets(state)

            val_size = len(actual_val_df)

            semantic_features_used = False

            data_summary = state.get("data_summary", {})
            base_metrics = data_summary.get("base_metrics", {})
            inferred_freq = base_metrics.get("inferred_freq") or base_metrics.get("frequency") or state.get("inferred_freq")
            mase_reference = state.get("mase_reference")
            if not isinstance(mase_reference, dict):
                raise RuntimeError("Missing immutable search MASE reference.")
            if not bool(mase_reference.get("defined")):
                raise ValueError(
                    "Candidate cannot be ranked because the raw training MASE "
                    "denominator is undefined."
                )
            if int(mase_reference.get("training_size", -1)) != len(raw_train_df):
                raise RuntimeError("MASE reference does not match the common raw training split.")
            mase_seasonality = int(mase_reference["period"])
            mase_seasonality_source = str(mase_reference["period_source"])

            logger.info(
                "[MASE] model=%s, chosen_mase_seasonality=%s, source=%s, y_train_len=%s, inferred_freq=%s, seasonality_detected=%s, seasonality_period=%s, suggested_lag=%s",
                model,
                mase_seasonality,
                mase_seasonality_source,
                len(raw_train_df),
                inferred_freq,
                data_summary.get("seasonality_detected"),
                data_summary.get("seasonal_periods"),
                data_summary.get("suggested_lag"),
            )

            training = self.trainer.train_and_evaluate(
                model_name=model,
                hyperparameters=hp,
                train_df=actual_train_df,
                val_df=actual_val_df,
                raw_train_df=raw_train_df,
                raw_val_df=raw_val_df,
                preprocessing_log=preprocessing_log,
                forecast_horizon=forecast_horizon,
                mase_seasonality=mase_seasonality,
                freq=inferred_freq or "D",
            )
            actual_denominator = training.metrics.get("mase_denominator_value")
            if not np.isclose(
                float(actual_denominator), float(mase_reference["denominator"])
            ):
                raise RuntimeError("Candidate MASE denominator differs from the fixed search reference.")
            training.metrics["mase_seasonality_source"] = mase_seasonality_source
            training.metrics["mase_reference"] = dict(mase_reference)

            trained_hp = training.hyperparameters
            from utils.configuration_identity import (
                canonical_configuration_signature,
            )

            requested_signature = canonical_configuration_signature(
                model,
                hp,
                self._candidate_preprocessing_transformations(candidate),
            )
            trained_signature = canonical_configuration_signature(
                model,
                trained_hp,
                self._candidate_preprocessing_transformations(candidate),
            )
            if trained_signature != requested_signature:
                raise ValueError(
                    "Trainer changed the selected candidate hyperparameters; "
                    "implicit tuning is forbidden inside an atomic candidate"
                )
            hp = trained_hp
            candidate["hyperparameters"] = hp
            state["current_candidate_config"] = candidate

            predictions = np.asarray(training.predictions, dtype=float)
            raw_y_true = np.asarray(raw_val_df["target"].to_numpy(dtype=float), dtype=float)
            if len(raw_y_true) != len(predictions):
                raise ValueError(
                    "Every validation observation must have one prediction: "
                    f"y_true={len(raw_y_true)}, y_pred={len(predictions)}"
                )
            training.metrics["validation_points_excluded_from_metrics"] = 0
            training.metrics["total_validation_points"] = len(raw_y_true)
            training.metrics["evaluated_points"] = len(raw_y_true)
            training.metrics["metrics_computed_on"] = "full_original_validation"

            current_mase = float(training.metrics["mase"])
            if not math.isfinite(current_mase):
                raise ValueError(
                    "Candidate validation MASE is not finite and cannot enter selection."
                )
            current_rmse = float(training.metrics["rmse"])
            current_mse = float(training.metrics.get("mse", float("inf")))
            scaled_mse = training.metrics.get("scaled_mse")
            scaled_mse = float(scaled_mse) if scaled_mse is not None else None
            current_r2 = float(training.metrics.get("r2", 0.0))
            current_mae = float(training.metrics.get("mae", 0.0))
            current_mape = float(training.metrics.get("mape", 0.0))

            val_metrics = training.metrics
            preprocessing_steps = len(preprocessing_transformations)
            from utils.configuration_identity import target_transform_label

            target_transform = target_transform_label(
                preprocessing_transformations
            )
            technical_decision_id = str(candidate.get("technical_decision_id", f"decision-{step}"))
            training_id = f"training-{step}"
            candidate_id = (
                candidate_record.get("candidate_id")
                if isinstance(candidate_record, dict)
                else None
            )
            configuration_signature = (
                str(candidate_record["configuration_signature"])
                if isinstance(candidate_record, dict)
                and candidate_record.get("configuration_signature")
                else requested_signature
            )

            result = {
                "step": step,
                "technical_decision_id": technical_decision_id,
                "candidate_id": candidate_id,
                "training_id": training_id,
                "configuration_signature": configuration_signature,
                "candidate_seed": candidate_seed,
                "model": model,
                "hyperparameters": hp,
                "validation_protocol": val_metrics.get("validation_protocol", "preprocessed_holdout"),
                "val_size": val_size,
                "forecast_horizon": val_metrics.get("forecast_horizon", forecast_horizon),
                "forecast_protocol": {
                    "configured_forecast_horizon": val_metrics.get(
                        "configured_forecast_horizon", forecast_horizon
                    ),
                    "validation_protocol": val_metrics.get(
                        "validation_protocol", "preprocessed_holdout"
                    ),
                    "number_of_validation_origins": val_metrics.get(
                        "number_of_validation_origins", 1
                    ),
                    "validation_effective_horizons": val_metrics.get(
                        "validation_effective_horizons", []
                    ),
                    "horizon_consistency": val_metrics.get(
                        "horizon_consistency", True
                    ),
                },
                "evaluated_points": val_metrics.get("evaluated_points", len(predictions)),
                "mase_seasonality": val_metrics.get("mase_seasonality", mase_seasonality),
                "mase_seasonality_source": mase_seasonality_source,
                "mase_reference": dict(mase_reference),
                "metrics_computed_on": val_metrics.get("metrics_computed_on", "raw_validation_after_inverse_transform"),
                "validation_points_excluded_from_metrics": 0,
                "y_true_len": val_metrics.get("y_true_len", len(raw_y_true)),
                "y_pred_len": val_metrics.get("y_pred_len", len(predictions)),
                "mase": current_mase,
                "rmse": current_rmse,
                "mse": current_mse,
                "scaled_mse": scaled_mse,
                "r2": current_r2,
                "mae": current_mae,
                "mape": current_mape,
                "preprocessing_applied": preprocessing_steps > 0,
                "preprocessing_steps": preprocessing_steps,
                "preprocessing_transformations": preprocessing_transformations,
                "semantic_features_used": semantic_features_used,
                "target_mase_reached": current_mase <= self.finalize_mase_threshold,
                "target_transform": target_transform,
                "metrics_scale": "original",
            }

            state.setdefault("tested_models", []).append(result)
            state.setdefault("performance_history", []).append(
                {
                    "step": step,
                    "technical_decision_id": technical_decision_id,
                    "candidate_id": candidate_id,
                    "training_id": training_id,
                    "configuration_signature": configuration_signature,
                    "candidate_seed": candidate_seed,
                    "model": model,
                    "hyperparameters": hp,
                    "preprocessing_transformations": preprocessing_transformations,
                    "target_transform": target_transform,
                    "metrics_scale": "original",
                    "mase_reference": dict(mase_reference),
                    "mase": current_mase,
                    "rmse": current_rmse,
                    "mse": current_mse,
                    "scaled_mse": scaled_mse,
                    "r2": current_r2,
                    "mae": current_mae,
                    "mape": current_mape,
                    "semantic_features_used": semantic_features_used,
                    "predictions": [float(x) for x in predictions.tolist()],
                    "y_true_val": [float(x) for x in raw_y_true.tolist()],
                    "train_metrics": training.train_metrics,
                    "val_metrics": training.metrics,
                }
            )

            # Track best model using MASE.
            best = state.get("best_model")
            prev_score = float("inf")
            if isinstance(best, dict):
                if best.get("validation_metric") == "mse":
                    prev_score = float("inf")
                else:
                    prev_score = float(best.get("mase", best.get("validation_score", float("inf"))))

            if not isinstance(best, dict) or current_mase < prev_score:
                state["best_model"] = {
                    "selected_model": model,
                    "step": step,
                    "technical_decision_id": technical_decision_id,
                    "candidate_id": candidate_id,
                    "training_id": training_id,
                    "configuration_signature": configuration_signature,
                    "candidate_seed": candidate_seed,
                    "validation_score": current_mase,
                    "validation_metric": "mase",
                    "mase": current_mase,
                    "rmse": current_rmse,
                    "mse": current_mse,
                    "scaled_mse": scaled_mse,
                    "r2": current_r2,
                    "mae": current_mae,
                    "mape": current_mape,
                    "hyperparameters": hp,
                    "preprocessing_log": preprocessing_log,
                    "target_transform": target_transform,
                    "metrics_scale": "original",
                    "semantic_features_used": semantic_features_used,
                    "val_predictions": [float(x) for x in predictions.tolist()],
                    "y_true_val": [float(x) for x in raw_y_true.tolist()],
                    "train_metrics": training.train_metrics,
                    "val_metrics": training.metrics,
                }
                state["best_score"] = current_mase

            # Final-selection invariant: no semantic/fusion field participates.
            try:
                validation_winner = select_validation_winner(
                    state.get("performance_history", [])
                )
            except ValueError:
                validation_winner = None
            if validation_winner is not None:
                assert float(state["best_model"]["mase"]) == float(
                    validation_winner["mase"]
                )
                state["best_model_selection_audit"] = {
                    "best_model_is_argmin_mase": True,
                    "fusion_output_not_used_in_best_selection": True,
                    "semantic_output_not_used_in_best_selection": True,
                    "best_model_fusion_modified": False,
                    "best_model_semantically_modified": False,
                }

            state["target_mase_reached"] = current_mase <= self.finalize_mase_threshold
            state["training_error"] = None

            return state

        except Exception as exc:
            err_msg = str(exc)
            state["training_error"] = err_msg
            print(f"[TrainingAgent] Training failed for model '{model}' at step {step}: {err_msg}")
            self._record_failed_attempt(
                state=state,
                model=model,
                hp=hp,
                preprocessing=self._candidate_preprocessing_transformations(candidate),
                err_msg=err_msg,
                forecast_horizon=forecast_horizon,
                val_size=len(state.get("preprocessed_val_df", [])) if isinstance(state.get("preprocessed_val_df"), pd.DataFrame) else None,
            )
            return state
