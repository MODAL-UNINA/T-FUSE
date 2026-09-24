"""Final holdout evaluation and artifact reporting for T-FUSE."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from evaluation.experiment_logger import compute_train_only_minmax_test_metrics
from evaluation.final_protocol import align_test_predictions, explicit_final_failure
from utils.json_logger import save_json
from utils.metrics import build_mase_reference
from utils.reproducibility import array_fingerprint, tabular_training_fingerprints


@dataclass
class FinalEvaluationResult:
    """Explicit values produced by final held-out test evaluation."""

    test_metrics: dict[str, Any]
    test_predictions: list[float]
    train_metrics: dict[str, Any]
    final_test_report: dict[str, Any]
    evaluation_status: dict[str, Any]
    aligned_test_y_true: list[float]
    aligned_test_index: list[str]
    retraining_audit: dict[str, Any]
    evaluation_integrity: dict[str, Any]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def evaluate_final_test(
    *,
    trainer: Any,
    preprocessing_agent: Any,
    best_tech: Mapping[str, Any],
    winning_candidate_seed: int,
    winning_candidate_signature: str,
    final_full_df: pd.DataFrame,
    preprocessed_test_df: pd.DataFrame | None,
    combined_train_val: pd.DataFrame,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    current_val: pd.DataFrame,
    raw_test: pd.DataFrame | None,
    raw_test_original: pd.DataFrame | None,
    new_preprocessing_log: Mapping[str, Any],
    final_state: Mapping[str, Any],
    forecast_horizon: int,
) -> FinalEvaluationResult:
    """Evaluate the frozen winner on the final test partition.

    """
    test_metrics_final: dict[str, Any] = {}
    test_preds_final: list[float] = []
    train_metrics_final: dict[str, Any] = {}
    final_test_report: dict[str, Any] = {}
    final_evaluation_status = {
        "run_status": "final_evaluation_not_started",
        "error_type": None,
        "error_message": None,
    }
    aligned_test_y_true: list[float] = []
    aligned_test_index: list[str] = []
    final_retraining_audit: dict[str, Any] = {}

    if preprocessed_test_df is not None and not preprocessed_test_df.empty:
        evaluation_test_df = raw_test_original.copy(deep=True)
        if len(preprocessed_test_df) != len(raw_test):
            raise RuntimeError(
                "Preprocessing changed test cardinality; every test point must be evaluated"
            )
        if "date" in preprocessed_test_df and not pd.DatetimeIndex(
            pd.to_datetime(preprocessed_test_df["date"])
        ).equals(pd.DatetimeIndex(pd.to_datetime(evaluation_test_df["date"]))):
            raise RuntimeError("Preprocessing changed test timestamps")
        test_y_true = np.asarray(
            evaluation_test_df["target"].values, dtype=float
        )
        original_y_combined = np.asarray(
            combined_train_val["target"].values, dtype=float
        )

        try:
            # Determine MASE seasonality from existing state.
            data_summary = final_state.get("data_summary", {})
            base_metrics = data_summary.get("base_metrics", {})
            seasonality_detected = bool(
                data_summary.get(
                    "seasonality_detected",
                    base_metrics.get("seasonality_detected", False),
                )
            )
            seasonality_period = _safe_int(
                data_summary.get("seasonal_periods")
            )
            inferred_freq_metric = base_metrics.get(
                "inferred_freq"
            ) or base_metrics.get("frequency")
            suggested_lag = _safe_int(data_summary.get("suggested_lag"))
            y_train_len_final = len(final_full_df)

            final_mase_reference = build_mase_reference(
                original_y_combined,
                inferred_freq=inferred_freq_metric,
                seasonality_detected=seasonality_detected,
                seasonality_period=seasonality_period,
                suggested_lag=suggested_lag,
                source="raw_observed_train_plus_validation",
            )
            if not final_mase_reference["defined"]:
                raise ValueError(
                    "Final test MASE is undefined on raw train+validation observations."
                )
            mase_seasonality_final = int(final_mase_reference["period"])

            print(
                f"[METRICS] MASE denominator configuration: inferred_freq={inferred_freq_metric}, "
                f"seasonality_detected={seasonality_detected}, seasonality_period={seasonality_period}, "
                f"suggested_lag={suggested_lag}, y_train_len={y_train_len_final} -> "
                f"chosen_mase_seasonality = {mase_seasonality_final}"
            )

            print(
                f"[TEST_METRICS_INPUT] model={best_tech['selected_model']}, len_y_true={len(test_y_true)}, len_y_pred={len(preprocessed_test_df)}, len_y_train={len(original_y_combined)}, mase_seasonality={mase_seasonality_final}"
            )

            final_retraining_audit = trainer.prepare_candidate_training(
                winning_candidate_seed,
                training_phase="FINAL_RETRAIN",
                candidate_signature=winning_candidate_signature,
            )
            final_retraining_audit["data_fingerprints"] = (
                tabular_training_fingerprints(
                    final_full_df,
                    preprocessed_test_df,
                    raw_train_df=combined_train_val,
                    raw_eval_df=evaluation_test_df,
                )
            )
            test_preds_final_raw, test_metrics_final, train_metrics_final = (
                trainer._rolling_window_evaluate(
                    model_name=str(best_tech["selected_model"]),
                    hp=best_tech["hyperparameters"],
                    train_df=final_full_df,
                    eval_df=preprocessed_test_df,
                    y_true=test_y_true,
                    original_y_train=original_y_combined,
                    prep_agent=preprocessing_agent,
                    preprocessing_transformations=new_preprocessing_log.get(
                        "transformations", []
                    ),
                    raw_train_df=combined_train_val,
                    raw_eval_df=evaluation_test_df,
                    mase_denominator_source="raw_observed_train_plus_validation",
                    forecast_horizon=int(forecast_horizon),
                    mase_seasonality=mase_seasonality_final,
                    freq=inferred_freq_metric,
                )
            )
            final_origin_audits = list(
                test_metrics_final.get("validation_origin_audits", []) or []
            )
            if final_origin_audits and all(
                isinstance(item, dict) and "model_fit_count" in item
                for item in final_origin_audits
            ):
                final_retraining_audit["forecast_origin_audits"] = copy.deepcopy(
                    final_origin_audits
                )
                final_retraining_audit["model_fit_count"] = sum(
                    int(item.get("model_fit_count", 0))
                    for item in final_origin_audits
                    if isinstance(item, dict)
                )
            final_retraining_audit["model_fingerprints"] = {
                key: train_metrics_final.get(key)
                for key in (
                    "initial_weights_hash", "final_weights_hash", "best_epoch",
                    "early_stop_epoch", "train_loss_curve",
                    "validation_score_curve", "max_steps"
                )
                if train_metrics_final.get(key) is not None
            }
            final_retraining_audit["test_predictions_hash"] = array_fingerprint(
                test_preds_final_raw
            )
            configured_final_horizon = int(forecast_horizon)
            if len(test_preds_final_raw) != len(preprocessed_test_df):
                raise RuntimeError(
                    "invalid_forecast_protocol: final test returned "
                    f"{len(test_preds_final_raw)} predictions, expected "
                    f"{len(preprocessed_test_df)} test points"
                )
            final_retraining_audit.update(
                {
                    "configured_forecast_horizon": configured_final_horizon,
                    "effective_model_horizon": configured_final_horizon,
                    "train_points": int(len(final_full_df)),
                    "test_points": int(len(preprocessed_test_df)),
                    "candidate_signature": winning_candidate_signature,
                    "candidate_seed": winning_candidate_seed,
                    "preprocessing_signature": winning_candidate_signature,
                }
            )
            test_preds_final = [float(x) for x in test_preds_final_raw]
            y_true_aligned, y_pred_aligned = align_test_predictions(
                evaluation_test_df,
                test_preds_final,
                prediction_index=(
                    pd.DatetimeIndex(
                        pd.to_datetime(preprocessed_test_df["date"])
                    )
                    if "date" in preprocessed_test_df
                    else None
                ),
            )
            aligned_test_y_true = [
                float(value) for value in y_true_aligned.to_numpy()
            ]
            aligned_test_index = [
                value.isoformat() for value in y_pred_aligned.index
            ]
            test_metrics_final = dict(test_metrics_final)
            test_evaluation_protocol = test_metrics_final.pop(
                "validation_protocol", "unknown"
            )
            test_metrics_final.update({
                "model_type": str(best_tech["selected_model"]),
                "technical_decision_id": best_tech.get("technical_decision_id"),
                "configuration_signature": best_tech.get("configuration_signature"),
                "split": "test",
                "metrics_computed_on": "held_out_test",
                "evaluation_protocol": test_evaluation_protocol,
                "n_observations": len(test_y_true),
                "forecast_horizon": int(forecast_horizon),
                "refit_scope": "train_plus_validation",
            })

            print(
                f"[TEST_METRICS] model={best_tech['selected_model']}, mase={test_metrics_final.get('mase')}, rmse={test_metrics_final.get('rmse')}, mae={test_metrics_final.get('mae')}, mse={test_metrics_final.get('mse')}, r2={test_metrics_final.get('r2')}, mape={test_metrics_final.get('mape')}"
            )
            final_evaluation_status = {
                "run_status": "final_evaluation_succeeded",
                "error_type": None,
                "error_message": None,
            }
        except Exception as exc:
            final_evaluation_status = explicit_final_failure(exc)
            print(f"Error during final test evaluation: {exc}")
    else:
        final_evaluation_status = {
            "run_status": "final_evaluation_failed",
            "error_type": "EmptyTestHoldout",
            "error_message": "Final test holdout is empty after preprocessing.",
        }

    if final_evaluation_status.get("run_status") == "final_evaluation_succeeded":
        final_test_report = compute_train_only_minmax_test_metrics(
            raw_train["target"].to_numpy(dtype=float),
            aligned_test_y_true,
            test_preds_final,
        )

    validation_target_modified = not np.array_equal(
        pd.to_numeric(raw_val["target"], errors="coerce").to_numpy(
            dtype=float
        ),
        pd.to_numeric(current_val["target"], errors="coerce").to_numpy(
            dtype=float
        ),
        equal_nan=True,
    )
    test_target_modified = bool(
        isinstance(raw_test_original, pd.DataFrame)
        and isinstance(raw_test, pd.DataFrame)
        and not np.array_equal(
            pd.to_numeric(
                raw_test_original["target"], errors="coerce"
            ).to_numpy(dtype=float),
            pd.to_numeric(raw_test["target"], errors="coerce").to_numpy(
                dtype=float
            ),
            equal_nan=True,
        )
    )
    if validation_target_modified or test_target_modified:
        raise RuntimeError("Evaluation ground truth was modified")
    evaluation_integrity = {
        "replace_outliers": copy.deepcopy(
            new_preprocessing_log.get("replace_outliers", {})
        ),
        "outlier_detection_scope": new_preprocessing_log.get(
            "replace_outliers", {}
        ).get("detection_scope", "not_applied"),
        "validation_target_modified": False,
        "test_target_modified": False,
        "validation_points_excluded_from_metrics": 0,
        "test_points_excluded_from_metrics": 0,
    }
    return FinalEvaluationResult(
        test_metrics=test_metrics_final,
        test_predictions=test_preds_final,
        train_metrics=train_metrics_final,
        final_test_report=final_test_report,
        evaluation_status=final_evaluation_status,
        aligned_test_y_true=aligned_test_y_true,
        aligned_test_index=aligned_test_index,
        retraining_audit=final_retraining_audit,
        evaluation_integrity=evaluation_integrity,
    )


_FINAL_SUMMARY_KEYS = (
    "series_id", "run_id", "influence_mode", "forecast_horizon",
    "winning_candidate_seed", "final_retraining_audit",
    "dataset_fingerprint", "candidate_catalog_fingerprint",
    "split_metadata", "semantic_split_meta",
    "semantic_document_filtering", "timemmd_metadata",
    "experiment_manifest",
    "rolling_origin_count",
    "validation_size", "validation_ratio_effective", "train_size",
    "test_size", "train_end", "validation_start", "validation_end",
    "test_start", "test_end",
    "mase_definition",
    "external_covariates_used", "external_covariate_policy",
    "allowed_numeric_inputs", "textual_evidence_used",
    "final_model_feature_columns", "calendar_feature_columns",
    "calendar_feature_source", "external_numeric_columns_excluded_at_load",
    "final_model_training_columns", "numeric_input_audit",
    "seed", "run_output_root",
    "semantic_reliability", "semantic_discriminability",
    "cross_modal_conflict", "search_breadth", "families_explored",
    "family_coverage_count", "best_validation_model",
    "best_validation_mase", "winning_model",
    "winning_candidate_id", "forecast_strategy",
    "best_model", "best_hyperparameters", "val_mase", "val_rmse",
    "val_mse", "val_scaled_mse", "val_mae", "val_mape", "val_r2",
    "test_rmse", "test_mse", "test_mae", "final_metric_metadata",
    "stop_reason",
    "valid_training_count", "actual_valid_trainings",
    "maximum_valid_trainings", "minimum_valid_trainings_before_stop",
    "early_stop_used",
    "budget_realization_status", "influence_components",
    "evidence_fusion", "search_decision_state",
    "candidate_error_counts", "technical_retry_summary",
    "final_evaluation_status", "final_test_evaluation_count",
    "execution_time_seconds", "target_mase_reached",
    "extreme_outliers_detected", "llm_usage",
    "semantic_context", "technical_compliance_trace",
    "evaluation_integrity", "outlier_detection_scope",
    "validation_target_modified", "test_target_modified",
    "validation_points_excluded_from_metrics",
    "test_points_excluded_from_metrics",
    "log_transform_recommended", "log_transform_supported",
    "log_candidate_required", "log_candidate_evaluated",
    "raw_candidate_evaluated", "raw_candidates_evaluated",
    "log_candidates_evaluated", "raw_exploration_pending",
    "log_exploration_pending", "evaluated_preprocessing_by_model",
    "raw_log_evaluated_by_model", "best_candidate_uses_log",
    "log_transform_audit", "mase_denominator_source",
    "mase_denominator_common_across_candidates",
    "mase_denominator_value", "mase_reference_training_size",
    "search_mase_reference",
)


def write_final_outputs(
    *,
    output_root: Path,
    final_result: Mapping[str, Any],
    performance_history: list[Any],
    search_trace: list[Any],
    validation_predictions: list[float],
    validation_ground_truth: list[float],
    test_predictions: list[float],
    test_ground_truth: list[float],
    test_index: list[str],
    forecast_strategy: str,
) -> None:
    """Persist final artifacts with the established paths and JSON schemas."""
    final_summary = {
        key: final_result.get(key)
        for key in _FINAL_SUMMARY_KEYS
    }
    save_json(output_root / "final" / "final_results.json", final_summary)
    save_json(output_root / "final" / "performance_history.json", performance_history)
    save_json(output_root / "final" / "search_trace.json", search_trace)
    save_json(
        output_root / "final" / "predictions.json",
        {
            "val_predictions": validation_predictions,
            "y_true_val": validation_ground_truth,
            "test_predictions": test_predictions,
            "y_true_test": test_ground_truth,
            "test_index": test_index,
            "forecast_strategy": forecast_strategy,
        },
    )


__all__ = [
    "FinalEvaluationResult",
    "evaluate_final_test",
    "write_final_outputs",
]
