"""Training orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

from models.model_library import HORIZON_STRUCTURAL_MODELS, ModelLibrary
from utils.metrics import mae, mape, mase, mase_scale, mse, r2, rmse
from utils.numeric_input_policy import (
    CALENDAR_FEATURE_COLUMNS,
    assert_raw_series_contract,
    validated_model_feature_columns,
)


@dataclass
class TrainingResult:
    """Container for training outputs."""

    model_name: str
    hyperparameters: Dict[str, object]
    metrics: Dict[str, float]
    scaler_used: str | None
    predictions: np.ndarray = field(repr=False)
    train_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trainer:
    """Trainer that evaluates candidate models."""

    random_state: int = 42

    def __post_init__(self) -> None:
        self.library = ModelLibrary(random_state=self.random_state)

    def prepare_candidate_training(
        self,
        candidate_seed: int,
        *,
        training_phase: str,
        candidate_signature: str,
    ) -> Dict[str, Any]:
        """Make one training invocation independent of prior search history."""
        from core.candidate_catalog import set_deterministic_seed

        seed = int(candidate_seed)
        audit = set_deterministic_seed(seed)
        audit.update({
            "candidate_signature": str(candidate_signature),
            "candidate_seed": seed,
            "training_phase": str(training_phase),
        })
        self.random_state = seed
        self.library.random_state = seed
        self.determinism_audit = audit
        print(
            "[DETERMINISM] "
            f"training_phase={training_phase}, "
            f"candidate_signature={candidate_signature}, candidate_seed={seed}, "
            f"effective_python_seed={audit.get('effective_python_seed')}, "
            f"effective_numpy_seed={audit.get('effective_numpy_seed')}, "
            f"effective_torch_seed={audit.get('effective_torch_seed')}, "
            f"effective_cuda_seed={audit.get('effective_cuda_seed')}"
        )
        return dict(audit)

    @staticmethod
    def _add_mase_trace(
        metrics: Dict[str, Any],
        mase_y_train: np.ndarray,
        mase_seasonality: int,
        source: str,
        spikes_replaced: int,
    ) -> None:
        """Record the original-scale training series used by the MASE scale."""
        values = np.asarray(mase_y_train, dtype=float).reshape(-1)
        metrics.update({
            "mase_denominator_source": source,
            "mase_denominator_common_across_candidates": True,
            "mase_reference_training_size": int(len(values)),
            "mase_training_size": int(len(values)),
            "mase_seasonal_period": int(mase_seasonality),
            "mase_denominator_value": mase_scale(
                values, seasonality=mase_seasonality
            ),
        })



    _GLOBAL_MODELS: frozenset[str] = frozenset(
        {
            "arima",
            "sarima",
            "ets",
            "prophet",
        }
    )

    _HORIZON_STRUCTURAL_MODELS: frozenset[str] = frozenset(
        name.strip().lower() for name in HORIZON_STRUCTURAL_MODELS
    )


    _TABULAR_FEATURE_MODELS: frozenset[str] = frozenset(
        {
            "mlp",
            "xgboost",
            "randomforest",
            "lightgbm",
            "catboost",
            "elasticnet",
            "knn",
            "svr",
        }
    )

    @staticmethod
    def _make_windowed_xy(
        y: np.ndarray, lookback: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create (X, y) arrays from a 1-D series using sliding windows."""
        X, Y = [], []
        for i in range(len(y) - lookback - horizon + 1):
            X.append(y[i : i + lookback])
            Y.append(y[i + lookback : i + lookback + horizon])
        if not X:
            return np.array([]).reshape(0, lookback), np.array([]).reshape(0, horizon)
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    @staticmethod
    def _temporal_inner_validation_indices(
        n_samples: int,
        *,
        fraction: float = 0.2,
        minimum_inner_samples: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ordered train/inner-validation indices from training only."""
        if n_samples < minimum_inner_samples + 2:
            return np.arange(n_samples, dtype=int), np.array([], dtype=int)
        inner_size = max(minimum_inner_samples, int(np.ceil(n_samples * fraction)))
        inner_size = min(inner_size, n_samples - 1)
        split = n_samples - inner_size
        return np.arange(split, dtype=int), np.arange(split, n_samples, dtype=int)

    def _fit_tree_with_inner_validation(
        self,
        *,
        base: object,
        key: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[object, dict]:
        """Tune stopping rounds on a temporal suffix of train, then refit all train."""
        import copy
        from sklearn.multioutput import MultiOutputRegressor

        fit_idx, inner_idx = self._temporal_inner_validation_indices(len(X_train))
        model = MultiOutputRegressor(base)
        model.estimators_ = []
        metadata = {
            "inner_validation_source": "train_only",
            "inner_validation_protocol": "temporal_suffix",
            "inner_train_start": int(fit_idx[0]) if len(fit_idx) else None,
            "inner_train_end": int(fit_idx[-1]) if len(fit_idx) else None,
            "inner_validation_start": int(inner_idx[0]) if len(inner_idx) else None,
            "inner_validation_end": int(inner_idx[-1]) if len(inner_idx) else None,
            "inner_train_samples": len(fit_idx),
            "inner_validation_samples": len(inner_idx),
            "outer_validation_used_in_fit": False,
            "final_refit_scope": "all_training_windows",
        }

        for output_index in range(y_train.shape[1]):
            final_estimator = copy.deepcopy(base)
            if len(inner_idx):
                tuned = copy.deepcopy(base)
                if key in {"xgboost", "catboost"}:
                    tuned.set_params(early_stopping_rounds=10)
                    tuned.fit(
                        X_train[fit_idx],
                        y_train[fit_idx, output_index],
                        eval_set=[
                            (
                                X_train[inner_idx],
                                y_train[inner_idx, output_index],
                            )
                        ],
                        verbose=False,
                    )
                elif key == "lightgbm":
                    from lightgbm import early_stopping, log_evaluation

                    tuned.fit(
                        X_train[fit_idx],
                        y_train[fit_idx, output_index],
                        eval_set=[
                            (
                                X_train[inner_idx],
                                y_train[inner_idx, output_index],
                            )
                        ],
                        callbacks=[
                            early_stopping(10, verbose=False),
                            log_evaluation(0),
                        ],
                    )
                best_iteration = getattr(
                    tuned,
                    "best_iteration_",
                    getattr(tuned, "best_iteration", None),
                )
                if best_iteration is not None:
                    iteration_parameter = (
                        "iterations" if key == "catboost" else "n_estimators"
                    )
                    final_estimator.set_params(
                        **{iteration_parameter: max(1, int(best_iteration) + 1)}
                    )
            final_estimator.fit(X_train, y_train[:, output_index])
            model.estimators_.append(final_estimator)
        model.n_features_in_ = X_train.shape[1]
        return model, metadata

    def _windowed_single_train_evaluate(
        self,
        model_name: str,
        hp: dict,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        y_true: np.ndarray,
        original_y_train: np.ndarray,
        prep_agent: object | None,
        preprocessing_transformations: list | None,
        forecast_horizon: int,
        mase_y_train: np.ndarray,
        mase_spikes_replaced: int,
        mase_denominator_source: str,
        mase_seasonality: int = 1,
        freq: str = "D"
    ) -> tuple[np.ndarray, dict]:
        """Train ONCE on windowed train data, evaluate window-by-window on eval.

        """
        key = model_name.strip().lower()
        lookback = int(hp.get("lag", hp.get("lookback", forecast_horizon)))
        lookback = max(2, lookback)

        y_train_arr = np.asarray(train_df["target"].values, dtype=float)
        y_eval_arr = np.asarray(eval_df["target"].values, dtype=float)

        X_train, y_win_train = self._make_windowed_xy(
            y_train_arr, lookback, forecast_horizon
        )

        n_eval_windows = (len(y_eval_arr) + forecast_horizon - 1) // forecast_horizon
        print(
            f"\n[WINDOW_EVAL] model={model_name}, val_len={len(y_eval_arr)}, "
            f"predicted_points={len(y_eval_arr)}, protocol=windowed_single_train"
        )

        if len(X_train) == 0:
            return self._rolling_window_evaluate(
                model_name,
                hp,
                train_df,
                eval_df,
                y_true,
                original_y_train,
                prep_agent,
                preprocessing_transformations,
                train_df,
                {"enabled": False},
                forecast_horizon,
                raw_eval_df=eval_df.assign(target=y_true),
                mase_denominator_source=mase_denominator_source,
                mase_seasonality=mase_seasonality,
                freq=freq
            )

        from sklearn.multioutput import MultiOutputRegressor

        try:
            base = self.library.get_base_estimator(key, hp)
            model = MultiOutputRegressor(base)
        except ValueError:
            base = None
            model = None

        if base is not None:
            inner_validation_info = {}
            if key in ("xgboost", "catboost", "lightgbm"):
                model, inner_validation_info = self._fit_tree_with_inner_validation(
                    base=base,
                    key=key,
                    X_train=X_train,
                    y_train=y_win_train,
                )
            elif key == "mlp":

                minimum_inner_samples = 5
                minimum_fit_samples = 2
                if len(X_train) >= minimum_inner_samples + minimum_fit_samples:
                    inner_samples = max(
                        minimum_inner_samples,
                        int(np.ceil(len(X_train) * 0.20)),
                    )
                    inner_samples = min(
                        inner_samples,
                        len(X_train) - minimum_fit_samples,
                    )
                    validation_fraction = inner_samples / len(X_train)
                    base.set_params(
                        early_stopping=True,
                        validation_fraction=validation_fraction,
                        n_iter_no_change=10,
                    )
                    inner_validation_info = {
                        "inner_validation_source": "train_only",
                        "inner_validation_protocol": (
                            "sklearn_internal_training_holdout"
                        ),
                        "inner_train_samples": len(X_train) - inner_samples,
                        "inner_validation_samples": inner_samples,
                        "outer_validation_used_in_fit": False,
                        "final_refit_scope": "training_windows_with_internal_holdout",
                    }
                else:
                    base.set_params(early_stopping=False)
                    inner_validation_info = {
                        "inner_validation_source": "not_used",
                        "inner_validation_protocol": (
                            "insufficient_training_windows"
                        ),
                        "inner_train_samples": len(X_train),
                        "inner_validation_samples": 0,
                        "outer_validation_used_in_fit": False,
                        "final_refit_scope": "all_training_windows",
                    }
                model = MultiOutputRegressor(base)
                model.fit(X_train, y_win_train)
            else:
                model.fit(X_train, y_win_train)
        else:
            preds_full = self.library.train_and_predict(
                model_name,
                train_df,
                eval_df,
                hp,
                forecast_horizon=forecast_horizon,
                preprocessing_transformations=preprocessing_transformations,
                freq=freq
            )
            if prep_agent and preprocessing_transformations:
                preds_full = prep_agent.inverse_transform(
                    preds_full, preprocessing_transformations, original_y_train
                )
            _y = y_true[: len(preds_full)]
            _p = preds_full[: len(_y)]
            metrics = {
                "mase": float(
                    mase(_y, _p, mase_y_train, seasonality=mase_seasonality)
                ),
                "rmse": float(rmse(_y, _p)),
                "mse": float(mse(_y, _p)),
                "r2": float(r2(_y, _p)),
                "mae": float(mae(_y, _p)),
                "mape": float(mape(_y, _p)),
            }
            self._add_mase_trace(
                metrics, mase_y_train, mase_seasonality,
                mase_denominator_source, mase_spikes_replaced,
            )
            train_metrics = getattr(self.library, "last_train_info", {})
            return np.asarray(preds_full, dtype=float), metrics, train_metrics

        train_preds = model.predict(X_train)
        train_mse = float(mse(y_win_train, train_preds))
        train_rmse = float(rmse(y_win_train, train_preds))
        train_mae = float(mae(y_win_train, train_preds))
        train_mape = float(mape(y_win_train, train_preds))
        train_r2 = float(r2(y_win_train, train_preds))

        last_loss = train_mse
        if key == "mlp" and hasattr(model, "estimators_"):
            losses = [
                est.loss_curve_[-1]
                for est in model.estimators_
                if hasattr(est, "loss_curve_") and est.loss_curve_
            ]
            if losses:
                last_loss = sum(losses) / len(losses)

        train_metrics = {
            "loss": last_loss,
            "mse": train_mse,
            "rmse": train_rmse,
            "mae": train_mae,
            "mape": train_mape,
            "r2": train_r2,
            **(
                inner_validation_info
                if "inner_validation_info" in locals()
                else {
                    "inner_validation_source": "not_applicable",
                    "outer_validation_used_in_fit": False,
                }
            ),
        }


        context = list(y_train_arr[-lookback:])
        all_preds: list[float] = []
        all_scaled_preds: list[float] = []

        for w in range(n_eval_windows):
            start = w * forecast_horizon
            x_input = np.asarray(context[-lookback:], dtype=float).reshape(1, -1)
            win_pred = model.predict(x_input)[0]  # shape: (horizon,)
            win_pred = np.asarray(win_pred, dtype=float)

            if prep_agent and preprocessing_transformations:
                win_pred_inv = prep_agent.inverse_transform(
                    win_pred,
                    preprocessing_transformations,
                    (
                        np.concatenate([original_y_train, np.array(all_preds)])
                        if all_preds
                        else original_y_train
                    ),
                )
            else:
                win_pred_inv = win_pred

            step_size = min(forecast_horizon, len(y_eval_arr) - start)

            win_pred_sliced = win_pred[:step_size]
            win_pred_inv_sliced = win_pred_inv[:step_size]

            all_scaled_preds.extend(win_pred_sliced.tolist())
            all_preds.extend(win_pred_inv_sliced.tolist())

            win_true = y_true[start : start + step_size]
            if len(win_true) == len(win_pred_inv_sliced):
                win_mse = float(mse(win_true, win_pred_inv_sliced))
                print(f"  Window {w + 1}/{n_eval_windows}: MSE={win_mse:.4f}")

            context.extend(y_eval_arr[start : start + step_size].tolist())

        preds = np.asarray(all_preds, dtype=float)
        scaled_preds_arr = np.asarray(all_scaled_preds, dtype=float)

        if len(y_true) != len(preds):
            raise ValueError(
                f"Length mismatch: len(y_true)={len(y_true)} != len(preds)={len(preds)}"
            )

        from utils.metrics import compute_forecast_metrics

        print(
            f"[METRICS_INPUT] model={model_name}, validation_protocol=rolling_origin_one_fit, len_y_true={len(y_true)}, len_y_pred={len(preds)}, len_y_train={len(original_y_train)}, evaluated_points={len(preds)}"
        )

        split_metrics = compute_forecast_metrics(
            y_true=y_true,
            y_pred=preds,
            y_train=mase_y_train,
            mase_seasonality=mase_seasonality,
            model_name=model_name,
            validation_protocol="rolling_origin_one_fit",
        )
        split_metrics["scaled_mse"] = float(mse(y_eval_arr, scaled_preds_arr))
        self._add_mase_trace(
            split_metrics, mase_y_train, mase_seasonality,
            mase_denominator_source, mase_spikes_replaced,
        )

        print(
            f"[METRICS] model={model_name}, mase={split_metrics['mase']:.4f}, rmse={split_metrics['rmse']:.4f}, mae={split_metrics['mae']:.4f}, mse={split_metrics['mse']:.4f}, r2={split_metrics['r2']:.4f}, mape={split_metrics['mape']:.4f}, evaluated_points={len(preds)}"
        )
        print(
            f"[METRICS_SCALE] model={model_name}, metrics_scale=original, inverse_transform_applied={True if prep_agent and preprocessing_transformations else False}"
        )
        print(f"[MASE] chosen_seasonality={mase_seasonality}, computed_mase={split_metrics['mase']:.4f}")

        split_metrics["validation_protocol"] = "rolling_origin_one_fit"
        split_metrics["forecast_horizon"] = forecast_horizon
        split_metrics["configured_forecast_horizon"] = forecast_horizon
        split_metrics["effective_model_horizon"] = forecast_horizon
        split_metrics["evaluation_window_size"] = forecast_horizon
        split_metrics["number_of_validation_origins"] = n_eval_windows
        split_metrics["validation_effective_horizons"] = [
            forecast_horizon for _ in range(n_eval_windows)
        ]
        split_metrics["horizon_consistency"] = (
            len(preds) == len(y_eval_arr)
        )
        split_metrics["evaluated_points"] = len(preds)
        split_metrics["mase_seasonality"] = mase_seasonality
        split_metrics["metrics_computed_on"] = "full_validation"
        split_metrics["y_true_len"] = len(y_true)
        split_metrics["y_pred_len"] = len(preds)

        return preds, split_metrics, train_metrics

    def _tabular_feature_evaluate(
        self,
        model_name: str,
        hp: Dict[str, object],
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        y_true: np.ndarray,
        original_y_train: np.ndarray,
        prep_agent: object | None,
        preprocessing_transformations: list | None,
        forecast_horizon: int,
        mase_y_train: np.ndarray,
        mase_spikes_replaced: int,
        mase_denominator_source: str,
        mase_seasonality: int = 1,
        freq: str = "D",
    ) -> tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """Evaluate tabular models on the exact preprocessed feature space.

        """
        feature_columns = validated_model_feature_columns(
            train_df, context=f"Trainer input for {model_name}"
        )
        if not feature_columns:
            raise ValueError(
                f"{model_name} requires preprocessed numeric feature columns; "
                "Trainer will not silently rebuild target-only lag windows."
            )
        missing_eval = [c for c in feature_columns if c not in eval_df.columns]
        if missing_eval:
            raise ValueError(
                f"{model_name} evaluation data is missing feature columns: {missing_eval}"
            )

        scaled_predictions = np.asarray(
            self.library.train_and_predict(
                model_name,
                train_df,
                eval_df,
                hp,
                forecast_horizon=forecast_horizon,
                preprocessing_transformations=preprocessing_transformations,
                freq=freq,
            ),
            dtype=float,
        ).reshape(-1)

        if len(scaled_predictions) != len(eval_df):
            raise ValueError(
                f"{model_name} returned {len(scaled_predictions)} predictions for "
                f"{len(eval_df)} evaluation rows."
            )

        if prep_agent and preprocessing_transformations:
            predictions = np.asarray(
                prep_agent.inverse_transform(
                    scaled_predictions,
                    preprocessing_transformations,
                    original_y_train,
                ),
                dtype=float,
            ).reshape(-1)
        else:
            predictions = scaled_predictions

        if len(y_true) != len(predictions):
            raise ValueError(
                f"Length mismatch: len(y_true)={len(y_true)} != "
                f"len(predictions)={len(predictions)}"
            )

        from utils.metrics import compute_forecast_metrics

        metrics = compute_forecast_metrics(
            y_true=y_true,
            y_pred=predictions,
            y_train=mase_y_train,
            mase_seasonality=mase_seasonality,
            model_name=model_name,
            validation_protocol="preprocessed_tabular_one_fit",
        )
        self._add_mase_trace(
            metrics, mase_y_train, mase_seasonality,
            mase_denominator_source, mase_spikes_replaced,
        )
        eval_target_scaled = np.asarray(eval_df["target"].values, dtype=float)
        metrics["scaled_mse"] = float(mse(eval_target_scaled, scaled_predictions))
        metrics.update(
            {
                "validation_protocol": "preprocessed_tabular_one_fit",
                "forecast_horizon": forecast_horizon,
                "configured_forecast_horizon": forecast_horizon,
                "effective_model_horizon": forecast_horizon,
                "evaluation_window_size": forecast_horizon,
                "number_of_validation_origins": len(eval_df) // forecast_horizon,
                "validation_effective_horizons": [
                    forecast_horizon
                    for _ in range(len(eval_df) // forecast_horizon)
                ],
                "horizon_consistency": (
                    len(eval_df) % forecast_horizon == 0
                    and len(predictions) == len(eval_df)
                ),
                "evaluated_points": len(predictions),
                "mase_seasonality": mase_seasonality,
                "metrics_computed_on": "full_validation",
                "y_true_len": len(y_true),
                "y_pred_len": len(predictions),
                "feature_representation": "preprocessed_numeric_features",
                "feature_columns": list(feature_columns),
            }
        )
        train_metrics = dict(getattr(self.library, "last_train_info", {}) or {})
        train_metrics["feature_representation"] = "preprocessed_numeric_features"
        train_metrics["feature_columns"] = list(feature_columns)
        return predictions, metrics, train_metrics

    def _causal_tabular_feature_evaluate(
        self,
        model_name: str,
        hp: Dict[str, object],
        raw_train_df: pd.DataFrame,
        raw_eval_df: pd.DataFrame,
        y_true: np.ndarray,
        original_y_train: np.ndarray,
        prep_agent: object,
        preprocessing_transformations: list,
        forecast_horizon: int,
        mase_denominator_source: str,
        mase_seasonality: int,
        freq: str,
    ) -> tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """Causal rolling-origin evaluation for target-derived tabular features.

        """
        from agents.preprocessing_agent import PreprocessingPlan
        from core.candidate_catalog import set_deterministic_seed
        from utils.metrics import compute_forecast_metrics

        assert_raw_series_contract(
            raw_train_df, context=f"{model_name} raw training history"
        )
        assert_raw_series_contract(
            raw_eval_df, context=f"{model_name} raw evaluation partition"
        )

        if len(raw_eval_df) % forecast_horizon:
            remainder = len(raw_eval_df) % forecast_horizon
            raise RuntimeError(
                f"Invalid forecast protocol for {model_name}: evaluation contains "
                f"a partial origin of {remainder} points"
            )

        determinism = dict(getattr(self, "determinism_audit", {}) or {})
        candidate_seed = int(determinism.get("candidate_seed", self.random_state))
        raw_history = raw_train_df.copy(deep=True)
        predictions: list[float] = []
        origin_audits: list[Dict[str, Any]] = []
        mase_per_origin: list[float] = []
        mae_per_origin: list[float] = []
        last_train_metrics: Dict[str, float] = {}

        for start in range(0, len(raw_eval_df), forecast_horizon):
            origin_index = len(origin_audits)
            raw_chunk = raw_eval_df.iloc[start : start + forecast_horizon].copy(deep=True)
            fold_plan = PreprocessingPlan(
                transformations=prep_agent.specifications_for_refit(
                    preprocessing_transformations
                )
            )
            history_for_fit, fold_transformations = prep_agent._apply_transformations(
                raw_history, fold_plan
            )
            fold_outlier_log = next(
                (item for item in fold_transformations if item.get("name") == "replace_outliers"),
                {},
            )
            recursive_history = raw_history.copy(deep=True)
            origin_predictions: list[float] = []

            set_deterministic_seed(candidate_seed)
            model_fit_count = 0
            fitted_model = self.library.fit_tabular_model(
                model_name,
                history_for_fit,
                hp,
                preprocessing_transformations=fold_transformations,
            )
            model_fit_count += 1
            last_train_metrics = dict(
                getattr(self.library, "last_train_info", {}) or {}
            )

            for offset in range(forecast_horizon):
                future_raw = raw_chunk.iloc[[offset]].copy(deep=True)
                future_raw.loc[:, "target"] = np.nan
                future_for_model = prep_agent.apply_transformations_from_log(
                    future_raw,
                    fold_transformations,
                    start_idx=len(recursive_history),
                    history_df=recursive_history,
                )
                scaled = np.asarray(
                    self.library.predict_tabular_model(
                        fitted_model,
                        future_for_model,
                    ),
                    dtype=float,
                ).reshape(-1)
                if len(scaled) != 1:
                    raise RuntimeError(
                        f"Invalid causal feature protocol for {model_name}: "
                        f"expected one recursive prediction, received {len(scaled)}"
                    )
                raw_prediction = float(
                    prep_agent.inverse_transform(
                        scaled,
                        fold_transformations,
                        recursive_history["target"].to_numpy(dtype=float),
                    )[0]
                )
                predictions.append(raw_prediction)
                origin_predictions.append(raw_prediction)
                predicted_row = raw_chunk.iloc[[offset]].copy(deep=True)
                predicted_row.loc[:, "target"] = raw_prediction
                recursive_history = pd.concat(
                    [recursive_history, predicted_row], ignore_index=True
                )

            if model_fit_count != 1:
                raise RuntimeError(
                    f"Invalid causal feature protocol for {model_name}: origin "
                    f"{origin_index} performed {model_fit_count} model fits; expected 1."
                )

            origin_array = np.asarray(origin_predictions, dtype=float)
            expected = np.asarray(
                y_true[start : start + forecast_horizon], dtype=float
            )
            mase_per_origin.append(
                float(
                    mase(
                        expected,
                        origin_array,
                        np.asarray(original_y_train, dtype=float),
                        seasonality=mase_seasonality,
                    )
                )
            )
            mae_per_origin.append(float(mae(expected, origin_array)))
            origin_audits.append(
                {
                    "origin_index": origin_index,
                    "train_start": pd.Timestamp(raw_history["date"].iloc[0]).isoformat(),
                    "train_end": pd.Timestamp(raw_history["date"].iloc[-1]).isoformat(),
                    "forecast_start": pd.Timestamp(raw_chunk["date"].iloc[0]).isoformat(),
                    "forecast_end": pd.Timestamp(raw_chunk["date"].iloc[-1]).isoformat(),
                    "configured_forecast_horizon": int(forecast_horizon),
                    "effective_model_horizon": int(forecast_horizon),
                    "evaluation_window_size": int(len(raw_chunk)),
                    "candidate_signature": str(determinism.get("candidate_signature", "")),
                    "candidate_seed": candidate_seed,
                    "model_name": model_name,
                    "preprocessing_signature": str(determinism.get("candidate_signature", "")),
                    "model_fit_count": model_fit_count,
                    "recursive_prediction_steps": int(len(origin_predictions)),
                    "target_derived_feature_policy": "recursive_predicted_history",
                    "observed_targets_used_within_origin": False,
                    "external_covariates_used": False,
                    "feature_columns_used": list(fitted_model.feature_columns),
                    "calendar_feature_source": (
                        "timestamp_only"
                        if any(
                            column in CALENDAR_FEATURE_COLUMNS
                            for column in fitted_model.feature_columns
                        )
                        else "not_used"
                    ),
                    "replace_outliers": fold_outlier_log,
                    "validation_target_modified": False,
                }
            )
            raw_history = pd.concat([raw_history, raw_chunk], ignore_index=True)

        predictions_array = np.asarray(predictions, dtype=float)
        if len(predictions_array) != len(y_true):
            raise RuntimeError(
                f"Invalid causal feature protocol for {model_name}: prediction "
                f"cardinality={len(predictions_array)}, expected={len(y_true)}"
            )
        metrics = compute_forecast_metrics(
            y_true=y_true,
            y_pred=predictions_array,
            y_train=np.asarray(original_y_train, dtype=float),
            mase_seasonality=mase_seasonality,
            model_name=model_name,
            validation_protocol="rolling_origin_refit_causal_features",
        )
        self._add_mase_trace(
            metrics,
            np.asarray(original_y_train, dtype=float),
            mase_seasonality,
            mase_denominator_source,
            0,
        )
        metrics.update(
            {
                "validation_protocol": "rolling_origin_refit_causal_features",
                "forecast_horizon": forecast_horizon,
                "configured_forecast_horizon": forecast_horizon,
                "effective_model_horizon": forecast_horizon,
                "evaluation_window_size": forecast_horizon,
                "number_of_validation_origins": len(origin_audits),
                "validation_effective_horizons": [
                    forecast_horizon for _ in origin_audits
                ],
                "validation_origin_audits": origin_audits,
                "validation_mase_per_origin": mase_per_origin,
                "validation_mae_per_origin": mae_per_origin,
                "horizon_consistency": True,
                "evaluated_points": len(predictions_array),
                "mase_seasonality": mase_seasonality,
                "metrics_computed_on": "full_validation",
                "y_true_len": len(y_true),
                "y_pred_len": len(predictions_array),
                "target_derived_feature_policy": "recursive_predicted_history",
                "model_fit_count": sum(
                    int(item["model_fit_count"]) for item in origin_audits
                ),
                "recursive_prediction_steps": len(predictions_array),
                "observed_targets_used_within_origin": False,
                "external_covariates_used": False,
                "feature_columns_used": list(
                    origin_audits[-1].get("feature_columns_used", [])
                    if origin_audits else []
                ),
            }
        )
        return predictions_array, metrics, last_train_metrics

    def _rolling_window_evaluate(
        self,
        model_name: str,
        hp: Dict[str, object],
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        y_true: np.ndarray,
        original_y_train: np.ndarray,
        prep_agent: object | None,
        preprocessing_transformations: list | None,
        raw_train_df: pd.DataFrame | None,
        forecast_horizon: int,
        raw_eval_df: pd.DataFrame | None = None,
        mase_denominator_source: str = "raw_observed_training",
        mase_seasonality: int = 1,
        freq: str = "D"
    ) -> tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """Dispatch to the correct evaluation strategy based on model type.
        """
        from agents.preprocessing_agent import PreprocessingPlan

        if isinstance(raw_train_df, pd.DataFrame):
            assert_raw_series_contract(
                raw_train_df, context=f"{model_name} raw training partition"
            )
        if isinstance(raw_eval_df, pd.DataFrame):
            assert_raw_series_contract(
                raw_eval_df, context=f"{model_name} raw evaluation partition"
            )
        validated_model_feature_columns(
            train_df, context=f"{model_name} preprocessed training partition"
        )
        validated_model_feature_columns(
            eval_df, context=f"{model_name} preprocessed evaluation partition"
        )

        key = model_name.strip().lower()
        is_global = key in self._GLOBAL_MODELS
        has_structural_horizon = key in self._HORIZON_STRUCTURAL_MODELS
        uses_rolling_refit = is_global or has_structural_horizon

        train_df = train_df.copy()
        if train_df["target"].isna().any():
            train_df["target"] = (
                pd.Series(train_df["target"].values)
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
                .to_numpy()
            )

        eval_df = eval_df.copy()
        if eval_df["target"].isna().any():
            missing = int(eval_df["target"].isna().sum())
            raise ValueError(
                f"Evaluation target contains {missing} missing values. "
                "Validation/test labels are never future-interpolated."
            )

        raw_evaluation_df = (
            raw_eval_df.copy(deep=True)
            if isinstance(raw_eval_df, pd.DataFrame)
            else eval_df.assign(target=np.asarray(y_true, dtype=float)).copy(deep=True)
        )
        if len(raw_evaluation_df) != len(eval_df):
            raise ValueError("Raw evaluation and model evaluation cardinality differ")
        raw_eval_target = raw_evaluation_df["target"].to_numpy(dtype=float)
        if not np.array_equal(raw_eval_target, np.asarray(y_true, dtype=float), equal_nan=True):
            raise RuntimeError("Raw evaluation target differs from metric ground truth")


        mase_y_train = np.asarray(original_y_train, dtype=float).reshape(-1)
        mase_spikes_replaced = 0

        has_target_derived_features = any(
            isinstance(step, dict)
            and step.get("name") in {"lag_features", "rolling_features"}
            for step in (preprocessing_transformations or [])
        )
        if key in self._TABULAR_FEATURE_MODELS and has_target_derived_features:
            if prep_agent is None or not isinstance(raw_train_df, pd.DataFrame):
                raise RuntimeError(
                    "Causal target-derived feature evaluation requires raw training "
                    "history and the fitted preprocessing specification"
                )
            return self._causal_tabular_feature_evaluate(
                model_name=model_name,
                hp=hp,
                raw_train_df=raw_train_df,
                raw_eval_df=raw_evaluation_df,
                y_true=y_true,
                original_y_train=original_y_train,
                prep_agent=prep_agent,
                preprocessing_transformations=preprocessing_transformations or [],
                forecast_horizon=forecast_horizon,
                mase_denominator_source=mase_denominator_source,
                mase_seasonality=mase_seasonality,
                freq=freq,
            )

        if key in self._TABULAR_FEATURE_MODELS:
            return self._tabular_feature_evaluate(
                model_name=model_name,
                hp=hp,
                train_df=train_df,
                eval_df=eval_df,
                y_true=y_true,
                original_y_train=original_y_train,
                prep_agent=prep_agent,
                preprocessing_transformations=preprocessing_transformations,
                forecast_horizon=forecast_horizon,
                mase_y_train=mase_y_train,
                mase_spikes_replaced=mase_spikes_replaced,
                mase_denominator_source=mase_denominator_source,
                mase_seasonality=mase_seasonality,
                freq=freq,
            )

        if uses_rolling_refit:
            print(
                f"\n[EVAL_PROTOCOL] model={model_name}, protocol=rolling_origin_refit"
            )
            raw_history = (
                raw_train_df.copy(deep=True)
                if isinstance(raw_train_df, pd.DataFrame)
                else train_df.assign(target=np.asarray(original_y_train, dtype=float)).copy(deep=True)
            )
            if not np.array_equal(
                raw_history["target"].to_numpy(dtype=float),
                np.asarray(original_y_train, dtype=float),
                equal_nan=True,
            ):
                raise RuntimeError("raw_history must start on the original target scale")
            all_preds = []
            all_scaled_preds = []
            all_scaled_truth = []
            origin_audits: list[Dict[str, Any]] = []
            mase_per_origin: list[float] = []
            mae_per_origin: list[float] = []

            start = 0
            while start < len(eval_df):
                step_size = min(forecast_horizon, len(eval_df) - start)
                chunk_end = start + step_size

                # Ground truth and expanding history always remain on raw scale.
                raw_val_chunk = raw_evaluation_df.iloc[start:chunk_end].copy(deep=True)
                expected_raw = np.asarray(y_true[start:chunk_end], dtype=float)
                if not np.array_equal(
                    raw_val_chunk["target"].to_numpy(dtype=float),
                    expected_raw,
                    equal_nan=True,
                ):
                    raise RuntimeError("Raw validation chunk differs from metric ground truth")

                fold_transformations = preprocessing_transformations
                history_for_fit = raw_history
                chunk_for_model = raw_val_chunk.copy(deep=True)
                if prep_agent and preprocessing_transformations:
                    fold_plan = PreprocessingPlan(
                        transformations=prep_agent.specifications_for_refit(
                            preprocessing_transformations
                        )
                    )
                    history_for_fit, fold_transformations = (
                        prep_agent._apply_transformations(raw_history, fold_plan)
                    )
                    chunk_for_model = prep_agent.apply_transformations_from_log(
                        raw_val_chunk.copy(deep=True),
                        fold_transformations,
                        start_idx=len(raw_history),
                        history_df=raw_history,
                    )
                fold_outlier_log = next(
                    (
                        item for item in (fold_transformations or [])
                        if item.get("name") == "replace_outliers"
                    ),
                    {},
                )

                determinism = dict(getattr(self, "determinism_audit", {}) or {})
                candidate_seed = int(determinism.get("candidate_seed", self.random_state))
                from core.candidate_catalog import set_deterministic_seed

                set_deterministic_seed(candidate_seed)

                preds = self.library.train_and_predict(
                    model_name,
                    history_for_fit,
                    chunk_for_model,
                    hp,
                    forecast_horizon=forecast_horizon,
                    preprocessing_transformations=fold_transformations,
                    freq=freq

                )

                win_pred = np.asarray(preds, dtype=float)

                expected_prediction_size = (
                    forecast_horizon if has_structural_horizon else step_size
                )
                if len(win_pred) != expected_prediction_size:
                    raise RuntimeError(
                        f"Invalid forecast protocol for {model_name}: origin "
                        f"{len(origin_audits)} returned {len(win_pred)} predictions; "
                        f"expected {expected_prediction_size}"
                    )
                all_scaled_preds.extend(win_pred[:step_size].tolist())
                all_scaled_truth.extend(
                    chunk_for_model["target"].to_numpy(dtype=float).tolist()
                )

                if prep_agent and preprocessing_transformations:
                    win_pred_inv = prep_agent.inverse_transform(
                        win_pred,
                        fold_transformations,
                        (
                            raw_history["target"].to_numpy(dtype=float)
                        ),
                    )
                else:
                    win_pred_inv = win_pred

                win_pred_inv = np.asarray(win_pred_inv, dtype=float).reshape(-1)
                if len(win_pred_inv) != expected_prediction_size:
                    raise RuntimeError(
                        f"Invalid forecast protocol for {model_name}: inverse "
                        f"prediction cardinality={len(win_pred_inv)}, expected "
                        f"{expected_prediction_size}"
                    )
                win_pred_inv = win_pred_inv[:step_size]

                origin_index = len(origin_audits)
                fold_truth = np.asarray(expected_raw, dtype=float)
                mase_per_origin.append(
                    float(
                        mase(
                            fold_truth,
                            win_pred_inv,
                            mase_y_train,
                            seasonality=mase_seasonality,
                        )
                    )
                )
                mae_per_origin.append(float(mae(fold_truth, win_pred_inv)))
                date_column = "date" if "date" in raw_history.columns else None
                forecast_date_column = (
                    "date" if "date" in raw_val_chunk.columns else None
                )
                origin_audits.append(
                    {
                        "origin_index": origin_index,
                        "train_start": (
                            pd.Timestamp(raw_history[date_column].iloc[0]).isoformat()
                            if date_column and len(raw_history)
                            else 0
                        ),
                        "train_end": (
                            pd.Timestamp(raw_history[date_column].iloc[-1]).isoformat()
                            if date_column and len(raw_history)
                            else len(raw_history) - 1
                        ),
                        "forecast_start": (
                            pd.Timestamp(raw_val_chunk[forecast_date_column].iloc[0]).isoformat()
                            if forecast_date_column and len(raw_val_chunk)
                            else start
                        ),
                        "forecast_end": (
                            pd.Timestamp(raw_val_chunk[forecast_date_column].iloc[-1]).isoformat()
                            if forecast_date_column and len(raw_val_chunk)
                            else chunk_end - 1
                        ),
                        "configured_forecast_horizon": int(forecast_horizon),
                        "effective_model_horizon": int(expected_prediction_size),
                        "evaluation_window_size": int(step_size),
                        "candidate_signature": str(
                            determinism.get("candidate_signature", "")
                        ),
                        "candidate_seed": candidate_seed,
                        "model_name": model_name,
                        "preprocessing_signature": str(
                            determinism.get("candidate_signature", "")
                        ),
                        "replace_outliers": fold_outlier_log,
                        "validation_target_modified": False,
                    }
                )

                all_preds.extend(win_pred_inv.tolist())
                print(
                    f"[REFIT_CHUNK] model={model_name}, chunk_start={start}, chunk_end={chunk_end}, history_len={len(raw_history)}, forecast_horizon={forecast_horizon}, evaluated_steps={len(win_pred_inv)}"
                )

                raw_history = pd.concat(
                    [raw_history, raw_val_chunk], ignore_index=False
                )
                if not np.array_equal(
                    raw_history["target"].iloc[-step_size:].to_numpy(dtype=float),
                    expected_raw,
                    equal_nan=True,
                ):
                    raise RuntimeError("raw_history accepted a transformed validation target")
                start += step_size

            train_metrics = getattr(self.library, "last_train_info", {})
            scaled_preds_arr = np.asarray(all_scaled_preds, dtype=float)
            scaled_truth_arr = np.asarray(all_scaled_truth, dtype=float)
            scaled_mse_val = float(mse(scaled_truth_arr, scaled_preds_arr))
        else:
            # Lag-based: single training + windowed evaluation
            return self._windowed_single_train_evaluate(
                model_name=model_name,
                hp=hp,
                train_df=train_df,
                eval_df=eval_df,
                y_true=y_true,
                original_y_train=original_y_train,
                prep_agent=prep_agent,
                preprocessing_transformations=preprocessing_transformations,
                forecast_horizon=forecast_horizon,
                mase_y_train=mase_y_train,
                mase_spikes_replaced=mase_spikes_replaced,
                mase_denominator_source=mase_denominator_source,
                mase_seasonality=mase_seasonality,
                freq=freq
            )

        preds_arr = np.array(all_preds)

        if len(y_true) != len(preds_arr):
            raise ValueError(
                f"Length mismatch: len(y_true)={len(y_true)} != len(preds_arr)={len(preds_arr)}"
            )

        from utils.metrics import compute_forecast_metrics

        print(
            f"[METRICS_INPUT] model={model_name}, validation_protocol=rolling_origin_refit, len_y_true={len(y_true)}, len_y_pred={len(preds_arr)}, len_y_train={len(original_y_train)}, evaluated_points={len(preds_arr)}"
        )

        split_metrics = compute_forecast_metrics(
            y_true=y_true,
            y_pred=preds_arr,
            y_train=mase_y_train,
            mase_seasonality=mase_seasonality,
            model_name=model_name,
            validation_protocol="rolling_origin_refit",
        )
        split_metrics["scaled_mse"] = scaled_mse_val
        self._add_mase_trace(
            split_metrics, mase_y_train, mase_seasonality,
            mase_denominator_source, mase_spikes_replaced,
        )
        split_metrics.update({
            "validation_target_modified": False,
            "validation_points_excluded_from_metrics": 0,
        })

        print(
            f"[METRICS] model={model_name}, mase={split_metrics['mase']:.4f}, rmse={split_metrics['rmse']:.4f}, mae={split_metrics['mae']:.4f}, mse={split_metrics['mse']:.4f}, r2={split_metrics['r2']:.4f}, mape={split_metrics['mape']:.4f}, evaluated_points={len(preds_arr)}"
        )
        print(
            f"[METRICS_SCALE] model={model_name}, metrics_scale=original, inverse_transform_applied={True if prep_agent and preprocessing_transformations else False}"
        )
        print(f"[MASE] chosen_seasonality={mase_seasonality}, computed_mase={split_metrics['mase']:.4f}")

        split_metrics["validation_protocol"] = "rolling_origin_refit"
        split_metrics["forecast_horizon"] = forecast_horizon
        split_metrics["configured_forecast_horizon"] = forecast_horizon
        split_metrics["effective_model_horizon"] = forecast_horizon
        split_metrics["evaluation_window_size"] = forecast_horizon
        split_metrics["number_of_validation_origins"] = len(origin_audits)
        split_metrics["validation_effective_horizons"] = [
            item["effective_model_horizon"] for item in origin_audits
        ]
        split_metrics["validation_origin_audits"] = origin_audits
        split_metrics["validation_mase_per_origin"] = mase_per_origin
        split_metrics["validation_mae_per_origin"] = mae_per_origin
        split_metrics["horizon_consistency"] = all(
            (
                item["effective_model_horizon"] == forecast_horizon
                if has_structural_horizon
                else item["effective_model_horizon"]
                == item["evaluation_window_size"]
            )
            and 0 < item["evaluation_window_size"] <= forecast_horizon
            for item in origin_audits
        )
        if not split_metrics["horizon_consistency"]:
            raise RuntimeError(f"invalid_forecast_protocol: {model_name}")
        split_metrics["evaluated_points"] = len(preds_arr)
        split_metrics["mase_seasonality"] = mase_seasonality
        split_metrics["metrics_computed_on"] = "full_validation"
        split_metrics["y_true_len"] = len(y_true)
        split_metrics["y_pred_len"] = len(preds_arr)

        return preds_arr, split_metrics, train_metrics

    def train_and_evaluate(
        self,
        model_name: str,
        hyperparameters: Dict[str, object],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        raw_train_df: pd.DataFrame | None = None,
        raw_val_df: pd.DataFrame | None = None,
        preprocessing_log: Dict[str, Any] | None = None,
        forecast_horizon: int = 10,
        mase_seasonality: int = 1,
        freq: str = "D"
    ) -> TrainingResult:
        print(f"  > [EXECUTION] Starting training for {model_name}...")
        from agents.preprocessing_agent import PreprocessingAgent, PreprocessingPlan

        if raw_val_df is not None and not raw_val_df.empty:
            y_true = np.asarray(raw_val_df["target"].values, dtype=float)
        else:
            y_true = np.asarray(val_df["target"].values, dtype=float)

        y_eval_arr = np.asarray(val_df["target"].values, dtype=float)
        if len(y_true) != len(y_eval_arr):
            raise RuntimeError(
                "Preprocessing changed validation cardinality: "
                f"raw={len(y_true)}, processed={len(y_eval_arr)}. "
                "Every validation observation must be evaluated."
            )

        y_train = np.asarray(train_df["target"].values, dtype=float)

        if raw_train_df is not None and not raw_train_df.empty:
            original_y_train = np.asarray(raw_train_df["target"].values, dtype=float)
        else:
            original_y_train = y_train

        prep_agent = None
        transformations = None
        if preprocessing_log and preprocessing_log.get("transformations"):
            transformations = preprocessing_log["transformations"]
            prep_agent = PreprocessingAgent(llm=None)

        print(f"[TRAINER_CONFIG] source=technical_agent, model={model_name}, hyperparameters={hyperparameters}, internal_tuning=False")

        hp = dict(hyperparameters)

        preds, val_metrics, train_metrics = self._rolling_window_evaluate(
            model_name=model_name,
            hp=hp,
            train_df=train_df,
            eval_df=val_df,
            y_true=y_true,
            original_y_train=original_y_train,
            prep_agent=prep_agent,
            preprocessing_transformations=transformations,
            raw_train_df=raw_train_df,
            forecast_horizon=forecast_horizon,
            raw_eval_df=raw_val_df,
            mase_seasonality=mase_seasonality,
            freq=freq
        )

        val_metrics["config_source"] = "technical_agent"
        val_metrics["internal_tuning"] = False
        val_metrics["evaluated_candidate_count"] = 1

        preprocessing_info = getattr(
            self.library, "last_preprocessing_info", {}
        )
        scaler_used = None
        if isinstance(preprocessing_info, dict):
            raw_scaler = preprocessing_info.get("scaler_used")
            scaler_used = str(raw_scaler) if raw_scaler is not None else None

        best_result = TrainingResult(
            model_name=model_name,
            hyperparameters=hp,
            metrics=val_metrics,
            scaler_used=scaler_used,
            predictions=preds,
            train_metrics=train_metrics,
        )

        return best_result
