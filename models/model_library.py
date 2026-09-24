"""Model library for time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from catboost import CatBoostRegressor
from chronos import BaseChronosPipeline
from lightgbm import LGBMRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    Autoformer,
    FEDformer,
    Informer,
    NHITS,
    PatchTST,
    TiDE,
    TimesNet,
)
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from utils.metrics import r2, mape
from utils.numeric_input_policy import validated_model_feature_columns
from utils.reproducibility import estimator_fingerprint, parameter_fingerprint
from models.defaults import HYPERPARAMETER_CONSTRAINTS


class AuditedMLPRegressor(MLPRegressor):
    """MLPRegressor with a read-only fingerprint of pre-optimizer weights."""

    def _initialize(self, y, layer_units, dtype):
        super()._initialize(y, layer_units, dtype)
        self.initial_weights_hash_ = parameter_fingerprint(
            [*self.coefs_, *self.intercepts_]
        )


_NEURALFORECAST_MODELS = {
    "Autoformer": Autoformer,
    "Informer": Informer,
    "FEDformer": FEDformer,
    "N-HiTS": NHITS,
    "PatchTST": PatchTST,
    "TimesNet": TimesNet,
    "TiDE": TiDE,
}


HORIZON_STRUCTURAL_MODELS = frozenset(
    {
        "N-BEATS",
        "LSTM",
        "DLinear",
        "Reformer",
        "Chronos",
        *_NEURALFORECAST_MODELS.keys(),
    }
)

TABULAR_FEATURE_MODELS = frozenset(
    {
        "MLP",
        "XGBoost",
        "RandomForest",
        "LightGBM",
        "CatBoost",
        "ElasticNet",
        "KNN",
        "SVR",
    }
)


@dataclass(frozen=True)
class FittedTabularModel:
    """One origin-local fitted estimator with a fixed feature contract."""

    model_name: str
    estimator: Any
    feature_columns: tuple[str, ...]


def _configured_horizon(forecast_horizon: int, *, model_name: str) -> int:
    horizon = int(forecast_horizon)
    if horizon <= 0:
        raise ValueError(
            f"Invalid forecast protocol for {model_name}: "
            "configured_forecast_horizon must be > 0"
        )
    return horizon


def _to_numpy(series: pd.Series) -> np.ndarray:
    return np.asarray(series.values, dtype=float)




def _make_multioutput_lag_features(
    y: np.ndarray, lag: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create tabular dataset for multi-output forecasting."""
    X, Y = [], []
    if len(y) < lag + horizon:
        return np.array([]), np.array([])
    for i in range(len(y) - lag - horizon + 1):
        X.append(y[i : i + lag])
        Y.append(y[i + lag : i + lag + horizon])
    return np.array(X), np.array(Y)


def _preprocessed_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns already created by PreprocessingAgent.

    ModelLibrary must not recreate lag/rolling/calendar features and must not fit
    scalers. Feature engineering and scaling are owned by PreprocessingAgent.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Feature extraction requires a pandas DataFrame.")
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain a 'target' column.")

    return validated_model_feature_columns(
        df, context="ModelLibrary feature extraction"
    )


def _make_preprocessed_tabular_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Use all numeric preprocessed features produced by PreprocessingAgent.

    This function intentionally does not fit any scaler, does not inspect a
    model-aware scaling policy, and does not recreate lag windows from target. If the
    required feature columns are missing, the preprocessing pipeline is invalid
    and the model must fail loudly.
    """
    feature_cols = _preprocessed_feature_columns(train_df)
    if not feature_cols:
        raise ValueError(
            f"{model_name} requires preprocessed numeric feature columns. "
            "Add target-derived lag/rolling features or timestamp-derived calendar "
            "features "
            "in the TechnicalAgent preprocessing pipeline before training."
        )

    missing_val = [c for c in feature_cols if c not in val_df.columns]
    if missing_val:
        raise ValueError(
            f"Validation dataframe is missing preprocessed feature columns for {model_name}: {missing_val}"
        )

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["target"].to_numpy(dtype=float)
    x_val = val_df[feature_cols].to_numpy(dtype=float)

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"{model_name} feature/target row mismatch: X={x_train.shape[0]}, y={y_train.shape[0]}"
        )
    if x_train.size == 0 or x_val.size == 0:
        raise ValueError(f"{model_name} received empty preprocessed feature matrices.")
    if not np.isfinite(x_train).all():
        raise ValueError(f"{model_name} training features contain NaN or infinite values after preprocessing.")
    if not np.isfinite(x_val).all():
        raise ValueError(f"{model_name} validation features contain NaN or infinite values after preprocessing.")
    if not np.isfinite(y_train).all():
        raise ValueError(f"{model_name} training target contains NaN or infinite values after preprocessing.")

    return x_train, y_train, x_val, feature_cols


def _preprocessing_info(
    model_name: str,
    *,
    feature_cols: list[str] | None = None,
) -> Dict[str, object]:
    """Audit metadata: scaling is external, never fitted inside ModelLibrary."""
    return {
        "model": model_name,
        "feature_columns_used": feature_cols or [],
        "n_feature_columns_used": len(feature_cols or []),
        "external_covariates_used": False,
        "feature_source_policy": "target_or_timestamp_derived_only",
        "scaler_used": "see_preprocessing_log",
        "scaler_fitted_on": "train_window_only_if_declared_in_preprocessing_log",
        "scaling_managed_by": "PreprocessingAgent",
    }



_CANONICAL_MODEL_NAMES: Dict[str, str] = {
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "ets": "ETS",
    "mlp": "MLP",
    "lstm": "LSTM",
    "xgboost": "XGBoost",
    "randomforest": "RandomForest",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "nbeats": "N-BEATS",
    "n-beats": "N-BEATS",
    "chronos": "Chronos",
    "dlinear": "DLinear",
    "reformer": "Reformer",
    "autoformer": "Autoformer",
    "informer": "Informer",
    "fedformer": "FEDformer",
    "nhits": "N-HiTS",
    "n-hits": "N-HiTS",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
    "tide": "TiDE",
    "elasticnet": "ElasticNet",
    "knn": "KNN",
    "svr": "SVR",
}


_RUNTIME_HYPERPARAMETER_KEYS: Dict[str, set[str]] = {
    "ARIMA": {"order"},
    "SARIMA": {"order", "seasonal_order"},
    "Prophet": {"seasonality_mode", "changepoint_prior_scale"},
    "ETS": {"trend", "seasonal", "seasonal_periods"},
    "MLP": {"hidden_layer_sizes"},
    "LSTM": {"lag", "hidden_size", "num_layers", "dropout", "epochs", "learning_rate"},
    "XGBoost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"},
    "RandomForest": {"n_estimators", "max_depth"},
    "LightGBM": {"n_estimators", "learning_rate", "max_depth"},
    "CatBoost": {"iterations", "learning_rate", "depth"},
    "N-BEATS": {"lookback", "hidden_units", "stacks", "epochs"},
    "Chronos": {"model_size"},
    "DLinear": {"lag", "epochs"},
    "Reformer": {"lag", "epochs", "nhead", "num_layers", "dim_feedforward"},
    "Autoformer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "Informer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "FEDformer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "N-HiTS": {"input_size", "max_steps"},
    "PatchTST": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers"},
    "TimesNet": {"input_size", "hidden_size", "max_steps"},
    "TiDE": {"input_size", "hidden_size", "max_steps", "num_decoder_layers"},
    "ElasticNet": {"alpha", "l1_ratio", "fit_intercept", "max_iter", "tol", "selection", "positive"},
    "KNN": {"n_neighbors", "weights", "algorithm", "leaf_size", "p", "metric"},
    "SVR": {"C", "epsilon", "kernel", "gamma", "degree", "coef0", "shrinking", "tol", "cache_size", "max_iter"},
}

_REQUIRED_HYPERPARAMETER_KEYS: Dict[str, set[str]] = {
    "ARIMA": {"order"},
    "SARIMA": {"order", "seasonal_order"},
    "Prophet": {"seasonality_mode", "changepoint_prior_scale"},
    "ETS": {"trend", "seasonal"},
    "MLP": {"hidden_layer_sizes"},
    "LSTM": {"lag", "hidden_size", "num_layers", "dropout", "epochs", "learning_rate"},
    "XGBoost": {"n_estimators", "max_depth", "learning_rate"},
    "RandomForest": {"n_estimators", "max_depth"},
    "LightGBM": {"n_estimators", "learning_rate", "max_depth"},
    "CatBoost": {"iterations", "learning_rate", "depth"},
    "N-BEATS": {"lookback", "hidden_units", "stacks", "epochs"},
    "Chronos": {"model_size"},
    "DLinear": {"lag", "epochs"},
    "Reformer": {"lag", "epochs", "nhead", "num_layers", "dim_feedforward"},
    "Autoformer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "Informer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "FEDformer": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers", "decoder_layers"},
    "N-HiTS": {"input_size", "max_steps"},
    "PatchTST": {"input_size", "hidden_size", "max_steps", "n_head", "encoder_layers"},
    "TimesNet": {"input_size", "hidden_size", "max_steps"},
    "TiDE": {"input_size", "hidden_size", "max_steps", "num_decoder_layers"},
    "ElasticNet": {"alpha", "l1_ratio"},
    "KNN": {"n_neighbors"},
    "SVR": {"C", "epsilon", "kernel"},
}


def _canonical_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in _CANONICAL_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    return _CANONICAL_MODEL_NAMES[key]


def _hyperparameter_contract(model_name: str) -> tuple[set[str], set[str], set[str]]:
    canonical = _canonical_model_name(model_name)
    runtime_keys = set(_RUNTIME_HYPERPARAMETER_KEYS.get(canonical, set()))
    schema_obj = HYPERPARAMETER_CONSTRAINTS.get(canonical)
    if not isinstance(schema_obj, dict):
        raise ValueError(
            f"Missing hyperparameter search space for {canonical}. "
            "No fallback search-space contract is allowed."
        )
    schema_keys = set(schema_obj.keys())
    accepted_keys = runtime_keys & schema_keys
    return accepted_keys, runtime_keys, schema_keys


def _validate_runtime_hyperparameters(model_name: str, hp: Dict[str, object] | None) -> Dict[str, object]:
    canonical = _canonical_model_name(model_name)
    if hp is None:
        raise ValueError(
            f"Hyperparameters missing for {canonical}. "
            "TechnicalAgent must provide explicit hyperparameters; no fallback defaults are allowed."
        )
    if not isinstance(hp, dict):
        raise ValueError(f"Hyperparameters for {canonical} must be a dictionary, got {type(hp).__name__}.")

    from core.candidate_catalog import DEFAULT_CANDIDATE_CATALOG

    DEFAULT_CANDIDATE_CATALOG.validate(canonical, hp)

    accepted_keys, runtime_keys, schema_keys = _hyperparameter_contract(canonical)
    proposed_keys = set(hp.keys())
    rejected_unknown = sorted(proposed_keys - schema_keys)
    rejected_not_implemented = sorted((proposed_keys & schema_keys) - runtime_keys)
    rejected = sorted(set(rejected_unknown) | set(rejected_not_implemented))

    missing_required = sorted(set(_REQUIRED_HYPERPARAMETER_KEYS.get(canonical, set())) - proposed_keys)
    if canonical == "ETS" and hp.get("seasonal") is not None and "seasonal_periods" not in proposed_keys:
        missing_required.append("seasonal_periods")

    if rejected or missing_required:
        report = {
            "model": canonical,
            "accepted_hyperparameters": sorted(proposed_keys & accepted_keys),
            "rejected_hyperparameters": rejected,
            "rejected_unknown_to_search_space": rejected_unknown,
            "rejected_declared_but_not_implemented": rejected_not_implemented,
            "missing_required_hyperparameters": missing_required,
            "allowed_hyperparameters": sorted(accepted_keys),
        }
        raise ValueError("Invalid hyperparameters: " + repr(report))


    return dict(hp)


def _require_hp(hp: Dict[str, object], name: str) -> object:
    if name not in hp:
        raise ValueError(f"Missing required hyperparameter: {name}")
    return hp[name]


def _optional_hp(hp: Dict[str, object], name: str, default: object) -> object:
    # Optional means optional in the explicit runtime contract. This is not used
    # to hide invalid proposed parameters; invalid keys are rejected before this.
    return hp[name] if name in hp else default

@dataclass
class ModelLibrary:
    """Forecast model implementations without internal preprocessing fallbacks."""

    random_state: int = 42

    def __post_init__(self) -> None:
        self.last_preprocessing_info: Dict[str, object] = {
            "model": None,
            "scaler_used": None,
            "scaler_fitted_on": None,
        }
        self.last_train_info: Dict[str, Any] = {}



    def _save_ml_train_info(self, model, x_train, y_supervised) -> None:
        train_preds = model.predict(x_train)
        train_mse = float(np.nanmean((train_preds - y_supervised) ** 2))
        train_mae = float(np.nanmean(np.abs(train_preds - y_supervised)))
        last_loss = train_mse
        if hasattr(model, "estimators_"):
            losses = [
                est.loss_curve_[-1]
                for est in model.estimators_
                if hasattr(est, "loss_curve_") and est.loss_curve_
            ]
            if losses:
                last_loss = sum(losses) / len(losses)
        self.last_train_info = {
            "loss": last_loss,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "r2": float(r2(y_supervised, train_preds)),
            "mape": float(mape(y_supervised, train_preds)),
            "y_true": y_supervised,
            "y_pred": train_preds,
        }
        initial_hash = getattr(model, "initial_weights_hash_", None)
        final_hash = estimator_fingerprint(model)
        if initial_hash is not None:
            self.last_train_info["initial_weights_hash"] = initial_hash
        if final_hash is not None:
            self.last_train_info["final_weights_hash"] = final_hash
        loss_curve = [float(value) for value in getattr(model, "loss_curve_", [])]
        validation_scores = [
            float(value) for value in (getattr(model, "validation_scores_", None) or [])
        ]
        self.last_train_info["train_loss_curve"] = loss_curve
        self.last_train_info["validation_score_curve"] = validation_scores
        self.last_train_info["early_stop_epoch"] = int(getattr(model, "n_iter_", 0))
        self.last_train_info["best_epoch"] = (
            int(np.nanargmax(validation_scores)) + 1
            if validation_scores
            else int(getattr(model, "n_iter_", 0))
        )

    def get_base_estimator(self, key: str, hp: Dict[str, Any]) -> Any:
        """Returns the un-fitted base estimator for lag-based ML models."""
        if key == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=int(_require_hp(hp, "n_estimators")),
                max_depth=int(_require_hp(hp, "max_depth")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                subsample=float(_optional_hp(hp, "subsample", 1.0)),
                colsample_bytree=float(_optional_hp(hp, "colsample_bytree", 1.0)),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="rmse",
            )
        elif key == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=int(_require_hp(hp, "n_estimators")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                max_depth=int(_require_hp(hp, "max_depth")),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif key == "catboost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=int(_require_hp(hp, "iterations")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                depth=int(_require_hp(hp, "depth")),
                random_seed=self.random_state,
                verbose=0,
                allow_writing_files=False,
            )
        elif key == "randomforest":
            from sklearn.ensemble import RandomForestRegressor

            max_depth = _require_hp(hp, "max_depth")
            return RandomForestRegressor(
                n_estimators=int(_require_hp(hp, "n_estimators")),
                max_depth=None if max_depth is None else int(max_depth),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif key == "mlp":
            from sklearn.neural_network import MLPRegressor

            return AuditedMLPRegressor(
                hidden_layer_sizes=tuple(_require_hp(hp, "hidden_layer_sizes")),
                max_iter=500,
                random_state=self.random_state,
            )
        elif key == "elasticnet":
            from sklearn.linear_model import ElasticNet

            return ElasticNet(
                alpha=float(_require_hp(hp, "alpha")),
                l1_ratio=float(_require_hp(hp, "l1_ratio")),
                fit_intercept=bool(_optional_hp(hp, "fit_intercept", True)),
                max_iter=int(_optional_hp(hp, "max_iter", 1000)),
                tol=float(_optional_hp(hp, "tol", 1e-4)),
                selection=str(_optional_hp(hp, "selection", "cyclic")),
                positive=bool(_optional_hp(hp, "positive", False)),
                random_state=self.random_state,
            )
        elif key == "knn":
            from sklearn.neighbors import KNeighborsRegressor

            return KNeighborsRegressor(
                n_neighbors=int(_require_hp(hp, "n_neighbors")),
                weights=str(_optional_hp(hp, "weights", "uniform")),
                algorithm=str(_optional_hp(hp, "algorithm", "auto")),
                leaf_size=int(_optional_hp(hp, "leaf_size", 30)),
                p=int(_optional_hp(hp, "p", 2)),
                metric=str(_optional_hp(hp, "metric", "minkowski")),
            )
        elif key == "svr":
            from sklearn.svm import SVR

            return SVR(
                C=float(_require_hp(hp, "C")),
                epsilon=float(_require_hp(hp, "epsilon")),
                kernel=str(_require_hp(hp, "kernel")),
                gamma=_optional_hp(hp, "gamma", "scale"),
                degree=int(_optional_hp(hp, "degree", 3)),
                coef0=float(_optional_hp(hp, "coef0", 0.0)),
                shrinking=bool(_optional_hp(hp, "shrinking", True)),
                tol=float(_optional_hp(hp, "tol", 1e-3)),
                cache_size=float(_optional_hp(hp, "cache_size", 200.0)),
                max_iter=int(_optional_hp(hp, "max_iter", -1)),
            )
        else:
            raise ValueError(f"No base estimator for key: {key}")

    def fit_tabular_model(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        hyperparameters: Dict[str, object] | None,
        *,
        preprocessing_transformations: list | None = None,
    ) -> FittedTabularModel:
        """Fit one tabular estimator on one origin's preprocessed history."""
        canonical_model = _canonical_model_name(model_name)
        if canonical_model not in TABULAR_FEATURE_MODELS:
            raise ValueError(f"{canonical_model} is not a tabular feature model.")

        from utils.configuration_identity import normalize_candidate_hyperparameters

        normalized_hp, normalization_audit = normalize_candidate_hyperparameters(
            canonical_model, hyperparameters, preprocessing_transformations
        )
        hp = _validate_runtime_hyperparameters(canonical_model, normalized_hp)
        if normalization_audit:
            self.last_configuration_normalization = normalization_audit

        feature_columns = _preprocessed_feature_columns(train_df)
        if not feature_columns:
            raise ValueError(
                f"{canonical_model} requires preprocessed numeric feature columns."
            )
        x_train = train_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df["target"].to_numpy(dtype=float)
        if x_train.size == 0 or y_train.size == 0:
            raise ValueError(f"{canonical_model} received empty training data.")
        if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
            raise ValueError(
                f"{canonical_model} training data contains NaN or infinite values."
            )

        key = canonical_model.lower().replace("-", "")
        estimator = self.get_base_estimator(key, hp)
        if key == "knn" and int(hp["n_neighbors"]) > len(x_train):
            raise ValueError(
                f"KNN n_neighbors={hp['n_neighbors']} exceeds the "
                f"{len(x_train)} available training rows."
            )
        estimator.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            canonical_model, feature_cols=feature_columns
        )
        self._save_ml_train_info(estimator, x_train, y_train)
        self.last_train_info["model_fit_count"] = 1
        self.last_train_info["feature_columns_used"] = list(feature_columns)
        self.last_train_info["external_covariates_used"] = False
        return FittedTabularModel(
            model_name=canonical_model,
            estimator=estimator,
            feature_columns=tuple(feature_columns),
        )

    def predict_tabular_model(
        self,
        fitted: FittedTabularModel,
        eval_df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict with an already-fitted origin-local tabular estimator."""
        if not isinstance(fitted, FittedTabularModel):
            raise TypeError("predict_tabular_model requires FittedTabularModel.")
        missing = [
            column
            for column in fitted.feature_columns
            if column not in eval_df.columns
        ]
        if missing:
            raise ValueError(
                f"{fitted.model_name} evaluation data is missing features: {missing}"
            )
        x_eval = eval_df[list(fitted.feature_columns)].to_numpy(dtype=float)
        if x_eval.size == 0 or not np.isfinite(x_eval).all():
            raise ValueError(
                f"{fitted.model_name} evaluation features are empty or non-finite."
            )
        return np.asarray(fitted.estimator.predict(x_eval), dtype=float).reshape(-1)

    def fit_predict_arima(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        order: Tuple[int, int, int] = (2, 1, 2),
    ) -> np.ndarray:
        self.last_preprocessing_info = {
            "model": "ARIMA",
            "scaler_used": None,
            "scaler_fitted_on": None,
        }
        y_train = _to_numpy(train_df["target"])
        model = ARIMA(y_train, exog=None, order=order)
        fitted = model.fit()
        pred = fitted.forecast(steps=len(val_df), exog=None)

        # Calculate train metrics
        fitted_vals = fitted.fittedvalues
        train_mse = float(np.nanmean((fitted_vals - y_train) ** 2))
        train_mae = float(np.nanmean(np.abs(fitted_vals - y_train)))
        self.last_train_info = {
            "loss": train_mse,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "r2": float(r2(y_train, fitted_vals)),
            "mape": float(mape(y_train, fitted_vals)),
            "y_true": y_train,
            "y_pred": fitted_vals,
        }
        return np.asarray(pred, dtype=float)

    def fit_predict_prophet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        freq: str = "D",
    ) -> np.ndarray:
        self.last_preprocessing_info = {
            "model": "Prophet",
            "scaler_used": None,
            "scaler_fitted_on": None,
        }
        train = train_df.rename(columns={"date": "ds", "target": "y"})[["ds", "y"]]
        
        freq_upper = freq.upper()
        daily_seasonality = False
        weekly_seasonality = False
        yearly_seasonality = False
        
        if freq_upper in ['H', 'MIN', 'T', 'S']:
            daily_seasonality = True
            weekly_seasonality = True
            yearly_seasonality = True
        elif freq_upper == 'D':
            weekly_seasonality = True
            yearly_seasonality = True
        elif freq_upper == 'W':
            yearly_seasonality = True
        elif freq_upper in ['M', 'MS']:
            yearly_seasonality = True

        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
        )
        model.fit(train)
        future = pd.DataFrame({"ds": val_df["date"]})
        forecast = model.predict(future)

        # Calculate train metrics
        y_train = train["y"].values
        train_forecast = model.predict(train[["ds"]])
        fitted_vals = train_forecast["yhat"].values
        train_mse = float(np.nanmean((fitted_vals - y_train) ** 2))
        train_mae = float(np.nanmean(np.abs(fitted_vals - y_train)))
        self.last_train_info = {
            "loss": train_mse,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "r2": float(r2(y_train, fitted_vals)),
            "mape": float(mape(y_train, fitted_vals)),
            "y_true": y_train,
            "y_pred": fitted_vals,
        }
        return np.asarray(forecast["yhat"].values, dtype=float)

    def fit_predict_mlp(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    ) -> np.ndarray:
        """MLP over all numeric features already produced by PreprocessingAgent."""
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="MLP",
        )
        base_model = AuditedMLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=self.random_state,
            max_iter=500,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info("MLP", feature_cols=feature_cols)
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_xgboost(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="XGBoost",
        )
        base_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="rmse",
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "XGBoost",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_sarima(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        order: Tuple[int, int, int] = (3, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> np.ndarray:
        self.last_preprocessing_info = {
            "model": "SARIMAX",
            "scaler_used": None,
            "scaler_fitted_on": None,
            "external_covariates_used": False,
            "feature_columns_used": [],
        }
        y_train = _to_numpy(train_df["target"])

        model = SARIMAX(
            y_train,
            exog=None,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
        )
        fitted = model.fit(disp=False)
        pred = fitted.forecast(steps=len(val_df), exog=None)

        # Calculate train metrics
        fitted_vals = fitted.fittedvalues
        train_mse = float(np.nanmean((fitted_vals - y_train) ** 2))
        train_mae = float(np.nanmean(np.abs(fitted_vals - y_train)))
        self.last_train_info = {
            "loss": train_mse,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "r2": float(r2(y_train, fitted_vals)),
            "mape": float(mape(y_train, fitted_vals)),
            "y_true": y_train,
            "y_pred": fitted_vals,
        }
        return np.asarray(pred, dtype=float)

    def fit_predict_ets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int | None = None,
    ) -> np.ndarray:
        self.last_preprocessing_info = {
            "model": "ETS",
            "scaler_used": None,
            "scaler_fitted_on": None,
        }
        y_train = _to_numpy(train_df["target"])

        if seasonal is not None and seasonal_periods is None:
            raise ValueError(
                "ETS requires seasonal_periods when seasonal is not None. "
                "seasonal_periods must be computed by the AnalyticalAgent and "
                "injected by the TechnicalAgent — no hardcoded default is allowed."
            )


        if (y_train <= 0).any():
            if trend == "mul" or seasonal == "mul":
                raise ValueError(
                    "ETS multiplicative components require strictly positive "
                    "training values."
                )

        model = ExponentialSmoothing(
            y_train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        fitted = model.fit(optimized=True)
        pred = fitted.forecast(steps=len(val_df))

        # Calculate train metrics
        fitted_vals = fitted.fittedvalues
        train_mse = float(np.nanmean((fitted_vals - y_train) ** 2))
        train_mae = float(np.nanmean(np.abs(fitted_vals - y_train)))
        self.last_train_info = {
            "loss": train_mse,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "r2": float(r2(y_train, fitted_vals)),
            "mape": float(mape(y_train, fitted_vals)),
            "y_true": y_train,
            "y_pred": fitted_vals,
        }
        return np.asarray(pred, dtype=float)

    def fit_predict_randomforest(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_estimators: int = 100,
        max_depth: int | None = 10,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="RandomForest",
        )
        base_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "RandomForest",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_lightgbm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="LightGBM",
        )
        base_model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            verbose=-1,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "LightGBM",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_catboost(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="CatBoost",
        )
        base_model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=self.random_state,
            verbose=0,
            allow_writing_files=False,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "CatBoost",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_elasticnet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        selection: str = "cyclic",
        positive: bool = False,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="ElasticNet",
        )
        base_model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            positive=positive,
            random_state=self.random_state,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "ElasticNet",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_knn(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="KNN",
        )
        if n_neighbors > len(x_train):
            raise ValueError(
                f"KNN n_neighbors={n_neighbors} is larger than the number of preprocessed training rows ({len(x_train)}). "
                "No automatic fallback/clipping is allowed."
            )
        base_model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "KNN",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_svr(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200.0,
        max_iter: int = -1,
    ) -> np.ndarray:
        x_train, y_train, x_val, feature_cols = _make_preprocessed_tabular_xy(
            train_df,
            val_df,
            model_name="SVR",
        )
        base_model = SVR(
            C=C,
            epsilon=epsilon,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
            max_iter=max_iter,
        )
        base_model.fit(x_train, y_train)
        self.last_preprocessing_info = _preprocessing_info(
            "SVR",
            feature_cols=feature_cols,
        )
        self._save_ml_train_info(base_model, x_train, y_train)
        preds = base_model.predict(x_val)
        return np.asarray(preds, dtype=float)

    def fit_predict_nbeats(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        lookback: int = 20,
        hidden_units: int = 256,
        stacks: int = 2,
        epochs: int = 100,
    ) -> np.ndarray:
        import torch
        import torch.nn as nn

        self.last_preprocessing_info = _preprocessing_info("N-BEATS")
        y_train = _to_numpy(train_df["target"])
        horizon = _configured_horizon(forecast_horizon, model_name="N-BEATS")

        if len(y_train) < lookback + horizon:
            raise ValueError(
                f"N-BEATS lookback={lookback} requires at least "
                f"{lookback + horizon} training points; received {len(y_train)}."
            )
        x_train_raw, y_train_raw = _make_multioutput_lag_features(
            y_train, lookback, horizon
        )

        # Target/series scaling, when required, has already been applied by PreprocessingAgent.
        x_train = x_train_raw
        y_train_scaled = y_train_raw

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Minimal N-BEATS block
        class _Block(nn.Module):
            def __init__(self, in_dim: int, h: int, out_dim: int) -> None:
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(in_dim, h),
                    nn.ReLU(),
                    nn.Linear(h, h),
                    nn.ReLU(),
                    nn.Linear(h, h),
                    nn.ReLU(),
                )
                self.backcast_fc = nn.Linear(h, in_dim)
                self.forecast_fc = nn.Linear(h, out_dim)

            def forward(self, x):
                h = self.fc(x)
                return self.backcast_fc(h), self.forecast_fc(h)

        class _NBeats(nn.Module):
            def __init__(
                self, lookback: int, horizon: int, n_stacks: int, hidden: int
            ) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [_Block(lookback, hidden, horizon) for _ in range(n_stacks)]
                )

            def forward(self, x):
                residual = x
                forecasts = []
                for block in self.blocks:
                    back, fore = block(residual)
                    residual = residual - back
                    forecasts.append(fore)
                return torch.stack(forecasts, dim=0).sum(dim=0)

        model = _NBeats(lookback, horizon, stacks, hidden_units).to(device)
        xs = torch.tensor(x_train, dtype=torch.float32).to(device)
        ys = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        last_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Calculate train metrics
        model.eval()
        with torch.no_grad():
            train_preds_scaled = model(xs).cpu().numpy()
        train_mse = float(np.nanmean((train_preds_scaled - y_train_scaled) ** 2))
        train_mae = float(np.nanmean(np.abs(train_preds_scaled - y_train_scaled)))
        self.last_train_info = {
            "loss": last_loss,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
        }

        # Inference
        model.eval()
        history = np.asarray(y_train[-lookback:], dtype=float).reshape(1, -1)
        with torch.no_grad():
            inp = torch.tensor(history, dtype=torch.float32).to(device)
            pred_scaled = model(inp).cpu().numpy()

        pred = pred_scaled[0]
        return np.asarray(pred, dtype=float)

    def fit_predict_lstm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        lookback: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> np.ndarray:
        import torch
        import torch.nn as nn

        self.last_preprocessing_info = _preprocessing_info(
            "LSTM",
        )
        y_train = _to_numpy(train_df["target"])
        horizon = _configured_horizon(forecast_horizon, model_name="LSTM")

        if len(y_train) < lookback + horizon:
            raise ValueError(
                f"LSTM lag={lookback} requires at least "
                f"{lookback + horizon} training points; received {len(y_train)}."
            )
        x_train_raw, y_train_raw = _make_multioutput_lag_features(
            y_train, lookback, horizon
        )

        # Target/series scaling, when required, has already been applied by PreprocessingAgent.
        x_train = x_train_raw
        y_train_scaled = y_train_raw

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_features = 1
        x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)

        xs = x_tensor.to(device)
        ys = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

        class _LSTMModel(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                output_size: int,
                dropout: float,
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        model = _LSTMModel(num_features, hidden_size, num_layers, horizon, dropout).to(
            device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        model.train()
        last_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            train_preds_scaled = model(xs).cpu().numpy()
        train_mse = float(np.nanmean((train_preds_scaled - y_train_scaled) ** 2))
        train_mae = float(np.nanmean(np.abs(train_preds_scaled - y_train_scaled)))
        self.last_train_info = {
            "loss": last_loss,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
        }

        model.eval()
        history = np.asarray(y_train[-lookback:], dtype=float).reshape(1, -1)
        history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(-1)
        

        with torch.no_grad():
            preds = model(history_tensor.to(device))

        preds = preds.cpu().numpy().flatten()
        return np.asarray(preds[:horizon], dtype=float)

    def fit_predict_dlinear(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        lag: int = 24,
        epochs: int = 100,
    ) -> np.ndarray:
        import torch
        import torch.nn as nn

        self.last_preprocessing_info = _preprocessing_info("DLinear")
        y_train = _to_numpy(train_df["target"])
        horizon = _configured_horizon(forecast_horizon, model_name="DLinear")

        if len(y_train) < lag + horizon:
            raise ValueError(
                f"DLinear lag={lag} requires at least {lag + horizon} "
                f"training points; received {len(y_train)}."
            )
        x_train_raw, y_train_raw = _make_multioutput_lag_features(y_train, lag, horizon)

        # Target/series scaling, when required, has already been applied by PreprocessingAgent.
        x_train = x_train_raw
        y_train_scaled = y_train_raw

        class _MovingAverage(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                self.kernel_size = kernel_size
                self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

            def forward(self, x):
                # x shape: (batch, seq_len)
                front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
                end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
                # handle even kernel size padding
                if self.kernel_size % 2 == 0:
                    end = torch.cat([end, x[:, -1:]], dim=1)
                x = torch.cat([front, x, end], dim=1)
                x = x.unsqueeze(1)  # (batch, 1, seq_len)
                res = self.avg(x)
                return res.squeeze(1)

        class _SeriesDecomp(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                self.moving_avg = _MovingAverage(kernel_size)

            def forward(self, x):
                trend = self.moving_avg(x)
                res = x - trend
                return res, trend

        class _DLinear(nn.Module):
            def __init__(self, seq_len, pred_len):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.decomp = _SeriesDecomp(kernel_size=25)
                self.linear_trend = nn.Linear(seq_len, pred_len)
                self.linear_seasonal = nn.Linear(seq_len, pred_len)

            def forward(self, x):
                seasonal_init, trend_init = self.decomp(x)
                trend_part = self.linear_trend(trend_init)
                seasonal_part = self.linear_seasonal(seasonal_init)
                return trend_part + seasonal_part

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _DLinear(lag, horizon).to(device)

        xs = torch.tensor(x_train, dtype=torch.float32).to(device)
        ys = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        model.train()
        last_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(xs)
            loss = criterion(out, ys)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Calculate train metrics
        model.eval()
        with torch.no_grad():
            train_preds_scaled = model(xs).cpu().numpy()
        train_mse = float(np.nanmean((train_preds_scaled - y_train_scaled) ** 2))
        train_mae = float(np.nanmean(np.abs(train_preds_scaled - y_train_scaled)))
        self.last_train_info = {
            "loss": last_loss,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
        }

        model.eval()
        history = np.asarray(y_train[-lag:], dtype=float).reshape(1, -1)
        with torch.no_grad():
            inp = torch.tensor(history, dtype=torch.float32).to(device)
            pred_scaled = model(inp).cpu().numpy()

        pred = pred_scaled[0]
        return np.asarray(pred, dtype=float)

        return np.asarray(pred, dtype=float)

    def fit_predict_reformer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        lag: int = 30,
        epochs: int = 100,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> np.ndarray:
        import torch
        import torch.nn as nn

        self.last_preprocessing_info = _preprocessing_info("Reformer")
        y_train = _to_numpy(train_df["target"])
        horizon = _configured_horizon(forecast_horizon, model_name="Reformer")

        if len(y_train) < lag + horizon:
            raise ValueError(
                f"Reformer lag={lag} requires at least {lag + horizon} "
                f"training points; received {len(y_train)}."
            )
        x_train_raw, y_train_raw = _make_multioutput_lag_features(y_train, lag, horizon)

        # Target/series scaling, when required, has already been applied by PreprocessingAgent.
        x_train = x_train_raw
        y_train_scaled = y_train_raw

        try:
            from transformers import ReformerConfig, ReformerModel
        except ImportError as exc:
            raise RuntimeError(
                "Reformer requires transformers.ReformerModel; no substitute "
                "attention backend is permitted."
            ) from exc

        hidden_size = 32
        if hidden_size % nhead:
            raise ValueError("Reformer nhead must divide hidden_size=32.")
        bucket_size = max(2, 2 ** max(0, (max(2, lag // 4)).bit_length() - 1))
        padded_length = ((lag + (2 * bucket_size) - 1) // (2 * bucket_size)) * (
            2 * bucket_size
        )
        left_padding = padded_length - lag
        reformer_hash_seed = self.random_state

        class _TimeSeriesReformer(nn.Module):
            def __init__(
                self,
                seq_len,
                pred_len,
            ):
                super().__init__()
                self.input_projection = nn.Linear(1, hidden_size)
                config = ReformerConfig(
                    hidden_size=hidden_size,
                    attention_head_size=hidden_size // nhead,
                    num_attention_heads=nhead,
                    feed_forward_size=dim_feedforward,
                    attn_layers=["lsh"] * num_layers,
                    lsh_attn_chunk_length=bucket_size,
                    num_hashes=2,
                    num_buckets=2,
                    axial_pos_embds=False,
                    max_position_embeddings=seq_len,
                    hidden_dropout_prob=0.1,
                    lsh_attention_probs_dropout_prob=0.1,
                    is_decoder=False,
                    hash_seed=reformer_hash_seed,
                )
                self.reformer = ReformerModel(config)
                # Reformer concatenates the two reversible streams.
                self.output_projection = nn.Linear(2 * hidden_size, pred_len)

            def forward(self, x):
                if left_padding:
                    x = torch.nn.functional.pad(x, (left_padding, 0))
                mask = torch.ones_like(x, dtype=torch.long)
                if left_padding:
                    mask[:, :left_padding] = 0
                embedded = self.input_projection(x.unsqueeze(-1))
                encoded = self.reformer(
                    inputs_embeds=embedded,
                    attention_mask=mask,
                    return_dict=True,
                ).last_hidden_state
                return self.output_projection(encoded[:, -1, :])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _TimeSeriesReformer(padded_length, horizon).to(device)

        xs = torch.tensor(x_train, dtype=torch.float32).to(device)
        ys = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        last_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(xs)
            loss = criterion(out, ys)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Calculate train metrics
        model.eval()
        with torch.no_grad():
            train_preds_scaled = model(xs).cpu().numpy()
        train_mse = float(np.nanmean((train_preds_scaled - y_train_scaled) ** 2))
        train_mae = float(np.nanmean(np.abs(train_preds_scaled - y_train_scaled)))
        self.last_train_info = {
            "loss": last_loss,
            "mse": train_mse,
            "mae": train_mae,
            "rmse": float(np.sqrt(train_mse)),
            "backend": "transformers.ReformerModel",
            "attention_type": "lsh",
            "reversible_layers": True,
            "hidden_size": hidden_size,
            "attention_heads": nhead,
            "attention_head_size": hidden_size // nhead,
            "lsh_bucket_size": bucket_size,
            "num_hashes": 2,
            "num_buckets": 2,
            "padded_sequence_length": padded_length,
            "observed_sequence_length": lag,
            "padding_side": "left",
            "attention_scope": "observed_history_only",
        }

        model.eval()
        history = np.asarray(y_train[-lag:], dtype=float).reshape(1, -1)
        with torch.no_grad():
            inp = torch.tensor(history, dtype=torch.float32).to(device)
            pred_scaled = model(inp).cpu().numpy()

        pred = pred_scaled[0]
        return np.asarray(pred, dtype=float)

    def fit_predict_chronos(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        model_size: str = "tiny",
    ) -> np.ndarray:
        import torch

        self.last_preprocessing_info = {
            "model": "Chronos",
            "scaler_used": None,
            "scaler_fitted_on": None,
        }

        y_train = _to_numpy(train_df["target"])
        horizon = _configured_horizon(forecast_horizon, model_name="Chronos")
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = f"amazon/chronos-t5-{model_size}"
        pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        context = torch.tensor(y_train, dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=horizon)
        # forecast shape: (num_samples, horizon) — take median
        median = np.quantile(forecast[0].numpy(), 0.5, axis=0)
        self.last_train_info = {
            "loss": 0.0,
            "mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
        }
        return np.asarray(median, dtype=float)

    def _fit_predict_neuralforecast(
        self,
        model_class: type | None,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        forecast_horizon: int,
        freq: str,
        **kwargs,
    ) -> np.ndarray:
        self.last_preprocessing_info = _preprocessing_info(model_name)

        if model_class is None:
            model_class = _NEURALFORECAST_MODELS[model_name]

        horizon = _configured_horizon(forecast_horizon, model_name=model_name)
        if len(val_df) != horizon:
            raise RuntimeError(
                f"Invalid forecast protocol for {model_name}: evaluation_window_size="
                f"{len(val_df)} but configured_forecast_horizon={horizon}. "
                "Pass exactly one rolling-origin window per fit."
            )

        # Format for NeuralForecast
        nf_df = pd.DataFrame(
            {
                "unique_id": "series1",
                "ds": pd.to_datetime(
                    train_df["date"] if "date" in train_df else range(len(train_df))
                ),
                "y": train_df["target"].values,
            }
        )

        # Target/series scaling, when required, has already been applied by PreprocessingAgent.
        # External normalization is owned by PreprocessingAgent. Disable the
        # backend scaler explicitly to prevent double scaling.
        kwargs["scaler_type"] = "identity"
        model = model_class(h=horizon, batch_size=32, **kwargs)
        effective_horizon = int(getattr(model, "h", horizon))
        if effective_horizon != horizon:
            raise RuntimeError(
                f"Invalid forecast protocol for {model_name}: "
                f"effective_model_horizon={effective_horizon}, "
                f"configured_forecast_horizon={horizon}"
            )
        self.last_forecast_protocol = {
            "model_name": model_name,
            "configured_forecast_horizon": horizon,
            "effective_model_horizon": effective_horizon,
            "evaluation_window_size": int(len(val_df)),
            "horizon_consistency": True,
            "max_steps": int(kwargs["max_steps"]),
        }
        
        model.trainer_kwargs = {
            "accelerator": "auto",
            "devices": 1,
            "max_steps": kwargs.get("max_steps", 100),
            "logger": False,
            "enable_checkpointing": False,
        }
        nf = NeuralForecast(models=[model], freq=freq)

        import logging

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        try:
            nf.fit(df=nf_df)
            forecast_df = nf.predict()
            pred_scaled = forecast_df[model_class.__name__].values

            # Calculate train metrics and training loss of last epoch/step
            last_loss = 0.0
            try:
                if hasattr(model, "trainer") and model.trainer is not None:
                    metrics_dict = model.trainer.callback_metrics
                    for k in ["train_loss", "loss", "train_loss_epoch"]:
                        if k in metrics_dict:
                            last_loss = float(metrics_dict[k].item())
                            break
            except Exception:
                pass

            train_mse = last_loss
            train_mae = 0.0
            try:
                insample = nf.predict_insample()
                if not insample.empty:
                    y_true_train = insample["y"].values
                    y_pred_train = insample[model_class.__name__].values
                    train_mse = float(np.nanmean((y_pred_train - y_true_train) ** 2))
                    train_mae = float(np.nanmean(np.abs(y_pred_train - y_true_train)))
            except Exception:
                pass

            self.last_train_info = {
                "loss": last_loss or train_mse,
                "mse": train_mse,
                "mae": train_mae,
                "rmse": float(np.sqrt(train_mse)),
                "backend_scaler": "identity",
                "external_target_scaling": True,
                "max_steps": int(kwargs["max_steps"]),
            }

            pred = pred_scaled
            if len(pred) != horizon:
                raise RuntimeError(
                    f"Invalid forecast protocol for {model_name}: model returned "
                    f"{len(pred)} predictions, expected {horizon}"
                )
            return np.asarray(pred, dtype=float)
        except Exception as exc:
            raise RuntimeError(
                f"NeuralForecast error ({model_name}): {exc}"
            ) from exc

    def fit_predict_autoformer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 256,
        max_steps: int = 10,
        n_head: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        freq: str = "D",
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "Autoformer",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )

    def fit_predict_informer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        n_head: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "Informer",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )

    def fit_predict_fedformer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        n_head: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "FEDformer",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
            n_head=n_head,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )

    def fit_predict_nhits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "N-HiTS",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            max_steps=max_steps,
        )

    def fit_predict_patchtst(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        n_head: int = 8,
        encoder_layers: int = 2,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "PatchTST",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
            n_head=n_head,
            encoder_layers=encoder_layers,
        )

    def fit_predict_timesnet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "TimesNet",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
        )

    def fit_predict_tide(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        forecast_horizon: int,
        input_size: int = 96,
        hidden_size: int = 512,
        max_steps: int = 10,
        num_decoder_layers: int = 1,
        freq: str = "D"
    ) -> np.ndarray:
        return self._fit_predict_neuralforecast(
            None,
            "TiDE",
            train_df,
            val_df,
            forecast_horizon=forecast_horizon,
            freq=freq,
            input_size=input_size,
            hidden_size=hidden_size,
            max_steps=max_steps,
            num_decoder_layers=num_decoder_layers,
        )

    def train_and_predict(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hyperparameters: Dict[str, object] | None = None,
        *,
        forecast_horizon: int,
        preprocessing_transformations: list | None = None,
        freq: str = "D",
    ) -> np.ndarray:
        """Dispatch training call by model name with strict hyperparameter validation.

        """
        validated_model_feature_columns(
            train_df, context=f"{model_name} training input"
        )
        validated_model_feature_columns(
            val_df, context=f"{model_name} evaluation input"
        )
        canonical_model = _canonical_model_name(model_name)
        configured_horizon = _configured_horizon(
            forecast_horizon, model_name=canonical_model
        )
        if canonical_model in TABULAR_FEATURE_MODELS:
            fitted = self.fit_tabular_model(
                canonical_model,
                train_df,
                hyperparameters,
                preprocessing_transformations=preprocessing_transformations,
            )
            return self.predict_tabular_model(fitted, val_df)
        key = canonical_model.lower().replace("-", "")
        from utils.configuration_identity import normalize_candidate_hyperparameters

        normalized_hp, normalization_audit = normalize_candidate_hyperparameters(
            canonical_model, hyperparameters, preprocessing_transformations
        )
        hp = _validate_runtime_hyperparameters(canonical_model, normalized_hp)
        if normalization_audit:
            self.last_configuration_normalization = normalization_audit

        if key == "arima":
            return self.fit_predict_arima(train_df, val_df, order=tuple(_require_hp(hp, "order")))  # type: ignore[arg-type]
        if key == "sarima":
            return self.fit_predict_sarima(
                train_df,
                val_df,
                order=tuple(_require_hp(hp, "order")),  # type: ignore[arg-type]
                seasonal_order=tuple(_require_hp(hp, "seasonal_order")),  # type: ignore[arg-type]
            )
        if key == "prophet":
            return self.fit_predict_prophet(
                train_df,
                val_df,
                changepoint_prior_scale=float(_require_hp(hp, "changepoint_prior_scale")),
                seasonality_mode=str(_require_hp(hp, "seasonality_mode")),
                freq=freq,
            )
        if key == "ets":
            trend = _require_hp(hp, "trend")
            seasonal = _require_hp(hp, "seasonal")
            seasonal_periods = hp.get("seasonal_periods")
            if seasonal_periods is not None:
                seasonal_periods = int(seasonal_periods)
            return self.fit_predict_ets(
                train_df,
                val_df,
                trend=None if trend is None else str(trend),
                seasonal=None if seasonal is None else str(seasonal),
                seasonal_periods=seasonal_periods,
            )
        if key == "mlp":
            return self.fit_predict_mlp(
                train_df,
                val_df,
                hidden_layer_sizes=tuple(_require_hp(hp, "hidden_layer_sizes")),  # type: ignore[arg-type]
            )
        if key == "lstm":
            return self.fit_predict_lstm(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                lookback=int(_require_hp(hp, "lag")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                num_layers=int(_require_hp(hp, "num_layers")),
                dropout=float(_require_hp(hp, "dropout")),
                epochs=int(_require_hp(hp, "epochs")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
            )
        if key == "xgboost":
            return self.fit_predict_xgboost(
                train_df,
                val_df,
                n_estimators=int(_require_hp(hp, "n_estimators")),
                max_depth=int(_require_hp(hp, "max_depth")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                subsample=float(_optional_hp(hp, "subsample", 1.0)),
                colsample_bytree=float(_optional_hp(hp, "colsample_bytree", 1.0)),
            )
        if key == "randomforest":
            return self.fit_predict_randomforest(
                train_df,
                val_df,
                n_estimators=int(_require_hp(hp, "n_estimators")),
                max_depth=None if _require_hp(hp, "max_depth") is None else int(_require_hp(hp, "max_depth")),
            )
        if key == "lightgbm":
            return self.fit_predict_lightgbm(
                train_df,
                val_df,
                n_estimators=int(_require_hp(hp, "n_estimators")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                max_depth=int(_require_hp(hp, "max_depth")),
            )
        if key == "catboost":
            return self.fit_predict_catboost(
                train_df,
                val_df,
                iterations=int(_require_hp(hp, "iterations")),
                learning_rate=float(_require_hp(hp, "learning_rate")),
                depth=int(_require_hp(hp, "depth")),
            )
        if key == "nbeats":
            return self.fit_predict_nbeats(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                lookback=int(_require_hp(hp, "lookback")),
                hidden_units=int(_require_hp(hp, "hidden_units")),
                stacks=int(_require_hp(hp, "stacks")),
                epochs=int(_require_hp(hp, "epochs")),
            )
        if key == "autoformer":
            return self.fit_predict_autoformer(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                n_head=int(_require_hp(hp, "n_head")),
                encoder_layers=int(_require_hp(hp, "encoder_layers")),
                decoder_layers=int(_require_hp(hp, "decoder_layers")),
                freq=freq,
            )
        if key == "informer":
            return self.fit_predict_informer(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                n_head=int(_require_hp(hp, "n_head")),
                encoder_layers=int(_require_hp(hp, "encoder_layers")),
                decoder_layers=int(_require_hp(hp, "decoder_layers")),
                freq=freq,
            )
        if key == "fedformer":
            return self.fit_predict_fedformer(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                n_head=int(_require_hp(hp, "n_head")),
                encoder_layers=int(_require_hp(hp, "encoder_layers")),
                decoder_layers=int(_require_hp(hp, "decoder_layers")),
                freq=freq,
            )
        if key == "nhits":
            return self.fit_predict_nhits(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                freq=freq,
            )
        if key == "patchtst":
            return self.fit_predict_patchtst(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                n_head=int(_require_hp(hp, "n_head")),
                encoder_layers=int(_require_hp(hp, "encoder_layers")),
                freq=freq,
            )
        if key == "timesnet":
            return self.fit_predict_timesnet(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                freq=freq,
            )
        if key == "tide":
            return self.fit_predict_tide(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                input_size=int(_require_hp(hp, "input_size")),
                hidden_size=int(_require_hp(hp, "hidden_size")),
                max_steps=int(_require_hp(hp, "max_steps")),
                num_decoder_layers=int(_require_hp(hp, "num_decoder_layers")),
                freq=freq,
            )
        if key == "elasticnet":
            return self.fit_predict_elasticnet(
                train_df,
                val_df,
                alpha=float(_require_hp(hp, "alpha")),
                l1_ratio=float(_require_hp(hp, "l1_ratio")),
                fit_intercept=bool(_optional_hp(hp, "fit_intercept", True)),
                max_iter=int(_optional_hp(hp, "max_iter", 1000)),
                tol=float(_optional_hp(hp, "tol", 1e-4)),
                selection=str(_optional_hp(hp, "selection", "cyclic")),
                positive=bool(_optional_hp(hp, "positive", False)),
            )
        if key == "knn":
            return self.fit_predict_knn(
                train_df,
                val_df,
                n_neighbors=int(_require_hp(hp, "n_neighbors")),
                weights=str(_optional_hp(hp, "weights", "uniform")),
                algorithm=str(_optional_hp(hp, "algorithm", "auto")),
                leaf_size=int(_optional_hp(hp, "leaf_size", 30)),
                p=int(_optional_hp(hp, "p", 2)),
                metric=str(_optional_hp(hp, "metric", "minkowski")),
            )
        if key == "svr":
            gamma_value = _optional_hp(hp, "gamma", "scale")
            if isinstance(gamma_value, str):
                gamma = gamma_value
            else:
                gamma = float(gamma_value)
            return self.fit_predict_svr(
                train_df,
                val_df,
                C=float(_require_hp(hp, "C")),
                epsilon=float(_require_hp(hp, "epsilon")),
                kernel=str(_require_hp(hp, "kernel")),
                gamma=gamma,
                degree=int(_optional_hp(hp, "degree", 3)),
                coef0=float(_optional_hp(hp, "coef0", 0.0)),
                shrinking=bool(_optional_hp(hp, "shrinking", True)),
                tol=float(_optional_hp(hp, "tol", 1e-3)),
                cache_size=float(_optional_hp(hp, "cache_size", 200.0)),
                max_iter=int(_optional_hp(hp, "max_iter", -1)),
            )
        if key == "dlinear":
            return self.fit_predict_dlinear(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                lag=int(_require_hp(hp, "lag")),
                epochs=int(_require_hp(hp, "epochs")),
            )
        if key == "reformer":
            return self.fit_predict_reformer(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                lag=int(_require_hp(hp, "lag")),
                epochs=int(_require_hp(hp, "epochs")),
                nhead=int(_require_hp(hp, "nhead")),
                num_layers=int(_require_hp(hp, "num_layers")),
                dim_feedforward=int(_require_hp(hp, "dim_feedforward")),
            )
        if key == "chronos":
            return self.fit_predict_chronos(
                train_df,
                val_df,
                forecast_horizon=configured_horizon,
                model_size=str(_require_hp(hp, "model_size")),
            )
        raise ValueError(f"Unsupported model: {model_name}")
