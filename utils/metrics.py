"""Forecasting metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def mase_scale(y_train: np.ndarray, seasonality: int = 1) -> float | None:
    """Return the in-sample naive MAE used as the MASE denominator."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(y_train) <= seasonality:
        return None
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    naive_mae = float(np.mean(naive_errors))
    return naive_mae if naive_mae >= 1e-10 else None


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    MASE = MAE_model / MAE_naive_in_sample

    The naive reference is the seasonal random walk:
        naive(t) = y(t - seasonality)

    When seasonality=1 this reduces to the standard random-walk (y[t] = y[t-1]).

    A MASE < 1 means the model beats the naive reference — the primary goal.
    A MASE = 1 means the model is equivalent to the naive.
    Scale-independent: directly comparable across series of different magnitudes.
    No divide-by-zero risk (uses in-sample naive error, not actuals on val set).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    naive_mae = mase_scale(y_train, seasonality=seasonality)
    if naive_mae is None:
        raise ValueError(
            "MASE is undefined: the raw training reference is too short or "
            "its naive error scale is zero."
        )

    model_mae = float(mae(y_true, y_pred))
    return model_mae / naive_mae


def choose_mase_seasonality(
    inferred_freq: str | None,
    seasonality_detected: bool,
    seasonality_period: int | None,
    suggested_lag: int | None,
    y_train_len: int,
) -> tuple[int, str]:
    """Choose the MASE denominator from the authoritative seasonality flag.

    A feature-engineering lookback (``suggested_lag``) is intentionally not a
    seasonality estimate. If deterministic analysis does not detect seasonality,
    MASE uses the non-seasonal naive reference (m=1).
    """
    if not bool(seasonality_detected):
        return 1, "nonseasonal_detector"
    if isinstance(seasonality_period, int) and seasonality_period > 1:
        if y_train_len >= 2 * seasonality_period:
            return seasonality_period, "detected_seasonality_period"
    return 1, "detected_period_unavailable"


def build_mase_reference(
    y_train: np.ndarray,
    *,
    inferred_freq: str | None,
    seasonality_detected: bool,
    seasonality_period: int | None,
    suggested_lag: int | None,
    source: str = "raw_observed_training",
) -> dict:
    """Build the immutable MASE reference from an unprocessed training target."""
    values = np.asarray(y_train, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("MASE reference requires finite raw observed targets.")
    try:
        detected_period = (
            int(seasonality_period) if seasonality_period is not None else None
        )
    except (TypeError, ValueError):
        detected_period = None
    period, period_source = choose_mase_seasonality(
        inferred_freq=inferred_freq,
        seasonality_detected=seasonality_detected,
        seasonality_period=detected_period,
        suggested_lag=suggested_lag,
        y_train_len=len(values),
    )
    denominator = mase_scale(values, seasonality=period)
    return {
        "period": int(period),
        "period_source": period_source,
        "denominator": denominator,
        "defined": denominator is not None,
        "source": source,
        "training_size": int(len(values)),
    }

def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    mase_seasonality: int,
    model_name: str | None = None,
    validation_protocol: str | None = None,
) -> dict:
    """Compute standard forecasting metrics and validate array lengths."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction/target length mismatch for {model_name}: "
            f"len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}, "
            f"validation_protocol={validation_protocol}"
        )

    return {
        "mase": float(mase(y_true, y_pred, y_train, seasonality=mase_seasonality)),
        "rmse": float(rmse(y_true, y_pred)),
        "mse": float(mse(y_true, y_pred)),
        "r2": float(r2(y_true, y_pred)),
        "mae": float(mae(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred)),
    }
