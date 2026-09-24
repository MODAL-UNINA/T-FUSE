"""Runtime reporting metrics for the final T-FUSE holdout forecast."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def compute_train_only_minmax_test_metrics(
    y_train: Sequence[float],
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, Any]:
    """Report final test metrics using a scaler fit only on search-training data."""
    train = np.asarray(list(y_train), dtype=float).reshape(-1)
    truth = np.asarray(list(y_true), dtype=float).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=float).reshape(-1)
    if train.size == 0:
        raise ValueError("Final metric scaling requires non-empty training targets.")
    if truth.size == 0 or pred.size == 0:
        raise ValueError("Final metrics require non-empty test targets and predictions.")
    if truth.size != pred.size:
        raise ValueError(
            "Final metric cardinality mismatch: "
            f"targets={truth.size}, predictions={pred.size}."
        )
    if not np.isfinite(train).all():
        raise ValueError("Final metric scaling requires finite training targets.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Final metrics require finite test targets and predictions.")

    train_min = float(np.min(train))
    train_max = float(np.max(train))
    if train_max == train_min:
        raise ValueError(
            "Final metric scaling is undefined because training targets are constant "
            "(train_min == train_max)."
        )

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train.reshape(-1, 1))
    truth_scaled = scaler.transform(truth.reshape(-1, 1)).reshape(-1)
    pred_scaled = scaler.transform(pred.reshape(-1, 1)).reshape(-1)
    errors = truth_scaled - pred_scaled
    test_mse = float(np.mean(np.square(errors)))
    return {
        "test_mse": test_mse,
        "test_mae": float(np.mean(np.abs(errors))),
        "test_rmse": float(np.sqrt(test_mse)),
        "final_metric_metadata": {
            "scale": "train_only_minmax_[0,1]",
            "scaler_fit_partition": "training",
            "train_min": train_min,
            "train_max": train_max,
        },
    }


__all__ = ["compute_train_only_minmax_test_metrics"]
