"""Strict index and status helpers for final test evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def align_test_predictions(
    test_df: pd.DataFrame,
    predictions: Any,
    *,
    prediction_index: pd.Index | None = None,
) -> tuple[pd.Series, pd.Series]:
    if not isinstance(test_df, pd.DataFrame) or test_df.empty:
        raise ValueError("Final test frame must be non-empty")
    if "date" not in test_df or "target" not in test_df:
        raise ValueError("Final test frame must contain date and target")
    expected_index = pd.DatetimeIndex(pd.to_datetime(test_df["date"]))
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) != len(expected_index):
        raise ValueError(
            f"Final test alignment mismatch: y_true={len(expected_index)}, "
            f"y_pred={len(values)}"
        )
    if prediction_index is not None:
        actual_index = pd.DatetimeIndex(pd.to_datetime(prediction_index))
        if not actual_index.equals(expected_index):
            raise ValueError("Final prediction timestamps differ from test timestamps")
    y_true = pd.Series(
        test_df["target"].to_numpy(dtype=float),
        index=expected_index,
        name="y_true",
    )
    y_pred = pd.Series(values, index=expected_index, name="y_pred")
    return y_true, y_pred


def explicit_final_failure(exc: Exception) -> dict[str, Any]:
    return {
        "run_status": "final_evaluation_failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


__all__ = [
    "align_test_predictions",
    "explicit_final_failure",
]
