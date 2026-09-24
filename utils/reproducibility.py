"""Deterministic fingerprints used by candidate-training audits."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.numeric_input_policy import validated_model_feature_columns


def array_fingerprint(value: Any) -> str:
    """Hash an array including shape and dtype, with stable object handling."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    if array.dtype.hasobject:
        digest.update(pd.util.hash_array(array.reshape(-1), categorize=True).tobytes())
    else:
        digest.update(np.ascontiguousarray(array).view(np.uint8).tobytes())
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash values, column names/dtypes and row index deterministically."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def parameter_fingerprint(arrays: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(array_fingerprint(value).encode("ascii"))
    return digest.hexdigest()


def estimator_fingerprint(model: Any) -> str | None:
    """Return a deterministic fitted-parameter hash for supported estimators."""
    if hasattr(model, "coefs_") and hasattr(model, "intercepts_"):
        return parameter_fingerprint([*model.coefs_, *model.intercepts_])
    if hasattr(model, "get_booster"):
        return hashlib.sha256(bytes(model.get_booster().save_raw())).hexdigest()
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        hashes = [estimator_fingerprint(item) for item in np.asarray(estimators).reshape(-1)]
        hashes = [item for item in hashes if item is not None]
        if hashes:
            return hashlib.sha256("|".join(hashes).encode("ascii")).hexdigest()
    return None


def tabular_training_fingerprints(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    raw_train_df: pd.DataFrame | None = None,
    raw_eval_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    features = validated_model_feature_columns(
        train_df, context="Reproducibility audit training input"
    )
    eval_features = validated_model_feature_columns(
        eval_df, context="Reproducibility audit evaluation input"
    )
    if eval_features != features:
        raise ValueError(
            "Training and evaluation feature columns differ in reproducibility audit."
        )
    result: dict[str, Any] = {
        "train_transformed_df": dataframe_fingerprint(train_df),
        "eval_transformed_df": dataframe_fingerprint(eval_df),
        "X_train_transformed": array_fingerprint(train_df[features].to_numpy()),
        "y_train_transformed": array_fingerprint(train_df["target"].to_numpy()),
        "X_eval_transformed": array_fingerprint(eval_df[features].to_numpy()),
        "y_eval_transformed": array_fingerprint(eval_df["target"].to_numpy()),
        "feature_columns": features,
        "final_train_size": int(len(train_df)),
        "final_train_indices_hash": array_fingerprint(train_df.index.to_numpy()),
    }
    if "date" in train_df and len(train_df):
        dates = pd.to_datetime(train_df["date"])
        result["final_train_start"] = dates.iloc[0].isoformat()
        result["final_train_end"] = dates.iloc[-1].isoformat()
    if isinstance(raw_train_df, pd.DataFrame):
        result["raw_train_df"] = dataframe_fingerprint(raw_train_df)
        result["y_train_raw"] = array_fingerprint(raw_train_df["target"].to_numpy())
    if isinstance(raw_eval_df, pd.DataFrame):
        result["raw_eval_df"] = dataframe_fingerprint(raw_eval_df)
        result["y_eval_raw"] = array_fingerprint(raw_eval_df["target"].to_numpy())
    return result


__all__ = ["array_fingerprint", "dataframe_fingerprint", "estimator_fingerprint", "parameter_fingerprint", "tabular_training_fingerprints"]
