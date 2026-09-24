"""Robust detection and replacement of training outliers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_REPLACE_OUTLIERS: dict[str, Any] = {
    "enabled": True,
    "direction": "upper",
    "detection_method": "robust_mad",
    "threshold": 8.0,
    "max_replacements": 2,
    "replacement_method": "rolling_median",
    "replacement_window": 3,
    "trace_value_limit": 100,
}


def normalize_replace_outliers_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate fixed framework settings for ``replace_outliers``."""
    result = dict(DEFAULT_REPLACE_OUTLIERS)
    if config:
        result.update(dict(config))
    result["enabled"] = bool(result["enabled"])
    result["direction"] = str(result["direction"]).strip().lower()
    result["detection_method"] = str(result["detection_method"]).strip().lower()
    result["replacement_method"] = str(result["replacement_method"]).strip().lower()
    result["threshold"] = float(result["threshold"])
    result["max_replacements"] = int(result["max_replacements"])
    result["replacement_window"] = int(result["replacement_window"])
    result["trace_value_limit"] = int(result["trace_value_limit"])
    if result["direction"] != "upper":
        raise ValueError("replace_outliers.direction must be 'upper'")
    if result["detection_method"] != "robust_mad":
        raise ValueError("replace_outliers supports detection_method='robust_mad'")
    if result["replacement_method"] != "rolling_median":
        raise ValueError("replace_outliers supports replacement_method='rolling_median'")
    if not np.isfinite(result["threshold"]) or result["threshold"] <= 0:
        raise ValueError("replace_outliers.threshold must be positive")
    if result["max_replacements"] < 1:
        raise ValueError("replace_outliers.max_replacements must be >= 1")
    if result["replacement_window"] < 1:
        raise ValueError("replace_outliers.replacement_window must be >= 1")
    if result["trace_value_limit"] < 0:
        raise ValueError("replace_outliers.trace_value_limit must be >= 0")
    return result


def detect_extreme_outliers(train_df: pd.DataFrame, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Detect extreme outliers from the supplied training history only."""
    cfg = normalize_replace_outliers_config(config)
    diagnostics: dict[str, Any] = {"found": False, "count": 0, "indices": [], "last_index": None, "variance_impact": 0.0, "action": "none_no_extreme_outliers", "detection_scope": "training_history_only", "observations_scanned": 0, "impact_estimation": "temporary_median_replacement", "training_outlier_count": 0, "candidates_above_threshold": 0, "candidate_indices_above_threshold": [], "max_replacements": cfg["max_replacements"], "selection_policy": "largest_robust_score", "training_outlier_fraction": 0.0, "extreme_outlier_detected": False, "outlier_strength": 0.0, "training_outlier_thresholds": {"method": cfg["detection_method"], "threshold": cfg["threshold"], "direction": cfg["direction"], "center": None, "scale": None, "mad": None}}
    if not cfg["enabled"] or not isinstance(train_df, pd.DataFrame) or "target" not in train_df:
        return diagnostics
    target = pd.to_numeric(train_df["target"], errors="coerce")
    finite = target[np.isfinite(target)]
    diagnostics["observations_scanned"] = int(len(finite))
    if len(finite) < 3:
        return diagnostics
    center = float(finite.median())
    mad = float(np.median(np.abs(finite.to_numpy(dtype=float) - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = np.finfo(float).eps * max(1.0, abs(center)) * 16.0
    robust_z = (target - center) / scale
    mask = (robust_z > cfg["threshold"]).fillna(False)
    candidate_positions = np.flatnonzero(mask.to_numpy()).astype(int).tolist()
    ranked = sorted(candidate_positions, key=lambda position: abs(float(robust_z.iloc[position])), reverse=True)
    positions = sorted(ranked[:cfg["max_replacements"]])
    diagnostics.update({"candidates_above_threshold": len(candidate_positions), "candidate_indices_above_threshold": candidate_positions, "training_outlier_thresholds": {"method": cfg["detection_method"], "threshold": cfg["threshold"], "direction": cfg["direction"], "center": center, "scale": float(scale), "mad": mad}})
    if not positions:
        return diagnostics
    original = target.to_numpy(dtype=float)
    reference = original.copy()
    reference[positions] = center
    variance_full = float(np.nanvar(original))
    variance_reference = float(np.nanvar(reference))
    variance_impact = max(0.0, (variance_full - variance_reference) / variance_full) if variance_full > 0 else 0.0
    diagnostics.update({"found": True, "count": len(positions), "indices": positions, "last_index": positions[-1], "variance_impact": round(variance_impact, 6), "action": "replace_outliers", "training_outlier_count": len(positions), "training_outlier_fraction": float(len(positions) / len(finite)), "extreme_outlier_detected": True, "outlier_strength": float(np.nanmax(np.abs(robust_z.to_numpy(dtype=float)[positions])) )})
    return diagnostics


def replace_outliers(train_df: pd.DataFrame, config: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect and replace outliers without consulting future observations."""
    cfg = normalize_replace_outliers_config(config)
    raw = train_df.copy(deep=True)
    diagnostics = detect_extreme_outliers(raw, cfg)
    processed = raw.copy(deep=True)
    processed["target"] = pd.to_numeric(processed["target"], errors="coerce").astype(float)
    target = pd.to_numeric(raw["target"], errors="coerce").astype(float)
    positions = [int(position) for position in diagnostics.get("indices", [])]
    outlier_positions = set(positions)
    radius = cfg["replacement_window"]
    finite_non_outlier = np.isfinite(target.to_numpy(dtype=float))
    if outlier_positions:
        finite_non_outlier[list(outlier_positions)] = False
    fallback_values = target.iloc[np.flatnonzero(finite_non_outlier)]
    global_fallback = float(fallback_values.median()) if not fallback_values.empty else float(target.median())
    originals: list[float] = []
    replacements: list[float] = []
    for position in positions:
        lower, upper = max(0, position - radius), min(len(target), position + radius + 1)
        neighbours = [float(target.iloc[index]) for index in range(lower, upper) if index != position and index not in outlier_positions and np.isfinite(target.iloc[index])]
        if neighbours:
            replacement = float(np.median(neighbours))
        else:
            prior = [float(target.iloc[index]) for index in range(position - 1, -1, -1) if index not in outlier_positions and np.isfinite(target.iloc[index])]
            replacement = prior[0] if prior else global_fallback
        originals.append(float(target.iloc[position]))
        replacements.append(replacement)
        processed.iloc[position, processed.columns.get_loc("target")] = replacement
    if len(processed) != len(raw):
        raise RuntimeError("replace_outliers changed row count")
    limit = cfg["trace_value_limit"]
    thresholds = diagnostics["training_outlier_thresholds"]
    dates = [str(raw.iloc[position]["date"]) if "date" in raw else position for position in positions[:limit]]
    trace = {"name": "replace_outliers", "transformation": "replace_outliers", "enabled": cfg["enabled"], "applied": bool(replacements), "detection_scope": "training_history_only", "detected_candidate_count": int(diagnostics.get("candidates_above_threshold", len(positions))), "replaced_count": len(replacements), "indices": positions[:limit], "dates": dates, "original_values": originals[:limit], "replacement_values": replacements[:limit], "robust_center": thresholds.get("center"), "mad": thresholds.get("mad"), "robust_scale": thresholds.get("scale"), "detection_method": cfg["detection_method"], "threshold": cfg["threshold"], "direction": cfg["direction"], "max_replacements": cfg["max_replacements"], "selection_policy": "largest_robust_score", "replacement_method": cfg["replacement_method"], "replacement_window": radius, "validation_target_modified": False, "test_target_modified": False, "diagnostics": diagnostics, "parameters": {"outlier_positions": positions[:limit], "replacement_values": replacements[:limit], "original_values": originals[:limit], "detection_scope": "training_history_only", "robust_center": thresholds.get("center"), "mad": thresholds.get("mad"), "robust_scale": thresholds.get("scale"), "detected_candidate_count": int(diagnostics.get("candidates_above_threshold", len(positions)),), "replaced_count": len(replacements)}, "reason": "Deterministic technical requirement fitted on training history only.", "validation": "passed"}
    return processed, trace


__all__ = ["DEFAULT_REPLACE_OUTLIERS", "detect_extreme_outliers", "normalize_replace_outliers_config", "replace_outliers"]
