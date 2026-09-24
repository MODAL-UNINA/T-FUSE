"""Deterministic mapping from measured analytical output to family compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from utils.model_family_state import CANONICAL_FAMILY_ORDER
from .fuzzy_membership import clamp


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit(value: Any) -> float | None:
    number = _finite(value)
    return clamp(number) if number is not None else None


@dataclass
class AnalyticalFamilyCompatibilityMapper:
    """Map only explicitly measured AnalyticalAgent fields; never training results."""

    reliability_weights: Mapping[str, float] = field(
        default_factory=lambda: {"coverage": 0.50, "quality": 0.25, "validity": 0.25}
    )
    model_registry: Mapping[str, Any] | None = None

    def map(self, analytical_output: Mapping[str, Any] | None) -> dict[str, Any]:
        source = analytical_output if isinstance(analytical_output, Mapping) else {}
        base = source.get("base_metrics", source)
        base = base if isinstance(base, Mapping) else {}
        n = _finite(base.get("n_obs"))
        horizon = _finite(source.get("forecast_horizon"))
        trend_raw = _finite(base.get("trend_strength"))
        trend = clamp(trend_raw / 3.0) if trend_raw is not None else None
        seasonality = _unit(base.get("seasonality_acf"))
        volatility_raw = _finite(base.get("volatility_cv"))
        volatility = clamp(volatility_raw / 2.0) if volatility_raw is not None else None
        confidence = _unit(source.get("analysis_confidence"))
        frequency = base.get("inferred_freq")
        regularity = None if frequency in (None, "", "unknown") else 1.0
        acfs = base.get("autocorrelations")
        valid_acfs = [_finite(v) for v in acfs.values()] if isinstance(acfs, Mapping) else []
        valid_acfs = [abs(v) for v in valid_acfs if v is not None]
        autocorrelation = clamp(max(valid_acfs)) if valid_acfs else None
        seasonal_period = _finite(base.get("seasonal_periods"))
        suggested_lag = _finite(base.get("suggested_lag"))
        horizon_ratio = clamp(horizon / n) if n and horizon is not None and n > 0 else None

        metrics = {
            "n_observations": n, "frequency": frequency if regularity is not None else None,
            "forecast_horizon": horizon, "horizon_ratio": horizon_ratio,
            "trend_strength": trend, "seasonality_strength": seasonality,
            "seasonal_period": seasonal_period, "autocorrelation_strength": autocorrelation,
            "volatility": volatility, "analysis_confidence": confidence,
            "sampling_regularity": regularity, "suggested_lag": suggested_lag,
        }
        expected = ("n_observations", "trend_strength", "seasonality_strength",
                    "autocorrelation_strength", "analysis_confidence", "sampling_regularity")
        valid_count = sum(metrics[key] is not None for key in expected)
        coverage = valid_count / len(expected)
        quality = confidence if confidence is not None else coverage
        validity = 1.0 if valid_count else 0.0
        weights = self._weights()
        reliability = clamp(weights["coverage"] * coverage + weights["quality"] * quality + weights["validity"] * validity)
        rel_source = "analysis_confidence_and_metric_coverage" if confidence is not None else "valid_metric_coverage"

        results = {}
        for family in CANONICAL_FAMILY_ORDER:
            score, rules, required = self._family_score(
                family, n=n, horizon=horizon, horizon_ratio=horizon_ratio, trend=trend,
                seasonality=seasonality, autocorrelation=autocorrelation,
                regularity=regularity, suggested_lag=suggested_lag,
            )
            used = [key for key in required if metrics.get(key) is not None]
            missing = [key for key in required if metrics.get(key) is None]
            uncertain = (not used or (family in {"Neural", "Transformer/Foundation"} and len(used) < 2)
                         or (family == "Transformer/Foundation" and not self.model_registry))
            if uncertain:
                score = 0.5
            results[family] = {
                "family": family, "analytical_compatibility": clamp(score),
                "analytical_reliability": reliability,
                "compatibility_status": "uncertain" if uncertain else "ready",
                "metrics_used": used, "missing_metrics": missing,
                "rules_applied": rules if not uncertain else [*rules, "insufficient_explicit_evidence_neutral"],
                "analytical_reliability_source": rel_source,
                "analytical_reliability_components": {"coverage": coverage, "quality": quality, "validity": validity, "weights": weights},
            }
        return {"status": "ready" if valid_count else "unavailable", "family_results": results}

    def _weights(self) -> dict[str, float]:
        raw = {key: max(0.0, float(self.reliability_weights.get(key, 0.0))) for key in ("coverage", "quality", "validity")}
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()} if total else {"coverage": 0.5, "quality": 0.25, "validity": 0.25}

    def _family_score(self, family: str, **m: Any) -> tuple[float, list[str], tuple[str, ...]]:
        def avg(values: list[float | None], neutral: float = 0.5) -> float:
            known = [v for v in values if v is not None]
            return sum(known) / len(known) if known else neutral
        n, horizon = m["n"], m["horizon"]
        history = clamp(n / max(30.0, 4.0 * horizon)) if n is not None and horizon else (clamp(n / 60.0) if n is not None else None)
        lag_ready = clamp((n - max(1.0, m["suggested_lag"] or 1.0)) / 60.0) if n is not None else None
        if family == "Statistical":
            return avg([m["autocorrelation"], m["seasonality"], m["trend"], m["regularity"], history]), ["measured_structure"], ("autocorrelation_strength", "seasonality_strength", "trend_strength", "sampling_regularity", "n_observations")
        if family == "Additive":
            return avg([m["trend"], m["seasonality"], m["regularity"]]), ["measured_decomposable_structure"], ("trend_strength", "seasonality_strength", "sampling_regularity")
        if family == "Tree-based":
            measured = avg([lag_ready, m["regularity"]])
            return min(0.65, 0.45 + 0.20 * measured), ["lag_and_temporal_readiness", "no_nonlinearity_bonus_without_measurement"], ("n_observations", "suggested_lag", "sampling_regularity", "nonlinearity_score")
        if family == "Classical ML":
            return min(0.75, 0.45 + 0.30 * avg([lag_ready, history])), ["valid_supervised_lag_examples"], ("n_observations", "suggested_lag", "forecast_horizon")
        if family == "Neural":
            sample = avg([history, None if m["horizon_ratio"] is None else 1.0 - m["horizon_ratio"]])
            return min(0.80, 0.50 + 0.30 * sample), ["explicit_sample_and_horizon_evidence", "no_runtime_or_complexity_bonus_without_measurement"], ("n_observations", "horizon_ratio")
        registry_known = bool(self.model_registry)
        return (avg([history, None if m["horizon_ratio"] is None else 1.0 - m["horizon_ratio"]]) if registry_known else 0.5), ["pretrained_and_from_scratch_are_distinct", "registry_metadata_available" if registry_known else "registry_metadata_missing"], ("n_observations", "horizon_ratio", "runtime_available", "pretrained")


__all__ = ["AnalyticalFamilyCompatibilityMapper"]
