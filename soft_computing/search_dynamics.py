"""Continuous empirical signals for the FUZZY search controller only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

from soft_computing.fuzzy_membership import clamp
from utils.model_family_state import MODEL_TO_FAMILY


@dataclass(frozen=True)
class SearchDynamics:
    incumbent_stability: float
    model_stagnation: float
    global_recent_information_gain: float
    global_search_stagnation: float
    family_saturation: float
    model_refinement_maturity: float
    incumbent_quality: float
    absolute_incumbent_quality: float
    relative_incumbent_quality: float
    family_coverage: float
    distinct_model_coverage: float
    cross_family_diversity: float
    empirical_evidence_strength: float
    semantic_empirical_misalignment: float
    trials_since_last_improvement: int
    current_family_attempts: int
    current_model_refinements: int
    recent_relative_improvement: float
    challenger_model: str | None
    challenger_mase: float | None
    challenger_relative_gap: float | None
    challenger_potential: float
    challenger_refinement_maturity: float
    refinement_responsiveness: float
    remaining_refinement_capacity: float
    unresolved_analytical_evidence: float
    analytical_evidence_urgency: float
    model_refinement_information_gain: float
    refinement_information_gain: float
    intra_family_model_coverage: float
    model_coverage_by_family: Mapping[str, float]
    models_available_by_family: Mapping[str, tuple[str, ...]]
    models_evaluated_by_family: Mapping[str, tuple[str, ...]]
    models_unevaluated_by_family: Mapping[str, tuple[str, ...]]
    family_model_residual_potential: Mapping[str, float]
    global_model_residual_potential: float
    strongest_model_residual_family: str | None
    family_residual_potential: float
    residual_search_value: float
    preprocessing_refinement_potential: float
    raw_challenger_potential: float
    effective_challenger_potential: float
    incumbent_dominance: float
    dominance_confidence: float
    evidence_supported_dominance: float
    search_exhaustion: float
    expected_search_value: float
    explore_model_in_family_potential: float
    late_breakthrough: float
    remaining_budget: int
    remaining_budget_ratio: float
    models_evaluated_in_family: tuple[str, ...]
    eligible_models_in_family: tuple[str, ...]
    unresolved_analytical_signals: tuple[str, ...]
    resolved_analytical_signals: tuple[str, ...]
    current_model: str | None
    current_family: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score(item: Mapping[str, Any]) -> float:
    try:
        value = float(item.get("mase"))
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def _signature(item: Mapping[str, Any]) -> str:
    existing = item.get("configuration_signature")
    if existing:
        return str(existing)
    return json.dumps(
        {
            "model": item.get("model"),
            "hyperparameters": item.get("hyperparameters", {}),
            "preprocessing": item.get("preprocessing_transformations", []),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _relative_improvements(
    values: Sequence[float], threshold: float
) -> tuple[list[float], int]:
    improvements: list[float] = []
    incumbent = math.inf
    since = 0
    for value in values:
        if not math.isfinite(incumbent):
            improvements.append(0.0)
            incumbent = value
            since = 0
            continue
        improvement = max(0.0, (incumbent - value) / (abs(incumbent) + 1e-12))
        improvements.append(improvement)
        if improvement >= threshold:
            since = 0
        else:
            since += 1
        incumbent = min(incumbent, value)
    return improvements, since


def _smoothstep(value: float, lower: float, upper: float) -> float:
    """Continuous S-curve from zero at ``lower`` to one at ``upper``."""
    if upper <= lower:
        return float(value >= upper)
    x = clamp((value - lower) / (upper - lower))
    return x * x * (3.0 - 2.0 * x)


def _refinement_maturity(
    trials: Sequence[Mapping[str, Any]], max_refinements: int
) -> float:
    refinements = max(0, len(trials) - 1)
    distinct = len({_signature(item) for item in trials})
    target = max(2, int(max_refinements) - 1)
    return clamp(
        0.65 * clamp(refinements / target)
        + 0.35 * clamp(max(0, distinct - 1) / target)
    )

def _preprocessing_signature(item: Mapping[str, Any]) -> str:
    return json.dumps(
        item.get("preprocessing_transformations", []) or [],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _information_gain(
    trials: Sequence[Mapping[str, Any]], threshold: float
) -> float:
    """Information in recent same-model refinements, independent of test data."""
    recent = list(trials[-4:])
    if len(recent) <= 1:
        return 1.0
    values = [_score(item) for item in recent]
    relative_changes = [
        abs(right - left) / (abs(left) + 1e-12)
        for left, right in zip(values, values[1:])
    ]
    mean_change = sum(relative_changes) / len(relative_changes)
    mean_score = sum(values) / len(values)
    dispersion = (
        math.sqrt(sum((value - mean_score) ** 2 for value in values) / len(values))
        / (abs(mean_score) + 1e-12)
    )
    improvements, _ = _relative_improvements(values, threshold)
    effective_improvement = max(improvements[1:], default=0.0)
    configuration_diversity = len({_signature(item) for item in recent}) / len(recent)
    change_signal = _smoothstep(mean_change, threshold * 0.25, 0.05)
    dispersion_signal = _smoothstep(dispersion, threshold * 0.20, 0.03)
    improvement_signal = _smoothstep(
        effective_improvement, threshold * 0.50, 0.10
    )
    directional_value = 0.20 + 0.80 * improvement_signal
    raw_gain = clamp(
        0.25 * change_signal * directional_value
        + 0.20 * dispersion_signal * directional_value
        + 0.50 * improvement_signal
        + 0.05 * configuration_diversity
    )
    # Distinct configurations are not informative by themselves when they
    # repeatedly produce the same validation score.  Decay low-signal runs
    # smoothly as consecutive refinements accumulate.
    consecutive_pressure = _smoothstep(len(recent) - 1, 1.0, 3.0)
    numerical_signal = max(
        change_signal * directional_value,
        dispersion_signal * directional_value,
        improvement_signal,
    )
    redundancy = consecutive_pressure * (1.0 - numerical_signal)
    return clamp(raw_gain * (1.0 - 0.70 * redundancy))


def _global_progress(
    values: Sequence[float], threshold: float, window: int
) -> tuple[float, float, int]:
    """Recent global best-so-far gain and gradual search stagnation."""
    improvements, trials_since = _relative_improvements(values, threshold)
    if len(values) < 2:
        return 1.0, 0.0, trials_since

    size = max(2, int(window))
    recent = improvements[-size:]
    meaningful = [
        _smoothstep(value, threshold, max(0.05, 10.0 * threshold))
        for value in recent
    ]
    weights = list(range(1, len(meaningful) + 1))
    global_recent_information_gain = clamp(
        sum(weight * value for weight, value in zip(weights, meaningful))
        / max(1, sum(weights))
    )

    incumbent = math.inf
    best_so_far: list[float] = []
    for value in values:
        incumbent = min(incumbent, value)
        best_so_far.append(incumbent)
    recent_best = best_so_far[-min(len(best_so_far), size + 1):]
    net_improvement = max(
        0.0,
        (recent_best[0] - recent_best[-1]) / (abs(recent_best[0]) + 1e-12),
    )
    flat_slope = 1.0 - _smoothstep(
        net_improvement, threshold, max(0.05, 10.0 * threshold)
    )
    since_pressure = _smoothstep(
        float(trials_since), max(1.0, size / 3.0), float(size + 1)
    )
    history_maturity = _smoothstep(float(len(values)), 1.0, float(size + 1))
    global_search_stagnation = clamp(
        history_maturity
        * (
            0.55 * since_pressure
            + 0.25 * (1.0 - global_recent_information_gain)
            + 0.20 * flat_slope
        )
    )
    return global_recent_information_gain, global_search_stagnation, trials_since


def compute_search_exhaustion(
    *,
    incumbent_dominance: float,
    incumbent_stability: float,
    refinement_information_gain: float,
    effective_challenger_potential: float,
    family_residual_potential: float,
    analytical_evidence_urgency: float,
    unexplored_potential: float,
    global_recent_information_gain: float | None = None,
    global_search_stagnation: float | None = None,
    residual_search_value: float | None = None,
    incumbent_quality: float | None = None,
    evidence_supported_dominance: float | None = None,
    family_coverage: float | None = None,
    model_coverage: float | None = None,
    empirical_evidence_strength: float | None = None,
    global_model_residual_potential: float | None = None,
) -> float:
    """Continuous global evidence that another validation trial has little value."""
    dominance = clamp(
        incumbent_dominance
        if evidence_supported_dominance is None
        else evidence_supported_dominance
    )
    stability = clamp(incumbent_stability)
    local_information = clamp(refinement_information_gain)
    challenger = clamp(effective_challenger_potential)
    residual = clamp(
        family_residual_potential
        if residual_search_value is None
        else residual_search_value
    )
    urgency = clamp(analytical_evidence_urgency)
    global_gain = clamp(
        local_information
        if global_recent_information_gain is None
        else global_recent_information_gain
    )
    global_stagnation = clamp(
        stability
        if global_search_stagnation is None
        else global_search_stagnation
    )


    family_maturity = clamp(
        dominance if family_coverage is None else family_coverage
    )
    model_maturity = clamp(
        stability if model_coverage is None else model_coverage
    )
    empirical_maturity = clamp(
        max(dominance, stability)
        if empirical_evidence_strength is None
        else empirical_evidence_strength
    )
    model_residual = clamp(
        residual
        if global_model_residual_potential is None
        else global_model_residual_potential
    )
    maturity = clamp(
        0.18 * global_stagnation
        + 0.16 * (1.0 - global_gain)
        + 0.18 * family_maturity
        + 0.18 * model_maturity
        + 0.18 * empirical_maturity
        + 0.07 * stability
        + 0.05 * (1.0 - challenger)
    )

    blocker = max(challenger, residual, model_residual, urgency)
    return clamp(
        maturity
        * (1.0 - 0.80 * blocker)
    )


def _model_responsiveness(
    trials: Sequence[Mapping[str, Any]], threshold: float
) -> float:
    values = [_score(item) for item in trials]
    improvements, _ = _relative_improvements(values, threshold)
    events = improvements[1:]
    positives = [value for value in events if value >= threshold]
    best = max(positives, default=0.0)
    frequency = len(positives) / len(events) if events else 0.0
    last_index = max(
        (index for index, value in enumerate(events) if value >= threshold),
        default=-1,
    )
    recency = 1.0 / (1.0 + len(events) - 1 - last_index) if last_index >= 0 else 0.0
    return clamp(
        0.45 * _smoothstep(best, threshold, 0.10)
        + 0.25 * frequency
        + 0.20 * recency
        + 0.10 * _smoothstep(sum(positives), threshold, 0.15)
    )


def _gap_potential(relative_gap: float) -> float:
    """Overlapping, continuous near-best potential (1 at 0%, 0 at 20%)."""
    gap = max(0.0, float(relative_gap))
    if gap <= 0.03:
        return 1.0 - 0.10 * gap / 0.03
    if gap <= 0.08:
        return 0.90 - 0.40 * (gap - 0.03) / 0.05
    return clamp(0.50 * (0.20 - gap) / 0.12)


def _model_resolves_signal(
    item: Mapping[str, Any], signal: str, analytical: Mapping[str, Any]
) -> bool:
    """Use only capabilities already representable by catalog candidates."""
    model = str(item.get("model", ""))
    hp = item.get("hyperparameters", {})
    hp = hp if isinstance(hp, Mapping) else {}
    preprocessing = item.get("preprocessing_transformations", []) or []
    names = {
        str(step.get("name", step.get("transformation", ""))).lower()
        for step in preprocessing
        if isinstance(step, Mapping)
    }
    if signal == "seasonality":
        if model == "SARIMA":
            return True
        if model == "ETS" and hp.get("seasonal") not in (None, False, "none"):
            return bool(hp.get("seasonal_periods") or analytical.get("seasonal_periods"))
        return bool(
            model == "Prophet"
            and (hp.get("yearly_seasonality") is True or hp.get("custom_seasonal_period"))
        )
    if signal == "trend":
        order = hp.get("order", ())
        differenced_arima = (
            model in {"ARIMA", "SARIMA"}
            and isinstance(order, (list, tuple))
            and len(order) >= 2
            and int(order[1]) > 0
        )
        return model == "Prophet" or differenced_arima or "calendar_features" in names
    if signal == "autocorrelation":
        if model in {"ARIMA", "SARIMA"}:
            return True
        wanted = analytical.get("suggested_lag") or analytical.get("seasonal_periods")
        lags: set[int] = set()
        for step in preprocessing:
            if isinstance(step, Mapping) and str(step.get("name", "")).lower() == "lag_features":
                params = step.get("parameters", {})
                raw_lags = params.get("lags", []) if isinstance(params, Mapping) else []
                lags.update(int(value) for value in raw_lags if str(value).isdigit())
        return wanted is not None and int(wanted) in lags
    if signal == "nonlinearity":
        return MODEL_TO_FAMILY.get(model) in {"Tree-based", "Neural", "Transformer/Foundation"} or model in {"KNN", "SVR"}
    if signal == "scale_sensitivity":
        return bool(names & {"standard_scaler", "minmax_scaler", "log_transform", "boxcox_transform"})
    return False


def _analytical_resolution(
    analytical_output: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, Any]],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    source = analytical_output if isinstance(analytical_output, Mapping) else {}
    base = source.get("base_metrics", source)
    base = base if isinstance(base, Mapping) else {}

    def number(key: str, default: float = 0.0) -> float:
        try:
            value = float(source.get(key, base.get(key, default)))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    confidence_labels = {"low": 0.35, "medium": 0.65, "high": 0.95}
    season_conf = confidence_labels.get(
        str(source.get("seasonality_confidence", base.get("seasonality_confidence", "low"))).lower(),
        0.35,
    )
    autocorrelations = source.get("autocorrelations", base.get("autocorrelations", {}))
    acf_values = []
    if isinstance(autocorrelations, Mapping):
        for value in autocorrelations.values():
            try:
                finite = abs(float(value))
            except (TypeError, ValueError):
                continue
            if math.isfinite(finite):
                acf_values.append(finite)
    explicit_nonlinearity = number("nonlinearity_score", -1.0)
    strengths: dict[str, float] = {}
    if bool(source.get("seasonality_detected", base.get("seasonality_detected", False))):
        strengths["seasonality"] = clamp(
            0.55 * max(number("seasonality_acf"), max(acf_values, default=0.0))
            + 0.45 * season_conf
        )
    trend = abs(number("trend_strength"))
    if trend > 0.0:
        strengths["trend"] = _smoothstep(trend, 0.75, 3.0)
    if acf_values:
        strengths["autocorrelation"] = _smoothstep(max(acf_values), 0.35, 0.85)
    if explicit_nonlinearity >= 0.0:
        strengths["nonlinearity"] = clamp(explicit_nonlinearity)
    if bool(source.get("strong_right_skew", base.get("strong_right_skew", False)) or source.get("extreme_dynamic_range", base.get("extreme_dynamic_range", False))):
        strengths["scale_sensitivity"] = 0.8

    relevant = {name: strength for name, strength in strengths.items() if strength >= 0.60}
    unresolved: list[str] = []
    resolved: list[str] = []
    unresolved_strengths: list[float] = []
    for signal, strength in relevant.items():
        if any(_model_resolves_signal(item, signal, source) for item in history):
            resolved.append(signal)
        else:
            unresolved.append(signal)
            unresolved_strengths.append(strength)
    if not unresolved_strengths:
        score = 0.0
    else:
        score = clamp(
            0.75 * max(unresolved_strengths)
            + 0.25 * sum(unresolved_strengths) / len(unresolved_strengths)
        )
    return score, tuple(unresolved), tuple(resolved)


def compute_search_dynamics(
    history: Sequence[Mapping[str, Any]],
    available_models: Sequence[str],
    *,
    family_priorities: Mapping[str, float] | None = None,
    minimum_relative_improvement: float = 0.002,
    max_refinement_trials_per_model: int = 4,
    analytical_output: Mapping[str, Any] | None = None,
    maximum_valid_trainings: int = 30,
    global_information_window: int = 3,
) -> SearchDynamics:
    """Derive smooth [0, 1] FUZZY inputs from validation history.

    No signal changes candidate scores or the validation argmin. They describe
    only how mature, stable or saturated the current search trajectory is.
    """
    valid = [
        dict(item)
        for item in history
        if str(item.get("model", "")) in MODEL_TO_FAMILY
        and math.isfinite(_score(item))
    ]
    available_families = {
        MODEL_TO_FAMILY[str(model)]
        for model in available_models
        if str(model) in MODEL_TO_FAMILY
    }
    if not valid:
        return SearchDynamics(
            incumbent_stability=0.0,
            model_stagnation=0.0,
            global_recent_information_gain=1.0,
            global_search_stagnation=0.0,
            family_saturation=0.0,
            model_refinement_maturity=0.0,
            incumbent_quality=0.0,
            absolute_incumbent_quality=0.0,
            relative_incumbent_quality=0.0,
            family_coverage=0.0,
            distinct_model_coverage=0.0,
            cross_family_diversity=0.0,
            empirical_evidence_strength=0.0,
            semantic_empirical_misalignment=0.0,
            trials_since_last_improvement=0,
            current_family_attempts=0,
            current_model_refinements=0,
            recent_relative_improvement=0.0,
            challenger_model=None,
            challenger_mase=None,
            challenger_relative_gap=None,
            challenger_potential=0.0,
            challenger_refinement_maturity=0.0,
            refinement_responsiveness=0.0,
            remaining_refinement_capacity=1.0,
            unresolved_analytical_evidence=0.0,
            analytical_evidence_urgency=0.0,
            model_refinement_information_gain=1.0,
            refinement_information_gain=1.0,
            intra_family_model_coverage=0.0,
            model_coverage_by_family={},
            models_available_by_family={},
            models_evaluated_by_family={},
            models_unevaluated_by_family={},
            family_model_residual_potential={},
            global_model_residual_potential=(
                1.0 if available_families else 0.0
            ),
            strongest_model_residual_family=None,
            family_residual_potential=1.0 if available_families else 0.0,
            residual_search_value=1.0 if available_families else 0.0,
            preprocessing_refinement_potential=0.0,
            raw_challenger_potential=0.0,
            effective_challenger_potential=0.0,
            incumbent_dominance=0.0,
            dominance_confidence=0.0,
            evidence_supported_dominance=0.0,
            search_exhaustion=0.0,
            expected_search_value=1.0,
            explore_model_in_family_potential=0.0,
            late_breakthrough=0.0,
            remaining_budget=max(0, int(maximum_valid_trainings)),
            remaining_budget_ratio=1.0,
            models_evaluated_in_family=(),
            eligible_models_in_family=(),
            unresolved_analytical_signals=(),
            resolved_analytical_signals=(),
            current_model=None,
            current_family=None,
        )

    threshold = max(float(minimum_relative_improvement), 1e-12)
    scores = [_score(item) for item in valid]
    global_improvements, _ = _relative_improvements(scores, threshold)
    global_recent_information_gain, global_search_stagnation, trials_since = (
        _global_progress(scores, threshold, global_information_window)
    )
    incumbent_item = min(valid, key=_score)
    incumbent_score = _score(incumbent_item)
    incumbent_model = str(incumbent_item["model"])
    incumbent_family = MODEL_TO_FAMILY[incumbent_model]
    current_model = str(valid[-1]["model"])
    current_family = MODEL_TO_FAMILY[current_model]

    sorted_scores = sorted(scores)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else incumbent_score
    advantage = max(
        0.0, (second_best - incumbent_score) / (abs(second_best) + 1e-12)
    )
    stable_advantage = clamp(advantage / (10.0 * threshold))
    stability_time = clamp(trials_since / 3.0)
    incumbent_stability = clamp(
        stability_time * (0.75 + 0.25 * stable_advantage)
    )

    recent_global = global_improvements[-min(3, len(global_improvements)) :]
    recent_relative_improvement = (
        sum(recent_global) / len(recent_global) if recent_global else 0.0
    )

    model_trials = [item for item in valid if str(item["model"]) == current_model]
    model_scores = [_score(item) for item in model_trials]
    model_improvements, _ = _relative_improvements(model_scores, threshold)
    model_refinements = max(0, len(model_trials) - 1)
    recent_model_improvements = model_improvements[-min(2, len(model_improvements)) :]
    recent_model_improvement = (
        sum(recent_model_improvements) / len(recent_model_improvements)
        if recent_model_improvements
        else 0.0
    )
    lack_of_model_improvement = 1.0 - clamp(
        recent_model_improvement / threshold
    )
    refinement_volume = clamp(model_refinements / 2.0)

    consecutive_refinements = 0
    for index in range(len(valid) - 1, -1, -1):
        item = valid[index]
        if str(item["model"]) != current_model:
            break
        if any(
            str(previous["model"]) == current_model
            for previous in valid[:index]
        ):
            consecutive_refinements += 1
    consecutive_component = clamp(consecutive_refinements / 2.0)
    latest_model_gap = max(
        0.0,
        (model_scores[-1] - incumbent_score) / (abs(incumbent_score) + 1e-12),
    )
    deterioration = clamp(latest_model_gap / (10.0 * threshold))
    model_stagnation = clamp(
        refinement_volume
        * (
            0.55 * lack_of_model_improvement
            + 0.20 * deterioration
            + 0.25 * consecutive_component
        )
    )

    family_trials = [
        item
        for item in valid
        if MODEL_TO_FAMILY[str(item["model"])] == current_family
    ]
    family_scores = [_score(item) for item in family_trials]
    family_improvements, _ = _relative_improvements(family_scores, threshold)
    recent_family = family_improvements[-min(3, len(family_improvements)) :]
    recent_family_improvement = (
        sum(recent_family) / len(recent_family) if recent_family else 0.0
    )
    lack_of_family_improvement = 1.0 - clamp(
        recent_family_improvement / threshold
    )
    family_attempt_volume = clamp(max(0, len(family_trials) - 1) / 4.0)
    nonvaluable_family_trials = sum(
        improvement < threshold for improvement in family_improvements[1:]
    )
    nonvalue_ratio = (
        nonvaluable_family_trials / max(1, len(family_improvements) - 1)
    )
    family_saturation = clamp(
        family_attempt_volume
        * (
            0.50 * lack_of_family_improvement
            + 0.30 * nonvalue_ratio
            + 0.20 * model_stagnation
        )
    )

    distinct_configurations = len({_signature(item) for item in model_trials})
    refinement_target = max(2, int(max_refinement_trials_per_model) - 1)
    refinement_coverage = clamp(model_refinements / refinement_target)
    configuration_coverage = clamp(max(0, distinct_configurations - 1) / 2.0)
    model_refinement_maturity = clamp(
        0.55 * refinement_coverage
        + 0.25 * configuration_coverage
        + 0.20 * model_stagnation * refinement_coverage
    )

    median_score = float(sorted_scores[len(sorted_scores) // 2])
    quality_gap = max(
        0.0, (median_score - incumbent_score) / (abs(median_score) + 1e-12)
    )
    relative_incumbent_quality = (
        0.5 if len(sorted_scores) == 1 else clamp(quality_gap / 0.05)
    )

    absolute_incumbent_quality = clamp((1.35 - incumbent_score) / 0.70)
    incumbent_quality = clamp(
        0.70 * absolute_incumbent_quality
        + 0.30 * relative_incumbent_quality
    )

    responsiveness = _model_responsiveness(model_trials, threshold)
    information_gain = _information_gain(model_trials, threshold)
    remaining_capacity = clamp(
        (max_refinement_trials_per_model - model_refinements)
        / max(1, max_refinement_trials_per_model)
    )

    challenger_item: Mapping[str, Any] | None = None
    latest = valid[-1]
    latest_score = _score(latest)
    if str(latest["model"]) != incumbent_model and latest_score >= incumbent_score:
        challenger_item = latest
    challenger_model = str(challenger_item["model"]) if challenger_item else None
    challenger_mase = _score(challenger_item) if challenger_item else None
    challenger_gap = (
        max(0.0, (challenger_mase - incumbent_score) / (abs(incumbent_score) + 1e-12))
        if challenger_mase is not None
        else None
    )
    challenger_trials = (
        [item for item in valid if str(item["model"]) == challenger_model]
        if challenger_model
        else []
    )
    challenger_maturity = _refinement_maturity(
        challenger_trials, max_refinement_trials_per_model
    )
    challenger_capacity = clamp(
        (max_refinement_trials_per_model - max(0, len(challenger_trials) - 1))
        / max(1, max_refinement_trials_per_model)
    )
    challenger_family = MODEL_TO_FAMILY.get(challenger_model or "")
    family_attempts = (
        sum(MODEL_TO_FAMILY.get(str(item["model"])) == challenger_family for item in valid)
        if challenger_family
        else 0
    )
    family_immaturity = 1.0 - clamp(family_attempts / 4.0)
    raw_challenger_potential = (
        clamp(
            _gap_potential(challenger_gap or 0.0)
            * (
                0.55
                + 0.20 * (1.0 - challenger_maturity)
                + 0.15 * family_immaturity
                + 0.10 * challenger_capacity
            )
        )
        if challenger_item
        else 0.0
    )
    challenger_responsiveness = _model_responsiveness(challenger_trials, threshold)
    challenger_information_gain = _information_gain(challenger_trials, threshold)
    # A first evaluation remains promising; repeated unresponsive trials decay smoothly.
    responsiveness_component = max(
        challenger_responsiveness, 1.0 - challenger_maturity
    )
    effective_challenger_potential = (
        clamp(
            raw_challenger_potential
            * (0.35 + 0.65 * responsiveness_component)
            * (1.0 - 0.75 * challenger_maturity)
            * (0.25 + 0.75 * challenger_information_gain)
        )
        if challenger_item
        else 0.0
    )
    unresolved_evidence, unresolved_signals, resolved_signals = _analytical_resolution(
        analytical_output, valid
    )
    analytical_evidence_urgency = clamp(
        unresolved_evidence * (1.0 - 0.55 * incumbent_quality)
    )

    seen_families = {
        MODEL_TO_FAMILY[str(item["model"])] for item in valid
    }
    family_coverage = (
        clamp(len(seen_families) / len(available_families))
        if available_families
        else 0.0
    )

    evaluated_models = {str(item["model"]) for item in valid}
    eligible_current_models = tuple(
        str(model) for model in available_models
        if MODEL_TO_FAMILY.get(str(model)) == current_family
    )
    evaluated_current_models = tuple(
        model for model in eligible_current_models if model in evaluated_models
    )
    intra_family_model_coverage = (
        len(evaluated_current_models) / len(eligible_current_models)
        if eligible_current_models else 0.0
    )
    priority_map = dict(family_priorities or {})
    global_model_coverage = (
        len(evaluated_models) / len(available_models)
        if available_models else 0.0
    )
    family_counts = [
        sum(MODEL_TO_FAMILY[str(item["model"])] == family for item in valid)
        for family in available_families
    ]
    if len(seen_families) > 1:
        probabilities = [count / len(valid) for count in family_counts if count]
        cross_family_diversity = clamp(
            -sum(value * math.log(value) for value in probabilities)
            / math.log(len(available_families))
        )
    else:
        cross_family_diversity = 0.0
    absolute_evaluation_maturity = _smoothstep(float(len(valid)), 3.0, 10.0)
    settled_evidence = 1.0 - clamp(recent_relative_improvement / 0.05)
    empirical_evidence_strength = clamp(
        0.30 * absolute_evaluation_maturity
        + 0.25 * family_coverage
        + 0.15 * global_model_coverage
        + 0.15 * cross_family_diversity
        + 0.10 * incumbent_stability
        + 0.05 * settled_evidence
    )
    global_search_maturity = clamp(
        0.50 * family_coverage + 0.50 * global_model_coverage
    )
    models_available_by_family: dict[str, tuple[str, ...]] = {}
    models_evaluated_by_family: dict[str, tuple[str, ...]] = {}
    models_unevaluated_by_family: dict[str, tuple[str, ...]] = {}
    model_coverage_by_family: dict[str, float] = {}
    family_model_residual_potential: dict[str, float] = {}
    for family in available_families:
        eligible = tuple(
            str(model) for model in available_models
            if MODEL_TO_FAMILY.get(str(model)) == family
        )
        evaluated = tuple(model for model in eligible if model in evaluated_models)
        unevaluated = tuple(model for model in eligible if model not in evaluated_models)
        model_coverage = len(evaluated) / len(eligible) if eligible else 1.0
        models_available_by_family[family] = eligible
        models_evaluated_by_family[family] = evaluated
        models_unevaluated_by_family[family] = unevaluated
        model_coverage_by_family[family] = model_coverage
        family_observations = [
            item for item in valid
            if MODEL_TO_FAMILY[str(item["model"])] == family
        ]
        if family_observations:
            family_best = min(_score(item) for item in family_observations)
            relative_gap = max(
                0.0,
                (family_best - incumbent_score) / (abs(incumbent_score) + 1e-12),
            )
            empirical_relevance = _gap_potential(relative_gap)
            _, family_stagnation, _ = _global_progress(
                [_score(item) for item in family_observations],
                threshold,
                global_information_window,
            )
        else:
            empirical_relevance = clamp(
                0.70
                * (
                    1.0
                    - 0.70
                    * global_search_maturity
                    * incumbent_quality
                )
                + 0.20 * analytical_evidence_urgency
            )
            family_stagnation = 0.0
        family_relevance = clamp(float(priority_map.get(family, 0.5)))
        family_model_residual_potential[family] = clamp(
            (1.0 - model_coverage)
            * family_relevance
            * (
                0.45
                + 0.40 * empirical_relevance
                + 0.15 * (1.0 - family_stagnation)
            )
        )
    global_model_residual_potential = max(
        family_model_residual_potential.values(), default=0.0
    )
    evaluated_residuals = {
        family: residual
        for family, residual in family_model_residual_potential.items()
        if family in seen_families
    }
    strongest_model_residual_family = (
        max(evaluated_residuals, key=evaluated_residuals.get)
        if evaluated_residuals
        else None
    )
    family_residual_potential = max(
        global_model_residual_potential,
        max(
            (
                clamp(float(priority_map.get(family, 0.5)))
                for family in available_families - seen_families
            ),
            default=0.0,
        ),
    )

    model_preprocessing_signatures = {
        _preprocessing_signature(item) for item in model_trials
    }
    observed_nonraw_signatures = {
        _preprocessing_signature(item) for item in valid
        if (item.get("preprocessing_transformations", []) or [])
    }
    source = analytical_output if isinstance(analytical_output, Mapping) else {}
    recommended_opportunities = sum(
        (
            bool(source.get("strong_right_skew") or source.get("extreme_dynamic_range")),
            bool(source.get("suggested_lag") or source.get("seasonal_periods")),
        )
    )
    plausible_signature_count = max(
        1, 1 + len(observed_nonraw_signatures) + recommended_opportunities
    )
    preprocessing_coverage = clamp(
        len(model_preprocessing_signatures) / plausible_signature_count
    )
    recommendation_strength = clamp(recommended_opportunities / 2.0)
    preprocessing_refinement_potential = clamp(
        (1.0 - preprocessing_coverage)
        * (0.65 + 0.35 * recommendation_strength)
    )

    alternative_best: dict[str, float] = {}
    alternative_families: set[str] = set()
    for item in valid:
        model = str(item["model"])
        if model == incumbent_model:
            continue
        alternative_best[model] = min(alternative_best.get(model, math.inf), _score(item))
        alternative_families.add(MODEL_TO_FAMILY[model])
    dominance_gaps = sorted(
        max(0.0, (value - incumbent_score) / (abs(incumbent_score) + 1e-12))
        for value in alternative_best.values()
    )
    if dominance_gaps:
        median_gap = dominance_gaps[len(dominance_gaps) // 2]
        min_gap = dominance_gaps[0]
        family_comparison_coverage = clamp(
            len(alternative_families) / max(1, min(3, len(available_families)))
        )
        model_comparison_coverage = clamp(
            len(alternative_best) / max(1, min(4, len(available_models) - 1))
        )
        incumbent_dominance = clamp(
            0.35 * _smoothstep(median_gap, 0.05, 0.50)
            + 0.25 * _smoothstep(min_gap, 0.03, 0.30)
            + 0.15 * family_comparison_coverage
            + 0.15 * model_comparison_coverage
            + 0.10 * incumbent_stability
        )
    else:
        incumbent_dominance = 0.0


    dominance_confidence = clamp(
        0.10
        + 0.25 * family_coverage
        + 0.20 * global_model_coverage
        + 0.20 * cross_family_diversity
        + 0.25 * empirical_evidence_strength
    )
    evidence_supported_dominance = clamp(
        incumbent_dominance * dominance_confidence
    )

    maximum = max(1, int(maximum_valid_trainings))
    remaining_budget = max(0, maximum - len(valid))
    remaining_budget_ratio = clamp(remaining_budget / maximum)
    last_improvement = global_improvements[-1] if global_improvements else 0.0
    late_breakthrough = clamp(
        _smoothstep(last_improvement, max(threshold, 0.01), 0.20)
        * (
            0.50
            + 0.25 * (1.0 - remaining_budget_ratio)
            + 0.20 * responsiveness
            + 0.05 * (1.0 - model_refinement_maturity)
        )
    )


    analytical_evidence_urgency = clamp(
        analytical_evidence_urgency
        * (1.0 - 0.45 * incumbent_quality * evidence_supported_dominance)
    )
    unseen_family_potential = max(
        (
            float(priority_map.get(family, 0.5))
            for family in available_families - seen_families
        ),
        default=0.0,
    )
    semantic_family_relevance = clamp(unseen_family_potential)
    empirical_opportunity = max(
        effective_challenger_potential,
        1.0 - evidence_supported_dominance,
    )
    raw_residual_search_value = clamp(
        0.22 * family_residual_potential
        + 0.28 * global_model_residual_potential
        + 0.14 * semantic_family_relevance
        + 0.12 * analytical_evidence_urgency
        + 0.10 * empirical_opportunity
        + 0.07 * global_recent_information_gain
        + 0.07 * (1.0 - incumbent_quality)
    )
    exhausted_incumbent_pressure = clamp(
        incumbent_quality
        * evidence_supported_dominance
        * global_search_stagnation
        * (1.0 - global_recent_information_gain)
    )
    residual_search_value = clamp(
        raw_residual_search_value * (1.0 - 0.75 * exhausted_incumbent_pressure)
    )
    weak_incumbent_guard = clamp(
        (1.0 - incumbent_quality)
        * max(
            family_residual_potential,
            global_model_residual_potential,
            semantic_family_relevance,
            analytical_evidence_urgency,
        )
    )
    residual_search_value = max(
        residual_search_value,
        clamp(0.85 * weak_incumbent_guard),
    )
    unexplored_search_potential = clamp(
        max(
            family_residual_potential,
            global_model_residual_potential,
            unseen_family_potential,
        )
    )
    residual_family = strongest_model_residual_family or current_family
    residual_family_coverage = model_coverage_by_family.get(
        residual_family, intra_family_model_coverage
    )
    residual_family_potential = family_model_residual_potential.get(
        residual_family, 0.0
    )
    explore_model_in_family_potential = clamp(
        (1.0 - residual_family_coverage)
        * residual_family_potential
        * (0.45 + 0.35 * model_stagnation + 0.20 * (1.0 - information_gain))
        * (0.50 + 0.50 * clamp(residual_search_value / 0.50))
    )
    expected_search_value = clamp(
        max(
            effective_challenger_potential,
            residual_search_value,
            analytical_evidence_urgency,
            responsiveness * information_gain,
            explore_model_in_family_potential,
        )
        * (0.75 + 0.25 * remaining_budget_ratio)
    )
    # A late responsive breakthrough protects the final slot even when the
    # general search space is otherwise mature.
    expected_search_value = max(
        expected_search_value,
        clamp(late_breakthrough * responsiveness),
    )
    search_exhaustion = compute_search_exhaustion(
        incumbent_dominance=incumbent_dominance,
        evidence_supported_dominance=evidence_supported_dominance,
        incumbent_stability=incumbent_stability,
        refinement_information_gain=information_gain,
        effective_challenger_potential=effective_challenger_potential,
        family_residual_potential=family_residual_potential,
        analytical_evidence_urgency=analytical_evidence_urgency,
        unexplored_potential=unexplored_search_potential,
        global_recent_information_gain=global_recent_information_gain,
        global_search_stagnation=global_search_stagnation,
        residual_search_value=residual_search_value,
        incumbent_quality=incumbent_quality,
        family_coverage=family_coverage,
        model_coverage=global_model_coverage,
        empirical_evidence_strength=empirical_evidence_strength,
        global_model_residual_potential=global_model_residual_potential,
    )

    semantic_misalignment = 0.0
    priorities = dict(family_priorities or {})
    if priorities:
        top_family = max(priorities, key=lambda family: float(priorities[family]))
        top_trials = [
            item
            for item in valid
            if MODEL_TO_FAMILY[str(item["model"])] == top_family
        ]
        if top_trials:
            top_best = min(_score(item) for item in top_trials)
            top_gap = max(
                0.0,
                (top_best - incumbent_score) / (abs(incumbent_score) + 1e-12),
            )
            top_attempt_volume = clamp((len(top_trials) - 1) / 3.0)
            semantic_misalignment = clamp(
                top_attempt_volume
                * (
                    0.60 * clamp(top_gap / (10.0 * threshold))
                    + 0.40
                    * (family_saturation if top_family == incumbent_family else 1.0)
                )
            )

    return SearchDynamics(
        incumbent_stability=incumbent_stability,
        model_stagnation=model_stagnation,
        global_recent_information_gain=global_recent_information_gain,
        global_search_stagnation=global_search_stagnation,
        family_saturation=family_saturation,
        model_refinement_maturity=model_refinement_maturity,
        incumbent_quality=incumbent_quality,
        absolute_incumbent_quality=absolute_incumbent_quality,
        relative_incumbent_quality=relative_incumbent_quality,
        family_coverage=family_coverage,
        distinct_model_coverage=global_model_coverage,
        cross_family_diversity=cross_family_diversity,
        empirical_evidence_strength=empirical_evidence_strength,
        semantic_empirical_misalignment=semantic_misalignment,
        trials_since_last_improvement=trials_since,
        current_family_attempts=len(family_trials),
        current_model_refinements=model_refinements,
        recent_relative_improvement=recent_relative_improvement,
        challenger_model=challenger_model,
        challenger_mase=challenger_mase,
        challenger_relative_gap=challenger_gap,
        challenger_potential=effective_challenger_potential,
        challenger_refinement_maturity=challenger_maturity,
        refinement_responsiveness=responsiveness,
        remaining_refinement_capacity=remaining_capacity,
        unresolved_analytical_evidence=unresolved_evidence,
        analytical_evidence_urgency=analytical_evidence_urgency,
        model_refinement_information_gain=information_gain,
        refinement_information_gain=information_gain,
        intra_family_model_coverage=intra_family_model_coverage,
        model_coverage_by_family=model_coverage_by_family,
        models_available_by_family=models_available_by_family,
        models_evaluated_by_family=models_evaluated_by_family,
        models_unevaluated_by_family=models_unevaluated_by_family,
        family_model_residual_potential=family_model_residual_potential,
        global_model_residual_potential=global_model_residual_potential,
        strongest_model_residual_family=strongest_model_residual_family,
        family_residual_potential=family_residual_potential,
        residual_search_value=residual_search_value,
        preprocessing_refinement_potential=preprocessing_refinement_potential,
        raw_challenger_potential=raw_challenger_potential,
        effective_challenger_potential=effective_challenger_potential,
        incumbent_dominance=incumbent_dominance,
        dominance_confidence=dominance_confidence,
        evidence_supported_dominance=evidence_supported_dominance,
        search_exhaustion=search_exhaustion,
        expected_search_value=expected_search_value,
        explore_model_in_family_potential=explore_model_in_family_potential,
        late_breakthrough=late_breakthrough,
        remaining_budget=remaining_budget,
        remaining_budget_ratio=remaining_budget_ratio,
        models_evaluated_in_family=evaluated_current_models,
        eligible_models_in_family=eligible_current_models,
        unresolved_analytical_signals=unresolved_signals,
        resolved_analytical_signals=resolved_signals,
        current_model=current_model,
        current_family=current_family,
    )


__all__ = [
    "SearchDynamics",
    "compute_search_dynamics",
    "compute_search_exhaustion",
]
