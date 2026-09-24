"""
Simple strategic Planner for TFUSE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from soft_computing.search_controller import SearchControl, build_search_control
from soft_computing.search_dynamics import compute_search_dynamics
from utils.model_family_state import CANONICAL_FAMILY_ORDER, MODEL_TO_FAMILY
from utils.configuration_identity import log_exploration_status, target_transform_label
from utils.stop_policy import stop_eligibility


class PlannerAction(str, Enum):
    EXPLORE = "explore"
    REFINE = "refine"
    RETRY = "retry"
    STOP = "stop"


ALLOWED_ACTIONS = tuple(item.value for item in PlannerAction)
FAMILY_COST = {
    "Statistical": 0.30,
    "Additive": 0.35,
    "Tree-based": 0.45,
    "Classical ML": 0.25,
    "Neural": 0.75,
    "Transformer/Foundation": 1.00,
}


def clip01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, min(1.0, number))


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def target_family_coverage(search_breadth: Any, available_count: int) -> int:
    """Map controller breadth to a deterministic number of families."""
    count = max(0, int(available_count))
    if count == 0:
        return 0
    breadth = clip01(search_breadth, 1.0)
    if breadth < 0.30:
        target = 1
    elif breadth < 0.60:
        target = 2
    elif breadth < 0.80:
        target = 4
    else:
        target = count
    return min(count, target)


@dataclass(frozen=True)
class SearchDecisionState:
    mode: str
    family_priorities: Mapping[str, float]
    preferred_families: tuple[str, ...]
    semantic_trust: float
    analytical_trust: float
    fusion_reliability: float
    conflict: float
    semantic_discriminability: float
    empirical_strength: float
    guidance_weight: float
    search_breadth: float
    explore_strength: float
    refine_strength: float
    stop_strength: float
    plateau_strength: float
    refinement_completeness: float
    unexplored_potential: float
    target_family_coverage: int
    target_families: tuple[str, ...]
    families_seen: tuple[str, ...]
    action_degrees: Mapping[str, float]
    incumbent_stability: float = 0.0
    model_stagnation: float = 0.0
    global_recent_information_gain: float = 1.0
    global_search_stagnation: float = 0.0
    family_saturation: float = 0.0
    model_refinement_maturity: float = 0.0
    incumbent_quality: float = 0.0
    absolute_incumbent_quality: float = 0.0
    relative_incumbent_quality: float = 0.0
    budget_coverage: float = 0.0
    family_coverage: float = 0.0
    distinct_model_coverage: float = 0.0
    cross_family_diversity: float = 0.0
    semantic_empirical_misalignment: float = 0.0
    trials_since_last_improvement: int = 0
    current_family_attempts: int = 0
    current_model_refinements: int = 0
    recent_relative_improvement: float = 0.0
    challenger_model: str | None = None
    challenger_mase: float | None = None
    challenger_relative_gap: float | None = None
    challenger_potential: float = 0.0
    challenger_refinement_maturity: float = 0.0
    refinement_responsiveness: float = 0.0
    remaining_refinement_capacity: float = 0.0
    unresolved_analytical_evidence: float = 0.0
    analytical_evidence_urgency: float = 0.0
    model_refinement_information_gain: float = 1.0
    refinement_information_gain: float = 1.0
    intra_family_model_coverage: float = 0.0
    family_residual_potential: float = 0.0
    model_coverage_by_family: Mapping[str, float] = field(default_factory=dict)
    models_available_by_family: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    models_evaluated_by_family: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    models_unevaluated_by_family: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    family_model_residual_potential: Mapping[str, float] = field(default_factory=dict)
    global_model_residual_potential: float = 0.0
    strongest_model_residual_family: str | None = None
    residual_search_value: float = 0.0
    preprocessing_refinement_potential: float = 0.0
    raw_challenger_potential: float = 0.0
    effective_challenger_potential: float = 0.0
    incumbent_dominance: float = 0.0
    dominance_confidence: float = 0.0
    evidence_supported_dominance: float = 0.0
    search_exhaustion: float = 0.0
    expected_search_value: float = 1.0
    explore_new_family_strength: float = 0.0
    explore_model_in_family_strength: float = 0.0
    family_escape_preferred: bool = False
    selected_escape_family: str | None = None
    family_escape_reason: str | None = None
    late_breakthrough: float = 0.0
    remaining_budget: int = 0
    remaining_budget_ratio: float = 1.0
    models_evaluated_in_family: tuple[str, ...] = ()
    eligible_models_in_family: tuple[str, ...] = ()
    dominant_rules: Mapping[str, str] = field(default_factory=dict)
    fuzzy_inference_trace: Mapping[str, Any] = field(default_factory=dict)
    unresolved_analytical_signals: tuple[str, ...] = ()
    resolved_analytical_signals: tuple[str, ...] = ()
    current_model: str | None = None
    current_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["discriminability"] = self.semantic_discriminability
        value["empirical_evidence_strength"] = self.empirical_strength
        value["preferred_families"] = list(self.preferred_families)
        value["target_families"] = list(self.target_families)
        value["families_seen"] = list(self.families_seen)
        value["unresolved_analytical_signals"] = list(
            self.unresolved_analytical_signals
        )
        value["resolved_analytical_signals"] = list(self.resolved_analytical_signals)
        value["models_evaluated_in_family"] = list(self.models_evaluated_in_family)
        value["eligible_models_in_family"] = list(self.eligible_models_in_family)
        value["models_available_by_family"] = {
            family: list(models)
            for family, models in self.models_available_by_family.items()
        }
        value["models_evaluated_by_family"] = {
            family: list(models)
            for family, models in self.models_evaluated_by_family.items()
        }
        value["models_unevaluated_by_family"] = {
            family: list(models)
            for family, models in self.models_unevaluated_by_family.items()
        }
        return value

    @property
    def discriminability(self) -> float:
        """Compatibility alias for semantic_discriminability (D_t)."""
        return self.semantic_discriminability


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    selected_family: str | None
    selected_model: str | None
    configuration_strategy: str | None
    controller_support: Mapping[str, Any]
    empirical_support: Mapping[str, Any]
    supporting_evidence: Mapping[str, Any]
    decision_confidence: float
    decision_score: float | None
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        controller_support = dict(value.get("controller_support", {}))
        controller_support.pop("fuzzy_inference_trace", None)
        value["controller_support"] = controller_support
        return value


@dataclass(frozen=True)
class _CandidateScore:
    model: str
    family: str
    score: float
    empirical_score: float
    exploration_bonus: float
    guidance_bonus: float
    cost_penalty: float
    failure_penalty: float
    trials: int


@dataclass
class PlannerPolicy:
    """Convert evidence uncertainty into an auditable search state."""

    empirical_full_strength_trials: int = 10
    stopping_patience: int = 5
    minimum_relative_improvement: float = 0.002
    exploration_bonus: float = 0.12
    guidance_exploration_bonus: float = 0.10
    cost_penalty: float = 0.06
    failure_penalty: float = 0.16
    max_refinement_trials_per_model: int = 4
    global_information_window: int = 3
    capture_full_trace: bool = False

    @staticmethod
    def history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for item in state.get("performance_history", []) or []:
            if not isinstance(item, Mapping) or item.get("error"):
                continue
            model = str(item.get("model", ""))
            mase = finite_float(item.get("mase"))
            if model in MODEL_TO_FAMILY and mase is not None:
                history.append({**dict(item), "model": model, "mase": mase})
        return history

    @staticmethod
    def available_models(state: Mapping[str, Any]) -> list[str]:
        catalog = state.get("model_catalog", {})
        if not isinstance(catalog, Mapping):
            return []
        excluded = set(catalog.get("unavailable_models", []) or [])
        excluded.update(catalog.get("disabled_models", []) or [])
        excluded.update(catalog.get("failed_models", []) or [])
        excluded.update(state.get("technical_exhausted_models", []) or [])
        return [
            str(model)
            for model in catalog.get("available_models", []) or []
            if str(model) in MODEL_TO_FAMILY and str(model) not in excluded
        ]

    @staticmethod
    def remaining_budget(state: Mapping[str, Any], valid_count: int) -> int:
        maximum = int(state.get("maximum_valid_trainings", 0) or 0)
        maximum = maximum or int(state.get("max_iterations", 0) or 0)
        return max(0, maximum - valid_count)

    @staticmethod
    def recent_improvements(
        history: Sequence[Mapping[str, Any]], window: int = 5
    ) -> list[float]:
        values: list[float] = []
        incumbent = math.inf
        for item in history:
            score = float(item["mase"])
            if math.isfinite(incumbent) and score < incumbent:
                values.append((incumbent - score) / max(abs(incumbent), 1e-9))
            else:
                values.append(0.0)
            incumbent = min(incumbent, score)
        return values[-max(1, int(window)) :]

    def empirical_strength(
        self,
        history: Sequence[Mapping[str, Any]],
        models: Sequence[str],
        *,
        maximum_valid_trainings: int | None = None,
        incumbent_stability: float = 0.0,
        recent_relative_improvement: float = 0.0,
    ) -> float:

        target = max(6, int(self.empirical_full_strength_trials))
        position = clip01((len(history) - 2.0) / max(1.0, target + 2.0))
        absolute_maturity = position * position * (3.0 - 2.0 * position)
        possible = {MODEL_TO_FAMILY[m] for m in models}
        tested = {MODEL_TO_FAMILY[str(item["model"])] for item in history}
        family_coverage = len(tested) / len(possible) if possible else 0.0
        model_coverage = (
            len({str(item["model"]) for item in history}) / len(models)
            if models
            else 0.0
        )
        counts = [
            sum(MODEL_TO_FAMILY[str(item["model"])] == family for item in history)
            for family in possible
        ]
        probabilities = (
            [count / len(history) for count in counts if count] if history else []
        )
        diversity = (
            -sum(value * math.log(value) for value in probabilities)
            / math.log(len(possible))
            if len(possible) > 1 and len(probabilities) > 1
            else 0.0
        )
        settled = 1.0 - clip01(recent_relative_improvement / 0.05)
        evidence_quality = 0.65 * clip01(incumbent_stability) + 0.35 * settled
        return clip01(
            0.35 * absolute_maturity
            + 0.25 * family_coverage
            + 0.15 * model_coverage
            + 0.15 * diversity
            + 0.10 * evidence_quality
        )

    @staticmethod
    def _evidence(state: Mapping[str, Any]) -> Mapping[str, Any]:
        guidance = state.get("planner_guidance", {})
        if isinstance(guidance, Mapping) and guidance:
            return guidance
        diagnostics = state.get("diagnostics", {})
        if isinstance(diagnostics, Mapping):
            value = diagnostics.get("evidence_fusion", {})
            if isinstance(value, Mapping):
                return value
        return {}

    def evaluate(self, state: Mapping[str, Any]) -> SearchDecisionState:
        mode = str(state.get("influence_mode", "fuzzy")).lower()
        if mode != "fuzzy":
            raise ValueError("T-FUSE Planner accepts only fuzzy influence")

        models = self.available_models(state)
        history = self.history(state)
        evidence = self._evidence(state)

        priorities = {family: 0.5 for family in CANONICAL_FAMILY_ORDER}
        raw_priorities = evidence.get("family_compatibility", {})
        if isinstance(raw_priorities, Mapping):
            priorities.update(
                {
                    family: clip01(raw_priorities.get(family, 0.5), 0.5)
                    for family in CANONICAL_FAMILY_ORDER
                }
            )

        available_families = [
            family
            for family in CANONICAL_FAMILY_ORDER
            if any(MODEL_TO_FAMILY[model] == family for model in models)
        ]
        ordered = sorted(
            available_families,
            key=lambda family: (
                -priorities[family],
                CANONICAL_FAMILY_ORDER.index(family),
            ),
        )
        counts = {family: 0 for family in CANONICAL_FAMILY_ORDER}
        seen_order: list[str] = []
        for item in history:
            family = MODEL_TO_FAMILY[str(item["model"])]
            counts[family] += 1
            if family not in seen_order:
                seen_order.append(family)
        coverage = (
            sum(counts[f] > 0 for f in available_families) / len(available_families)
            if available_families
            else 0.0
        )
        stop_inputs = stop_eligibility(
            state,
            {
                "stopping_patience": self.stopping_patience,
                "minimum_relative_improvement": (self.minimum_relative_improvement),
                "max_refinement_trials_per_model": (
                    self.max_refinement_trials_per_model
                ),
            },
        )
        unseen_families = [
            family for family in available_families if counts[family] == 0
        ]
        maximum_valid_trainings = int(
            state.get("maximum_valid_trainings", 0)
            or state.get("max_iterations", 0)
            or self.empirical_full_strength_trials
        )
        dynamics = compute_search_dynamics(
            history,
            models,
            family_priorities=priorities,
            minimum_relative_improvement=self.minimum_relative_improvement,
            max_refinement_trials_per_model=(self.max_refinement_trials_per_model),
            analytical_output=state.get("data_summary", {}),
            maximum_valid_trainings=maximum_valid_trainings,
            global_information_window=self.global_information_window,
        )
        budget_coverage = clip01(len(history) / max(1, maximum_valid_trainings))
        empirical = self.empirical_strength(
            history,
            models,
            maximum_valid_trainings=maximum_valid_trainings,
            incumbent_stability=dynamics.incumbent_stability,
            recent_relative_improvement=dynamics.recent_relative_improvement,
        )

        empirical = dynamics.empirical_evidence_strength
        if unseen_families:
            best_unseen_compatibility = max(
                priorities[family] for family in unseen_families
            )
            uncertainty = clip01(evidence.get("conflict", 0.0))
            unexplored_potential = clip01(
                best_unseen_compatibility
                * (0.70 + 0.15 * uncertainty + 0.15 * (1.0 - empirical))
            )
        else:
            unexplored_potential = 0.0

        unexplored_potential = clip01(
            max(unexplored_potential, dynamics.family_residual_potential)
        )
        control: SearchControl = build_search_control(
            mode,
            evidence,
            empirical,
            plateau_strength=float(stop_inputs["plateau_strength"]),
            refinement_completeness=float(stop_inputs["refinement_completeness"]),
            unexplored_potential=unexplored_potential,
            incumbent_stability=dynamics.incumbent_stability,
            model_stagnation=dynamics.model_stagnation,
            global_recent_information_gain=(dynamics.global_recent_information_gain),
            global_search_stagnation=dynamics.global_search_stagnation,
            family_saturation=dynamics.family_saturation,
            model_refinement_maturity=dynamics.model_refinement_maturity,
            incumbent_quality=dynamics.incumbent_quality,
            absolute_incumbent_quality=dynamics.absolute_incumbent_quality,
            relative_incumbent_quality=dynamics.relative_incumbent_quality,
            budget_coverage=budget_coverage,
            family_coverage=dynamics.family_coverage,
            distinct_model_coverage=dynamics.distinct_model_coverage,
            cross_family_diversity=dynamics.cross_family_diversity,
            semantic_empirical_misalignment=dynamics.semantic_empirical_misalignment,
            recent_relative_improvement=dynamics.recent_relative_improvement,
            challenger_potential=dynamics.challenger_potential,
            challenger_refinement_maturity=dynamics.challenger_refinement_maturity,
            refinement_responsiveness=dynamics.refinement_responsiveness,
            remaining_refinement_capacity=dynamics.remaining_refinement_capacity,
            unresolved_analytical_evidence=dynamics.unresolved_analytical_evidence,
            analytical_evidence_urgency=dynamics.analytical_evidence_urgency,
            refinement_information_gain=dynamics.refinement_information_gain,
            model_refinement_information_gain=(
                dynamics.model_refinement_information_gain
            ),
            intra_family_model_coverage=dynamics.intra_family_model_coverage,
            family_residual_potential=dynamics.family_residual_potential,
            global_model_residual_potential=(dynamics.global_model_residual_potential),
            residual_search_value=dynamics.residual_search_value,
            preprocessing_refinement_potential=dynamics.preprocessing_refinement_potential,
            raw_challenger_potential=dynamics.raw_challenger_potential,
            effective_challenger_potential=dynamics.effective_challenger_potential,
            incumbent_dominance=dynamics.incumbent_dominance,
            dominance_confidence=dynamics.dominance_confidence,
            evidence_supported_dominance=(dynamics.evidence_supported_dominance),
            search_exhaustion=dynamics.search_exhaustion,
            expected_search_value=dynamics.expected_search_value,
            explore_model_in_family_potential=(
                dynamics.explore_model_in_family_potential
            ),
            late_breakthrough=dynamics.late_breakthrough,
            remaining_budget_ratio=dynamics.remaining_budget_ratio,
            capture_full_trace=self.capture_full_trace,
        )
        target_count = target_family_coverage(control.search_breadth, len(ordered))
        target_families = tuple(ordered[:target_count])
        preferred = target_families
        selected_escape_family = next(
            (family for family in ordered if family in unseen_families),
            None,
        )
        family_escape_preferred = bool(
            selected_escape_family
            and control.explore_new_family_strength
            >= max(
                control.explore_model_in_family_strength,
                control.refine_strength,
            )
            and dynamics.global_search_stagnation
            >= dynamics.global_recent_information_gain
        )
        family_escape_reason = (
            "stagnating_region_with_guided_unseen_family"
            if family_escape_preferred
            else None
        )
        explore = control.explore_strength
        refine = control.refine_strength

        retry_requested = bool(
            state.get("technical_error", False)
            and not state.get("non_retryable_candidate_failure", False)
        )
        retry_allowed = retry_requested and int(
            state.get("technical_retry_count", 0)
        ) < int(state.get("max_technical_retries", 0))
        retry = 0.98 if retry_allowed else 0.01
        if retry_allowed:
            explore *= 0.25
            refine *= 0.25

        stop = control.stop_strength

        return SearchDecisionState(
            mode=mode,
            family_priorities=priorities,
            preferred_families=preferred,
            semantic_trust=clip01(evidence.get("semantic_trust", 0.0)),
            analytical_trust=clip01(evidence.get("analytical_trust", 0.0)),
            fusion_reliability=clip01(evidence.get("fusion_reliability", 0.0)),
            conflict=clip01(evidence.get("conflict", 0.0)),
            semantic_discriminability=clip01(
                evidence.get(
                    "semantic_discriminability",
                    evidence.get("discriminability", 0.0),
                )
            ),
            empirical_strength=empirical,
            guidance_weight=control.guidance_weight,
            search_breadth=control.search_breadth,
            explore_strength=control.explore_strength,
            refine_strength=control.refine_strength,
            stop_strength=control.stop_strength,
            plateau_strength=control.plateau_strength,
            refinement_completeness=control.refinement_completeness,
            unexplored_potential=control.unexplored_potential,
            incumbent_stability=control.incumbent_stability,
            model_stagnation=control.model_stagnation,
            global_recent_information_gain=(control.global_recent_information_gain),
            global_search_stagnation=control.global_search_stagnation,
            family_saturation=control.family_saturation,
            model_refinement_maturity=control.model_refinement_maturity,
            incumbent_quality=control.incumbent_quality,
            absolute_incumbent_quality=control.absolute_incumbent_quality,
            relative_incumbent_quality=control.relative_incumbent_quality,
            budget_coverage=control.budget_coverage,
            family_coverage=control.family_coverage,
            distinct_model_coverage=control.distinct_model_coverage,
            cross_family_diversity=control.cross_family_diversity,
            semantic_empirical_misalignment=control.semantic_empirical_misalignment,
            trials_since_last_improvement=dynamics.trials_since_last_improvement,
            current_family_attempts=dynamics.current_family_attempts,
            current_model_refinements=dynamics.current_model_refinements,
            recent_relative_improvement=dynamics.recent_relative_improvement,
            challenger_model=dynamics.challenger_model,
            challenger_mase=dynamics.challenger_mase,
            challenger_relative_gap=dynamics.challenger_relative_gap,
            challenger_potential=control.challenger_potential,
            challenger_refinement_maturity=control.challenger_refinement_maturity,
            refinement_responsiveness=control.refinement_responsiveness,
            remaining_refinement_capacity=control.remaining_refinement_capacity,
            unresolved_analytical_evidence=control.unresolved_analytical_evidence,
            analytical_evidence_urgency=control.analytical_evidence_urgency,
            model_refinement_information_gain=(
                control.model_refinement_information_gain
            ),
            refinement_information_gain=control.refinement_information_gain,
            intra_family_model_coverage=control.intra_family_model_coverage,
            family_residual_potential=control.family_residual_potential,
            model_coverage_by_family=dynamics.model_coverage_by_family,
            models_available_by_family=dynamics.models_available_by_family,
            models_evaluated_by_family=dynamics.models_evaluated_by_family,
            models_unevaluated_by_family=dynamics.models_unevaluated_by_family,
            family_model_residual_potential=(dynamics.family_model_residual_potential),
            global_model_residual_potential=(control.global_model_residual_potential),
            strongest_model_residual_family=(dynamics.strongest_model_residual_family),
            residual_search_value=control.residual_search_value,
            preprocessing_refinement_potential=control.preprocessing_refinement_potential,
            raw_challenger_potential=control.raw_challenger_potential,
            effective_challenger_potential=control.effective_challenger_potential,
            incumbent_dominance=control.incumbent_dominance,
            dominance_confidence=control.dominance_confidence,
            evidence_supported_dominance=(control.evidence_supported_dominance),
            search_exhaustion=control.search_exhaustion,
            expected_search_value=control.expected_search_value,
            explore_new_family_strength=control.explore_new_family_strength,
            explore_model_in_family_strength=(control.explore_model_in_family_strength),
            family_escape_preferred=family_escape_preferred,
            selected_escape_family=selected_escape_family,
            family_escape_reason=family_escape_reason,
            late_breakthrough=control.late_breakthrough,
            remaining_budget=dynamics.remaining_budget,
            remaining_budget_ratio=control.remaining_budget_ratio,
            models_evaluated_in_family=dynamics.models_evaluated_in_family,
            eligible_models_in_family=dynamics.eligible_models_in_family,
            dominant_rules=control.dominant_rules,
            fuzzy_inference_trace=control.fuzzy_inference_trace,
            unresolved_analytical_signals=dynamics.unresolved_analytical_signals,
            resolved_analytical_signals=dynamics.resolved_analytical_signals,
            current_model=dynamics.current_model,
            current_family=dynamics.current_family,
            target_family_coverage=target_count,
            target_families=target_families,
            families_seen=tuple(seen_order),
            action_degrees={
                "explore": explore,
                "refine": refine,
                "retry": retry,
                "stop": clip01(stop),
            },
        )


@dataclass
class PlannerAgent:
    """Strategic search Planner with a small, explicit decision surface."""

    max_iterations: int = 30
    policy_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = set(PlannerPolicy.__dataclass_fields__)
        self.policy = PlannerPolicy(
            **{
                key: value
                for key, value in dict(self.policy_config or {}).items()
                if key in fields
            }
        )

    @staticmethod
    def build_input(state: Mapping[str, Any]) -> dict[str, Any]:
        history = PlannerPolicy.history(state)
        best = min(history, key=lambda item: float(item["mase"]), default=None)
        return {
            "evaluated_candidates": history,
            "current_best_candidate": dict(best) if best else None,
            "current_best_mase": float(best["mase"]) if best else None,
            "remaining_budget": PlannerPolicy.remaining_budget(state, len(history)),
            "recent_improvements": PlannerPolicy.recent_improvements(history),
        }

    @staticmethod
    def _failures(state: Mapping[str, Any], model: str) -> int:
        return sum(
            str(item.get("model", "")) == model
            and item.get("error_type")
            in {
                "execution_failure",
                "invalid_configuration",
                "technical_generation_failure",
            }
            for item in state.get("candidate_ledger", []) or []
            if isinstance(item, Mapping)
        )

    def _rank(
        self,
        state: Mapping[str, Any],
        control: SearchDecisionState,
        rejected_models: set[str],
    ) -> list[_CandidateScore]:
        models = self.policy.available_models(state)
        history = self.policy.history(state)
        attempts = {model: [] for model in models}
        for item in history:
            model = str(item["model"])
            if model in attempts:
                attempts[model].append(float(item["mase"]))
        best_by_model = {
            model: min(values, default=math.inf) for model, values in attempts.items()
        }
        observed = [value for value in best_by_model.values() if math.isfinite(value)]
        low, high = min(observed, default=0.0), max(observed, default=0.0)

        priorities = control.family_priorities
        priority_values = [priorities[MODEL_TO_FAMILY[m]] for m in models] or [0.5]
        mean_priority = sum(priority_values) / len(priority_values)
        result: list[_CandidateScore] = []

        for model in models:
            if model in rejected_models:
                continue
            family = MODEL_TO_FAMILY[model]
            trials = len(attempts[model])
            empirical = (
                0.5
                if not math.isfinite(best_by_model[model]) or high <= low
                else 1.0 - (best_by_model[model] - low) / (high - low)
            )
            exploration = float(self.policy.exploration_bonus) / math.sqrt(1.0 + trials)
            guidance = 0.0
            if trials == 0:
                centered = max(
                    -1.0, min(1.0, 2.0 * (priorities[family] - mean_priority))
                )


                guidance = (
                    max(0.0, centered)
                    * control.guidance_weight
                    * float(self.policy.guidance_exploration_bonus)
                )
            cost = float(self.policy.cost_penalty) * FAMILY_COST[family]
            failures = self._failures(state, model)
            failure = (
                float(self.policy.failure_penalty)
                * failures
                / max(1, failures + trials)
            )
            score = clip01(empirical, 0.5) + exploration + guidance - cost - failure
            result.append(
                _CandidateScore(
                    model,
                    family,
                    score,
                    clip01(empirical, 0.5),
                    exploration,
                    guidance,
                    cost,
                    failure,
                    trials,
                )
            )

        order = {model: index for index, model in enumerate(models)}
        return sorted(result, key=lambda item: (-item.score, order[item.model]))

    def plan(
        self,
        state: Mapping[str, Any],
        *,
        safety_rejections: Sequence[Mapping[str, Any]] = (),
    ) -> PlannerDecision:
        planner_input = self.build_input(state)
        control = self.policy.evaluate(state)
        rejected = {
            str(item.get("model")) for item in safety_rejections if item.get("model")
        }
        ranked = self._rank(state, control, rejected)

        stop_status = stop_eligibility(
            state,
            {
                "stopping_patience": self.policy.stopping_patience,
                "minimum_relative_improvement": self.policy.minimum_relative_improvement,
                "max_refinement_trials_per_model": (
                    self.policy.max_refinement_trials_per_model
                ),
            },
        )
        explore_score = float(control.action_degrees.get("explore", 0.0))
        refine_score = float(control.action_degrees.get("refine", 0.0))
        retry_score = float(control.action_degrees.get("retry", 0.0))
        raw_fuzzy_stop_strength = float(control.stop_strength)
        stop_score = raw_fuzzy_stop_strength if stop_status["stop_eligible"] else 0.0
        stop_blocked = bool(
            not stop_status["stop_eligible"] and not stop_status["hard_stop_required"]
        )
        stop_block_reason = (
            stop_status["reasons_blocking_stop"][0]
            if stop_blocked and stop_status["reasons_blocking_stop"]
            else None
        )
        action_scores = {
            "explore": explore_score,
            "refine": refine_score,
            "retry": retry_score,
            "stop": stop_score,
        }
        tie_priority = {"retry": 4, "refine": 3, "explore": 2, "stop": 1}
        if stop_status["hard_stop_required"]:
            action = "stop"
        elif retry_score > 0.01:
            action = "retry"
        else:
            action = max(
                ("explore", "refine", "stop"),
                key=lambda value: (action_scores[value], tie_priority[value]),
            )
        remaining = int(planner_input["remaining_budget"])
        log_status = log_exploration_status(state)
        raw_request_pending = bool(
            log_status["raw_exploration_pending"] and remaining > 0 and ranked
        )
        current_best = planner_input.get("current_best_candidate")
        current_best_model = (
            str(current_best.get("model"))
            if isinstance(current_best, Mapping)
            else None
        )
        current_best_transform = (
            target_transform_label(
                current_best.get("preprocessing_transformations", [])
            )
            if isinstance(current_best, Mapping)
            else None
        )
        best_transform_status = log_status["evaluated_preprocessing_by_model"].get(
            current_best_model,
            {"raw_evaluated": False, "log_evaluated": False},
        )
        counterpart_transform = None
        if (
            current_best_model
            and log_status["log_transform_recommended"]
            and log_status["log_transform_supported"]
        ):
            if (
                current_best_transform == "raw"
                and not best_transform_status["log_evaluated"]
            ):
                counterpart_transform = "log1p"
            elif (
                current_best_transform in {"log1p", "log"}
                and not best_transform_status["raw_evaluated"]
            ):
                counterpart_transform = "raw"
        required_target_transform = (
            "raw"
            if raw_request_pending
            else counterpart_transform
            or (
                "log1p"
                if log_status["log_exploration_pending"] and remaining > 0 and ranked
                else None
            )
        )
        required_refinement_model = (
            current_best_model
            if counterpart_transform
            else None
        )
        targeted_transform_pending = bool(
            required_target_transform
            and remaining > 0
            and ranked
            and not stop_status["hard_stop_required"]
        )
        model_refinement_pending = bool(
            stop_status["required_model_refinement_pending"]
            and remaining > 0
            and ranked
            and not targeted_transform_pending
            and not stop_status["hard_stop_required"]
        )
        recent = planner_input["recent_improvements"]
        if not ranked:
            action = "stop"
        if targeted_transform_pending and action != "retry":
            target_model = (
                current_best_model
                if counterpart_transform
                else None
            )
            requested_transform_choice = next(
                (item for item in ranked if item.model == target_model),
                ranked[0],
            )
            action = "refine" if requested_transform_choice.trials > 0 else "explore"
        elif model_refinement_pending and action != "retry":
            requested_transform_choice = next(
                (item for item in ranked if item.model == current_best_model),
                ranked[0],
            )
            action = "refine"

        if action == "refine" and ranked:
            current_best = planner_input.get("current_best_candidate")
            current_best_model = (
                str(current_best.get("model"))
                if isinstance(current_best, Mapping)
                else None
            )
            refinement_target_model = (
                control.challenger_model
                if control.challenger_model
                and control.challenger_potential >= 0.60
                and control.challenger_refinement_maturity < 0.80
                else current_best_model
            )
            refinable_best = [
                item
                for item in ranked
                if item.model == refinement_target_model
                and item.trials < int(self.policy.max_refinement_trials_per_model)
            ]
            if (
                not refinable_best
                and not targeted_transform_pending
                and not model_refinement_pending
            ):
                action_scores["refine"] = 0.0
                available_actions = (
                    ("retry",) if retry_score > 0.01 else ("explore", "stop")
                )
                action = max(
                    available_actions,
                    key=lambda value: (action_scores[value], tie_priority[value]),
                )

        family_escape_triggered = bool(
            action == "explore"
            and control.family_escape_preferred
            and control.selected_escape_family
            and not targeted_transform_pending
            and not model_refinement_pending
        )

        if action == "stop":
            exhausted_without_target = bool(
                "target_mase_reached" in state
                and not state.get("target_mase_reached", False)
                and control.search_exhaustion >= 0.70
            )
            reason = (
                "budget_exhausted"
                if stop_status["hard_stop_required"]
                else (
                    "search_space_exhausted"
                    if not ranked
                    else (
                        "fuzzy_search_exhausted_no_target"
                        if exhausted_without_target
                        else "fuzzy_controller_converged"
                    )
                )
            )
            return PlannerDecision(
                action="stop",
                selected_family=None,
                selected_model=None,
                configuration_strategy=None,
                controller_support={
                    **control.to_dict(),
                    "action_scores": action_scores,
                },
                empirical_support={
                    "current_best_mase": planner_input.get("current_best_mase"),
                    "recent_improvement": max(recent, default=None),
                },
                supporting_evidence={
                    "remaining_budget": remaining,
                    "eligible_candidates": len(ranked),
                    "recent_improvements": list(recent),
                    **stop_status,
                    "explore_score": explore_score,
                    "refine_score": refine_score,
                    "stop_score": stop_score,
                    "selected_action": "stop",
                    "final_action": "stop",
                    "family_escape_triggered": False,
                    "current_family": control.current_family,
                    "selected_escape_family": None,
                    "family_escape_reason": None,
                    "raw_fuzzy_stop_strength": raw_fuzzy_stop_strength,
                    "effective_stop_strength": stop_score,
                    "stop_blocked": stop_blocked,
                    "stop_block_reason": stop_block_reason,
                    "fuzzy_outputs": {
                        "explore_strength": control.explore_strength,
                        "refine_strength": control.refine_strength,
                        "stop_strength": control.stop_strength,
                    },
                    "dominant_fuzzy_rules": dict(control.dominant_rules),
                    "fuzzy_inputs": {
                        "plateau_strength": control.plateau_strength,
                        "refinement_completeness": (control.refinement_completeness),
                        "unexplored_potential": control.unexplored_potential,
                        "empirical_evidence_strength": (control.empirical_strength),
                        "budget_coverage": control.budget_coverage,
                        "evidence_conflict": control.conflict,
                        "incumbent_stability": control.incumbent_stability,
                        "model_stagnation": control.model_stagnation,
                        "global_recent_information_gain": control.global_recent_information_gain,
                        "global_search_stagnation": control.global_search_stagnation,
                        "family_saturation": control.family_saturation,
                        "model_refinement_maturity": control.model_refinement_maturity,
                        "incumbent_quality": control.incumbent_quality,
                        "absolute_incumbent_quality": control.absolute_incumbent_quality,
                        "relative_incumbent_quality": control.relative_incumbent_quality,
                        "family_coverage": control.family_coverage,
                        "semantic_empirical_misalignment": control.semantic_empirical_misalignment,
                        "trials_since_last_improvement": control.trials_since_last_improvement,
                        "current_family_attempts": control.current_family_attempts,
                        "current_model_refinements": control.current_model_refinements,
                        "recent_relative_improvement": control.recent_relative_improvement,
                        "challenger_model": control.challenger_model,
                        "challenger_mase": control.challenger_mase,
                        "challenger_relative_gap": control.challenger_relative_gap,
                        "challenger_potential": control.challenger_potential,
                        "challenger_refinement_maturity": control.challenger_refinement_maturity,
                        "refinement_responsiveness": control.refinement_responsiveness,
                        "remaining_refinement_capacity": control.remaining_refinement_capacity,
                        "unresolved_analytical_evidence": control.unresolved_analytical_evidence,
                        "analytical_evidence_urgency": control.analytical_evidence_urgency,
                        "model_refinement_information_gain": control.model_refinement_information_gain,
                        "refinement_information_gain": control.refinement_information_gain,
                        "intra_family_model_coverage": control.intra_family_model_coverage,
                        "model_coverage_by_family": dict(
                            control.model_coverage_by_family
                        ),
                        "family_model_residual_potential": dict(
                            control.family_model_residual_potential
                        ),
                        "global_model_residual_potential": control.global_model_residual_potential,
                        "family_residual_potential": control.family_residual_potential,
                        "residual_search_value": control.residual_search_value,
                        "preprocessing_refinement_potential": control.preprocessing_refinement_potential,
                        "raw_challenger_potential": control.raw_challenger_potential,
                        "effective_challenger_potential": control.effective_challenger_potential,
                        "incumbent_dominance": control.incumbent_dominance,
                        "search_exhaustion": control.search_exhaustion,
                        "expected_search_value": control.expected_search_value,
                        "explore_new_family_strength": control.explore_new_family_strength,
                        "explore_model_in_family_strength": control.explore_model_in_family_strength,
                        "late_breakthrough": control.late_breakthrough,
                        "remaining_budget": control.remaining_budget,
                        "remaining_budget_ratio": control.remaining_budget_ratio,
                        "current_family": control.current_family,
                        "models_evaluated_in_family": list(
                            control.models_evaluated_in_family
                        ),
                        "eligible_models_in_family": list(
                            control.eligible_models_in_family
                        ),
                        "models_available_by_family": {
                            family: list(models)
                            for family, models in control.models_available_by_family.items()
                        },
                        "models_evaluated_by_family": {
                            family: list(models)
                            for family, models in control.models_evaluated_by_family.items()
                        },
                        "models_unevaluated_by_family": {
                            family: list(models)
                            for family, models in control.models_unevaluated_by_family.items()
                        },
                        "unresolved_analytical_signals": list(
                            control.unresolved_analytical_signals
                        ),
                        "resolved_analytical_signals": list(
                            control.resolved_analytical_signals
                        ),
                    },
                    "hard_constraints": {
                        "minimum_valid_trainings_reached": (
                            stop_status["minimum_budget_reached"]
                        ),
                        "maximum_budget_reached": (stop_status["hard_stop_required"]),
                        "mandatory_refinement_pending": (
                            stop_status["mandatory_refinement_pending"]
                        ),
                    },
                    "effective_action_scores": {
                        "EXPLORE": action_scores["explore"],
                        "REFINE": action_scores["refine"],
                        "STOP": action_scores["stop"],
                    },
                    "stop_reason": reason,
                    "target_mase_reached": bool(
                        state.get("target_mase_reached", False)
                    ),
                },
                decision_confidence=(
                    1.0 if stop_status["hard_stop_required"] else clip01(stop_score)
                ),
                decision_score=None,
                reason_code=reason,
            )

        selection_reason = "empirical_search_score_argmax"
        if action == "explore":
            residual_family = (
                control.strongest_model_residual_family or control.current_family
            )
            in_family_candidates = [
                item
                for item in ranked
                if item.family == residual_family and item.trials == 0
            ]
            unseen_target = [
                family
                for family in control.target_families
                if family not in control.families_seen
            ]
            unseen_family_candidates = [
                item
                for item in ranked
                if item.family in unseen_target and item.trials == 0
            ]
            escape_candidates = [
                item
                for item in unseen_family_candidates
                if item.family == control.selected_escape_family
            ]
            if family_escape_triggered and escape_candidates:
                eligible = escape_candidates
                selection_reason = "controlled_family_escape"
            elif (
                in_family_candidates
                and control.explore_model_in_family_strength > control.refine_strength
            ):
                eligible = in_family_candidates
                selection_reason = "explore_model_in_promising_family"
            elif unseen_family_candidates:
                eligible = unseen_family_candidates
                selection_reason = "explore_unseen_family_for_breadth"
            else:
                eligible = [item for item in ranked if item.trials == 0] or ranked
                selection_reason = (
                    "explore_after_refinement_cap"
                    if planner_input.get("current_best_candidate")
                    else "explore_ranked_candidate"
                )
        elif action == "refine":
            current_best = planner_input.get("current_best_candidate")
            current_best_model = (
                str(current_best.get("model"))
                if isinstance(current_best, Mapping)
                else None
            )
            refinement_target_model = (
                control.challenger_model
                if control.challenger_model
                and control.challenger_potential >= 0.60
                and control.challenger_refinement_maturity < 0.80
                else current_best_model
            )
            eligible = (
                [
                    item
                    for item in ranked
                    if item.model == refinement_target_model
                    and item.trials < int(self.policy.max_refinement_trials_per_model)
                ]
                or [item for item in ranked if item.trials > 0]
                or ranked
            )
            selection_reason = (
                "refine_near_best_challenger"
                if refinement_target_model == control.challenger_model
                else "refine_empirically_observed_candidate"
            )
        else:
            previous = state.get("candidate_request", {})
            previous_model = (
                previous.get("model") if isinstance(previous, Mapping) else None
            )
            eligible = [
                item for item in ranked if item.model == previous_model
            ] or ranked
            selection_reason = "retry_previous_candidate"

        if targeted_transform_pending and action != "retry":
            choice = requested_transform_choice
            if raw_request_pending:
                selection_reason = "required_raw_candidate_exploration"
            elif counterpart_transform == "log1p":
                selection_reason = "current_best_log_refinement"
            elif counterpart_transform == "raw":
                selection_reason = "current_best_raw_refinement"
            else:
                selection_reason = "required_log_candidate_exploration"
        elif model_refinement_pending and action != "retry":
            choice = requested_transform_choice
            selection_reason = "current_best_model_refinement"
        else:
            choice = eligible[0]
        return PlannerDecision(
            action=action,
            selected_family=choice.family,
            selected_model=choice.model,
            configuration_strategy={
                "explore": "default_then_local_refinement",
                "refine": "local_refinement",
                "retry": "recovery_configuration",
            }[action],
            controller_support={
                **control.to_dict(),
                "action_scores": action_scores,
                "selected_family_priority": control.family_priorities.get(
                    choice.family, 0.5
                ),
                "guidance_bonus": choice.guidance_bonus,
            },
            empirical_support={
                "empirical_score": choice.empirical_score,
                "model_trials": choice.trials,
                "exploration_bonus": choice.exploration_bonus,
                "cost_penalty": choice.cost_penalty,
                "failure_penalty": choice.failure_penalty,
                "current_best_mase": planner_input.get("current_best_mase"),
            },
            supporting_evidence={
                "preferred_families": list(control.preferred_families),
                "target_family_coverage": control.target_family_coverage,
                "target_families": list(control.target_families),
                "families_seen_before": list(control.families_seen),
                "selection_reason": selection_reason,
                "remaining_budget": remaining,
                **log_status,
                "required_target_transform": required_target_transform,
                "required_refinement_model": required_refinement_model,
                **stop_status,
                "explore_score": explore_score,
                "refine_score": refine_score,
                "stop_score": stop_score,
                "selected_action": action,
                "final_action": action,
                "family_escape_triggered": family_escape_triggered,
                "current_family": control.current_family,
                "selected_escape_family": (
                    control.selected_escape_family if family_escape_triggered else None
                ),
                "family_escape_reason": (
                    control.family_escape_reason if family_escape_triggered else None
                ),
                "raw_fuzzy_stop_strength": raw_fuzzy_stop_strength,
                "effective_stop_strength": stop_score,
                "stop_blocked": stop_blocked,
                "stop_block_reason": stop_block_reason,
                "fuzzy_outputs": {
                    "explore_strength": control.explore_strength,
                    "refine_strength": control.refine_strength,
                    "stop_strength": control.stop_strength,
                },
                "dominant_fuzzy_rules": dict(control.dominant_rules),
                "fuzzy_inputs": {
                    "plateau_strength": control.plateau_strength,
                    "refinement_completeness": (control.refinement_completeness),
                    "unexplored_potential": control.unexplored_potential,
                    "empirical_evidence_strength": control.empirical_strength,
                    "budget_coverage": control.budget_coverage,
                    "evidence_conflict": control.conflict,
                    "incumbent_stability": control.incumbent_stability,
                    "model_stagnation": control.model_stagnation,
                    "global_recent_information_gain": control.global_recent_information_gain,
                    "global_search_stagnation": control.global_search_stagnation,
                    "family_saturation": control.family_saturation,
                    "model_refinement_maturity": control.model_refinement_maturity,
                    "incumbent_quality": control.incumbent_quality,
                    "absolute_incumbent_quality": control.absolute_incumbent_quality,
                    "relative_incumbent_quality": control.relative_incumbent_quality,
                    "family_coverage": control.family_coverage,
                    "distinct_model_coverage": control.distinct_model_coverage,
                    "cross_family_diversity": control.cross_family_diversity,
                    "semantic_empirical_misalignment": control.semantic_empirical_misalignment,
                    "trials_since_last_improvement": control.trials_since_last_improvement,
                    "current_family_attempts": control.current_family_attempts,
                    "current_model_refinements": control.current_model_refinements,
                    "recent_relative_improvement": control.recent_relative_improvement,
                    "challenger_model": control.challenger_model,
                    "challenger_mase": control.challenger_mase,
                    "challenger_relative_gap": control.challenger_relative_gap,
                    "challenger_potential": control.challenger_potential,
                    "challenger_refinement_maturity": control.challenger_refinement_maturity,
                    "refinement_responsiveness": control.refinement_responsiveness,
                    "remaining_refinement_capacity": control.remaining_refinement_capacity,
                    "unresolved_analytical_evidence": control.unresolved_analytical_evidence,
                    "analytical_evidence_urgency": control.analytical_evidence_urgency,
                    "model_refinement_information_gain": control.model_refinement_information_gain,
                    "refinement_information_gain": control.refinement_information_gain,
                    "intra_family_model_coverage": control.intra_family_model_coverage,
                    "model_coverage_by_family": dict(control.model_coverage_by_family),
                    "family_model_residual_potential": dict(
                        control.family_model_residual_potential
                    ),
                    "global_model_residual_potential": control.global_model_residual_potential,
                    "family_residual_potential": control.family_residual_potential,
                    "residual_search_value": control.residual_search_value,
                    "preprocessing_refinement_potential": control.preprocessing_refinement_potential,
                    "raw_challenger_potential": control.raw_challenger_potential,
                    "effective_challenger_potential": control.effective_challenger_potential,
                    "incumbent_dominance": control.incumbent_dominance,
                    "dominance_confidence": control.dominance_confidence,
                    "evidence_supported_dominance": control.evidence_supported_dominance,
                    "search_exhaustion": control.search_exhaustion,
                    "expected_search_value": control.expected_search_value,
                    "explore_new_family_strength": control.explore_new_family_strength,
                    "explore_model_in_family_strength": control.explore_model_in_family_strength,
                    "late_breakthrough": control.late_breakthrough,
                    "remaining_budget": control.remaining_budget,
                    "remaining_budget_ratio": control.remaining_budget_ratio,
                    "current_family": control.current_family,
                    "models_evaluated_in_family": list(
                        control.models_evaluated_in_family
                    ),
                    "eligible_models_in_family": list(
                        control.eligible_models_in_family
                    ),
                    "models_available_by_family": {
                        family: list(models)
                        for family, models in control.models_available_by_family.items()
                    },
                    "models_evaluated_by_family": {
                        family: list(models)
                        for family, models in control.models_evaluated_by_family.items()
                    },
                    "models_unevaluated_by_family": {
                        family: list(models)
                        for family, models in control.models_unevaluated_by_family.items()
                    },
                    "unresolved_analytical_signals": list(
                        control.unresolved_analytical_signals
                    ),
                    "resolved_analytical_signals": list(
                        control.resolved_analytical_signals
                    ),
                },
                "hard_constraints": {
                    "minimum_valid_trainings_reached": (
                        stop_status["minimum_budget_reached"]
                    ),
                    "maximum_budget_reached": (stop_status["hard_stop_required"]),
                    "mandatory_refinement_pending": (
                        stop_status["mandatory_refinement_pending"]
                    ),
                },
                "effective_action_scores": {
                    "EXPLORE": action_scores["explore"],
                    "REFINE": action_scores["refine"],
                    "STOP": action_scores["stop"],
                },
                "stop_reason": None,
                "exploration_scope": (
                    "new_model_in_family"
                    if selection_reason == "explore_model_in_promising_family"
                    else None
                ),
                "family_model_coverage_before": (
                    control.model_coverage_by_family.get(choice.family)
                    if selection_reason == "explore_model_in_promising_family"
                    else None
                ),
                "family_model_residual_before": (
                    control.family_model_residual_potential.get(choice.family)
                    if selection_reason == "explore_model_in_promising_family"
                    else None
                ),
            },
            decision_confidence=clip01(
                0.55 * control.action_degrees[action] + 0.45 * clip01(choice.score)
            ),
            decision_score=choice.score,
            reason_code=selection_reason,
        )

    def replanning_exhausted(
        self,
        state: Mapping[str, Any],
        rejections: Sequence[Mapping[str, Any]],
    ) -> PlannerDecision:
        control = self.policy.evaluate(state)
        return PlannerDecision(
            action="stop",
            selected_family=None,
            selected_model=None,
            configuration_strategy=None,
            controller_support=control.to_dict(),
            empirical_support={
                "current_best_mase": self.build_input(state).get("current_best_mase")
            },
            supporting_evidence={
                "replanning_attempts": len(rejections),
                "guard_reasons": [str(item.get("reason")) for item in rejections],
                "remaining_budget": self.build_input(state)["remaining_budget"],
            },
            decision_confidence=1.0,
            decision_score=None,
            reason_code="replanning_limit_exhausted",
        )

    def run(self, state: Mapping[str, Any], step: int = 0) -> dict[str, Any]:
        if not state.get("data_summary"):
            return {
                "next_action": "analytical",
                "reason": "analytical_prerequisite",
                "confidence": 1.0,
            }
        if state.get("semantic_enabled") and not state.get("semantic_completed"):
            return {
                "next_action": "semantic",
                "reason": "semantic_prerequisite",
                "confidence": 1.0,
            }
        return self.plan(state).to_dict()


# Small compatibility alias for external imports; new code should use SearchDecisionState.
FuzzyDecisionState = SearchDecisionState
FuzzyPlannerPolicy = PlannerPolicy
ALLOWED_DEVIATIONS: tuple[str, ...] = ()


__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_DEVIATIONS",
    "FuzzyDecisionState",
    "FuzzyPlannerPolicy",
    "PlannerAction",
    "PlannerAgent",
    "PlannerDecision",
    "PlannerPolicy",
    "SearchDecisionState",
    "clip01",
    "finite_float",
    "target_family_coverage",
]
