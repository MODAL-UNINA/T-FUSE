"""Fuzzy search controller for Planner search behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import inspect
import textwrap
from functools import lru_cache
from typing import Any, Mapping

from .fuzzy_membership import clamp, high_input, low_input, medium_input


@dataclass(frozen=True)
class SearchControl:
    mode: str
    guidance_weight: float
    search_breadth: float
    explore_strength: float
    refine_strength: float
    stop_strength: float
    plateau_strength: float
    refinement_completeness: float
    unexplored_potential: float
    reliability: float
    conflict: float
    semantic_discriminability: float
    empirical_strength: float
    incumbent_stability: float = 0.0
    model_stagnation: float = 0.0
    global_recent_information_gain: float = 1.0
    global_search_stagnation: float = 0.0
    family_saturation: float = 0.0
    model_refinement_maturity: float = 1.0
    incumbent_quality: float = 0.0
    absolute_incumbent_quality: float = 0.0
    relative_incumbent_quality: float = 0.0
    family_coverage: float = 0.0
    distinct_model_coverage: float = 0.0
    cross_family_diversity: float = 0.0
    semantic_empirical_misalignment: float = 0.0
    recent_relative_improvement: float = 0.0
    budget_coverage: float = 0.0
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
    global_model_residual_potential: float = 0.0
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
    late_breakthrough: float = 0.0
    remaining_budget_ratio: float = 1.0
    dominant_rules: Mapping[str, str] = field(default_factory=dict)
    fuzzy_inference_trace: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["discriminability"] = self.semantic_discriminability
        return value

    @property
    def discriminability(self) -> float:
        """Compatibility alias for semantic_discriminability (D_t)."""
        return self.semantic_discriminability


def _weighted(
    rules: list[tuple[float, float]],
    default: float,
    group: str | None = None,
    trace_rules: dict[str, list[tuple[float, float]]] | None = None,
) -> float:
    if group is not None and trace_rules is not None:
        trace_rules[group] = rules
    denominator = sum(max(0.0, strength) for strength, _ in rules)
    if denominator <= 1e-12:
        return clamp(default)
    return clamp(
        sum(max(0.0, strength) * value for strength, value in rules)
        / denominator
    )


def _rule_activations(
    reliability: float,
    conflict: float,
    semantic_discriminability: float,
    trace_memberships: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float, float]:
    """Return uncertain, moderate and confident rule activations."""
    r_l, r_m, r_h = (
        low_input(reliability),
        medium_input(reliability),
        high_input(reliability),
    )
    c_l, c_m, c_h = (
        low_input(conflict),
        medium_input(conflict),
        high_input(conflict),
    )
    d_l, d_m, d_h = (
        low_input(semantic_discriminability),
        medium_input(semantic_discriminability),
        high_input(semantic_discriminability),
    )
    uncertain = max(r_l, c_h, d_l)
    moderate = max(
        min(r_m, max(c_l, c_m), max(d_m, d_h)),
        min(r_h, d_m),
        min(r_h, c_m, d_h),
    )
    confident = min(r_h, c_l, d_h)
    if trace_memberships is not None:
        trace_memberships.update({
            "reliability": {"LOW": r_l, "MEDIUM": r_m, "HIGH": r_h},
            "conflict": {"LOW": c_l, "MEDIUM": c_m, "HIGH": c_h},
            "semantic_discriminability": {"LOW": d_l, "MEDIUM": d_m, "HIGH": d_h},
        })
    return uncertain, moderate, confident


@lru_cache(maxsize=1)
def _controller_rule_expressions() -> dict[str, list[tuple[str, int]]]:
    """Read exact antecedent expressions from this controller definition."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fuzzy_control)))
    result: dict[str, list[tuple[str, int]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "guidance", "breadth", "explore", "refine", "stop"
        }:
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not call.args:
            continue
        rule_list = call.args[0]
        if not isinstance(rule_list, ast.List):
            continue
        expressions = []
        for item in rule_list.elts:
            activation = item.elts[0]
            names = {
                child.id for child in ast.walk(activation)
                if isinstance(child, ast.Name) and child.id not in {"min", "max", "bool"}
            }
            expressions.append((ast.unparse(activation), len(names)))
        result[target.id] = expressions
    return result
def _membership_vector(low: float, medium: float, high: float) -> dict[str, float]:
    """Package membership values already used by inference (never recompute)."""
    return {"LOW": low, "MEDIUM": medium, "HIGH": high}


def _build_inference_trace(
    *,
    inputs: Mapping[str, float | bool],
    memberships: Mapping[str, Mapping[str, float]],
    rule_groups: Mapping[str, list[tuple[float, float]]],
    raw_action_strengths: Mapping[str, float],
) -> dict[str, Any]:
    """Build a JSON-safe observation from the exact inference primitives."""
    registry: list[dict[str, Any]] = []
    activations: dict[str, float] = {}
    contributions: dict[str, dict[str, dict[str, float]]] = {}
    expressions = _controller_rule_expressions()
    for group, rules in rule_groups.items():
        consequent_name = group.upper()
        contributions[consequent_name] = {}
        for index, (activation, consequent_value) in enumerate(rules, start=1):
            rule_id = f"{group.lower()}.rule_{index:03d}"
            alpha = float(activation)
            weight = max(0.0, alpha)
            expression, antecedent_count = expressions[group.lower()][index - 1]
            activations[rule_id] = alpha
            registry.append({
                "rule_id": rule_id,
                "group": consequent_name,
                "antecedents": [{
                    "representation": "exact_python_expression",
                    "expression": expression,
                }],
                "consequent": [consequent_name, float(consequent_value)],
                "consequent_value": float(consequent_value),
                "length": antecedent_count,
                "controller_source": "soft_computing.search_controller:fuzzy_control",
            })
            contributions[consequent_name][rule_id] = {
                "activation": alpha,
                "consequent_value": float(consequent_value),
                "aggregation_weight": weight,
                "weighted_numerator_term": weight * float(consequent_value),
            }
    return {
        "inputs": {
            name: {
                "value": value,
                "normalized_domain": [0.0, 1.0],
                "is_direct_rule_input": True,
            }
            for name, value in inputs.items()
        },
        "memberships": {name: dict(vector) for name, vector in memberships.items()},
        "rule_activations": activations,
        "rule_contributions": contributions,
        "raw_action_strengths": dict(raw_action_strengths),
        "aggregation": {
            "type": "normalized_weighted_average",
            "nonnegative_activation": "max(0, alpha)",
            "zero_denominator_fallbacks": {
                "GUIDANCE": 0.0,
                "BREADTH": 1.0,
                "EXPLORE": 0.65,
                "REFINE": 0.35,
                "STOP": 0.05,
            },
        },
        "rule_registry": registry,
        "metadata": {
            "observational_only": True,
            "memberships_captured_from_inference_pass": True,
            "activations_captured_from_inference_pass": True,
            "rule_count": len(registry),
        },
    }
def fuzzy_control(
    *,
    reliability: float,
    conflict: float,
    semantic_discriminability: float,
    empirical_strength: float,
    plateau_strength: float = 0.0,
    refinement_completeness: float = 0.0,
    unexplored_potential: float = 1.0,
    incumbent_stability: float = 0.0,
    model_stagnation: float = 0.0,
    global_recent_information_gain: float = 1.0,
    global_search_stagnation: float = 0.0,
    family_saturation: float = 0.0,
    model_refinement_maturity: float = 1.0,
    incumbent_quality: float = 0.0,
    family_coverage: float = 0.0,
    distinct_model_coverage: float = 0.0,
    cross_family_diversity: float = 0.0,
    semantic_empirical_misalignment: float = 0.0,
    recent_relative_improvement: float = 0.0,
    absolute_incumbent_quality: float = 0.0,
    relative_incumbent_quality: float = 0.0,
    budget_coverage: float = 0.0,
    challenger_potential: float = 0.0,
    challenger_refinement_maturity: float = 0.0,
    refinement_responsiveness: float = 0.0,
    remaining_refinement_capacity: float = 0.0,
    unresolved_analytical_evidence: float = 0.0,
    analytical_evidence_urgency: float | None = None,
    refinement_information_gain: float = 1.0,
    model_refinement_information_gain: float | None = None,
    intra_family_model_coverage: float = 0.0,
    family_residual_potential: float = 0.0,
    global_model_residual_potential: float = 0.0,
    residual_search_value: float | None = None,
    preprocessing_refinement_potential: float = 0.0,
    raw_challenger_potential: float = 0.0,
    effective_challenger_potential: float | None = None,
    incumbent_dominance: float = 0.0,
    dominance_confidence: float | None = None,
    evidence_supported_dominance: float | None = None,
    search_exhaustion: float | None = None,
    expected_search_value: float = 1.0,
    explore_model_in_family_potential: float = 0.0,
    late_breakthrough: float = 0.0,
    remaining_budget_ratio: float = 1.0,
    capture_full_trace: bool = False,
) -> SearchControl:
    """Sugeno-style controller producing EXPLORE, REFINE and STOP together."""
    r, c, d, e = map(
        clamp, (reliability, conflict, semantic_discriminability, empirical_strength)
    )
    p, completeness, unexplored = map(
        clamp,
        (plateau_strength, refinement_completeness, unexplored_potential),
    )
    stability, stagnation, global_gain, global_stagnation, saturation, maturity = map(
        clamp,
        (
            incumbent_stability,
            model_stagnation,
            global_recent_information_gain,
            global_search_stagnation,
            family_saturation,
            model_refinement_maturity,
        ),
    )
    quality, coverage, misalignment, recent_improvement = map(
        clamp,
        (
            incumbent_quality,
            family_coverage,
            semantic_empirical_misalignment,
            recent_relative_improvement,
        ),
    )
    absolute_quality, relative_quality, budget = map(
        clamp,
        (absolute_incumbent_quality, relative_incumbent_quality, budget_coverage),
    )
    challenger_maturity, responsiveness, remaining_capacity, unresolved = map(
        clamp,
        (
            challenger_refinement_maturity,
            refinement_responsiveness,
            remaining_refinement_capacity,
            unresolved_analytical_evidence,
        ),
    )
    raw_challenger = clamp(raw_challenger_potential or challenger_potential)
    challenger = clamp(
        challenger_potential
        if effective_challenger_potential is None
        else effective_challenger_potential
    )
    urgency = clamp(
        unresolved * (1.0 - 0.65 * quality)
        if analytical_evidence_urgency is None
        else analytical_evidence_urgency
    )
    information_gain, intra_coverage, family_residual, residual_value, preprocessing_potential = map(
        clamp,
        (
            refinement_information_gain
            if model_refinement_information_gain is None
            else model_refinement_information_gain,
            intra_family_model_coverage,
            family_residual_potential,
            family_residual_potential
            if residual_search_value is None
            else residual_search_value,
            preprocessing_refinement_potential,
        ),
    )
    dominance = clamp(incumbent_dominance)
    dominance_support = clamp(
        0.10
        + 0.25 * coverage
        + 0.20 * distinct_model_coverage
        + 0.20 * cross_family_diversity
        + 0.25 * e
        if dominance_confidence is None
        else dominance_confidence
    )
    supported_dominance = clamp(
        dominance * dominance_support
        if evidence_supported_dominance is None
        else evidence_supported_dominance
    )
    model_coverage, family_diversity = map(
        clamp, (distinct_model_coverage, cross_family_diversity)
    )
    exhaustion = clamp(
        0.5 if search_exhaustion is None else search_exhaustion
    )
    expected_value = clamp(expected_search_value)
    in_family_potential = clamp(explore_model_in_family_potential)
    breakthrough, remaining_ratio = map(
        clamp, (late_breakthrough, remaining_budget_ratio)
    )
    e_l, e_m, e_h = low_input(e), medium_input(e), high_input(e)
    p_l, p_m, p_h = low_input(p), medium_input(p), high_input(p)
    rc_l, rc_m, rc_h = (
        low_input(completeness),
        medium_input(completeness),
        high_input(completeness),
    )
    u_l, u_m, u_h = (
        low_input(unexplored),
        medium_input(unexplored),
        high_input(unexplored),
    )
    stability_l, stability_m, stability_h = (
        low_input(stability), medium_input(stability), high_input(stability)
    )
    stagnation_l, stagnation_m, stagnation_h = (
        low_input(stagnation), medium_input(stagnation), high_input(stagnation)
    )
    global_gain_l, global_gain_m, global_gain_h = (
        low_input(global_gain), medium_input(global_gain), high_input(global_gain)
    )
    global_stagnation_l, global_stagnation_m, global_stagnation_h = (
        low_input(global_stagnation),
        medium_input(global_stagnation),
        high_input(global_stagnation),
    )
    saturation_l, saturation_m, saturation_h = (
        low_input(saturation), medium_input(saturation), high_input(saturation)
    )
    maturity_l, maturity_m, maturity_h = (
        low_input(maturity), medium_input(maturity), high_input(maturity)
    )
    quality_l, quality_m, quality_h = (
        low_input(quality), medium_input(quality), high_input(quality)
    )
    absolute_quality_l, absolute_quality_m, absolute_quality_h = (
        low_input(absolute_quality),
        medium_input(absolute_quality),
        high_input(absolute_quality),
    )
    coverage_l, coverage_m, coverage_h = (
        low_input(coverage), medium_input(coverage), high_input(coverage)
    )
    misalignment_l, misalignment_m, misalignment_h = (
        low_input(misalignment), medium_input(misalignment), high_input(misalignment)
    )
    improvement_l = low_input(recent_improvement)
    # Captured for diagnostics only; neither degree participates in a rule.
    improvement_m = medium_input(recent_improvement)
    improvement_h = high_input(recent_improvement)
    challenger_l, challenger_m, challenger_h = (
        low_input(challenger), medium_input(challenger), high_input(challenger)
    )
    challenger_maturity_l, challenger_maturity_m, challenger_maturity_h = (
        low_input(challenger_maturity),
        medium_input(challenger_maturity),
        high_input(challenger_maturity),
    )
    responsiveness_l, responsiveness_m, responsiveness_h = (
        low_input(responsiveness), medium_input(responsiveness), high_input(responsiveness)
    )
    capacity_l, capacity_m, capacity_h = (
        low_input(remaining_capacity), medium_input(remaining_capacity), high_input(remaining_capacity)
    )
    unresolved_l, unresolved_m, unresolved_h = (
        low_input(unresolved), medium_input(unresolved), high_input(unresolved)
    )
    urgency_l, urgency_m, urgency_h = (
        low_input(urgency), medium_input(urgency), high_input(urgency)
    )
    info_l, info_m, info_h = (
        low_input(information_gain), medium_input(information_gain), high_input(information_gain)
    )
    intra_l, intra_m, intra_h = (
        low_input(intra_coverage), medium_input(intra_coverage), high_input(intra_coverage)
    )
    residual_l, residual_m, residual_h = (
        low_input(residual_value), medium_input(residual_value), high_input(residual_value)
    )
    model_residual = clamp(global_model_residual_potential)
    model_residual_l, model_residual_m, model_residual_h = (
        low_input(model_residual),
        medium_input(model_residual),
        high_input(model_residual),
    )
    preprocessing_l, preprocessing_m, preprocessing_h = (
        low_input(preprocessing_potential),
        medium_input(preprocessing_potential),
        high_input(preprocessing_potential),
    )
    dominance_l, dominance_m, dominance_h = (
        low_input(dominance), medium_input(dominance), high_input(dominance)
    )
    supported_dominance_l, supported_dominance_m, supported_dominance_h = (
        low_input(supported_dominance),
        medium_input(supported_dominance),
        high_input(supported_dominance),
    )
    exhaustion_l, exhaustion_m, exhaustion_h = (
        low_input(exhaustion), medium_input(exhaustion), high_input(exhaustion)
    )
    expected_l, expected_m, expected_h = (
        low_input(expected_value), medium_input(expected_value), high_input(expected_value)
    )
    breakthrough_l, breakthrough_m, breakthrough_h = (
        low_input(breakthrough), medium_input(breakthrough), high_input(breakthrough)
    )
    remaining_l, remaining_m, remaining_h = (
        low_input(remaining_ratio), medium_input(remaining_ratio), high_input(remaining_ratio)
    )
    evidence_memberships: dict[str, dict[str, float]] = {}
    uncertain, moderate, confident = _rule_activations(
        r,
        c,
        d,
        evidence_memberships if capture_full_trace else None,
    )

    explore_model_in_family = clamp(
        in_family_potential
        * (
            0.35
            + 0.35 * stagnation
            + 0.30 * (1.0 - information_gain)
        )
    )
    explore_new_family = clamp(
        residual_value
        * (
            0.20
            + 0.25 * global_stagnation
            + 0.20 * max(saturation, maturity)
            + 0.20 * (1.0 - coverage)
            + 0.15 * remaining_ratio
        )
        * (0.65 + 0.35 * (1.0 - global_gain))
        * (1.0 - 0.30 * quality)
    )
    in_family_l, in_family_m, in_family_h = (
        low_input(explore_model_in_family),
        medium_input(explore_model_in_family),
        high_input(explore_model_in_family),
    )
    escape_l, escape_m, escape_h = (
        low_input(explore_new_family),
        medium_input(explore_new_family),
        high_input(explore_new_family),
    )

    trace_rules: dict[str, list[tuple[float, float]]] | None = {} if capture_full_trace else None
    guidance = _weighted(
        [(uncertain, 0.05), (moderate, 0.45), (confident, 0.90), (e_h, 0.15)],
        0.0,
        "guidance",
        trace_rules,
    )
    breadth = _weighted(
        [
            (uncertain, 1.00),
            (moderate, 0.65),
            (confident, 0.30),
            (e_l, 0.85),
            (min(global_stagnation_h, residual_h), 0.95),
            (saturation_h, 0.95),
            (misalignment_h, 0.90),
        ],
        1.0,
        "breadth",
        trace_rules,
    )
    explore = _weighted(
        [
            (u_h, 0.92),
            (u_m, 0.58),
            (u_l, 0.16),
            (uncertain, 0.82),
            (moderate, 0.58),
            (confident, 0.38),
            (e_l, 0.82),
            (min(e_h, p_h, rc_h), 0.12),
            (min(stagnation_h, residual_h), 0.92),
            (min(global_stagnation_h, global_gain_l, residual_h), 0.99),
            (min(global_gain_l, residual_h), 0.96),
            (min(saturation_h, max(residual_m, residual_h)), 0.95),
            (min(max(stagnation_m, stagnation_h), max(maturity_m, maturity_h), improvement_l), 0.90),
            (misalignment_h, 0.92),
            (min(coverage_l, u_h), 0.97),
            (min(coverage_l, e_l, remaining_h, max(residual_m, residual_h)), 0.98),
            (min(global_stagnation_h, saturation_h, coverage_l, residual_h, remaining_h), 0.99),
            (min(dominance_h, supported_dominance_l, remaining_h), 0.86),
            (min(stability_h, maturity_h, coverage_h), 0.15),
            (min(stability_h, maturity_h, saturation_l, max(u_l, u_m)), 0.12),
            (urgency_h, 0.96),
            (urgency_m, 0.68),
            (min(stagnation_h, info_l, residual_h), 0.94),
            (min(info_l, residual_h), 0.98),
            (residual_h, 0.93),
            (residual_m, 0.70),
            (model_residual_h, 0.97),
            (model_residual_m, 0.74),
            (min(intra_l, max(residual_m, residual_h)), 0.96),
            (min(breakthrough_h, remaining_l), 0.10),
            (min(stagnation_h, responsiveness_l), 0.96),
            (min(quality_l, max(e_l, medium_input(e)), max(exhaustion_l, residual_m, residual_h, urgency_m, urgency_h)), 0.90),
            (min(quality_l, max(exhaustion_l, residual_m, residual_h, urgency_m, urgency_h)), 0.95),
            (min(responsiveness_h, max(capacity_m, capacity_h)), 0.25),
            (exhaustion_l, 0.94),
            (exhaustion_h, 0.05),
            (in_family_h, 0.99),
            (in_family_m, 0.78),
            (escape_h, 0.99),
            (escape_m, 0.82),
        ],
        0.65,
        "explore",
        trace_rules,
    )
    refine = _weighted(
        [
            (rc_l, 0.90),
            (rc_m, 0.62),
            (min(rc_h, p_h, maturity_h), 0.14),
            (uncertain, 0.28),
            (moderate, 0.52),
            (confident, 0.66),
            (min(e_h, max(rc_l, rc_m)), 0.86),
            (min(e_h, rc_h, p_h, maturity_h), 0.12),
            (min(quality_h, maturity_l, stagnation_l), 0.97),
            (min(global_stagnation_l, info_h), 0.95),
            (min(quality_h, maturity_l), 0.93),
            (stagnation_h, 0.05),
            (saturation_h, 0.08),
            (min(maturity_h, improvement_l), 0.18),
            (min(challenger_h, challenger_maturity_l), 0.98),
            (min(max(challenger_m, challenger_h), max(u_m, u_h)), 0.84),
            (min(responsiveness_h, max(maturity_l, maturity_m), max(capacity_m, capacity_h)), 0.91),
            (min(responsiveness_h, max(capacity_m, capacity_h)), 0.96),
            (min(info_l, stagnation_h), 0.03),
            (min(info_l, stagnation_h, max(residual_l, residual_m, residual_h)), 0.02),
            (min(info_l, responsiveness_l, maturity_h), 0.04),
            (min(quality_h, maturity_h, preprocessing_h), 0.95),
            (min(preprocessing_l, stagnation_h), 0.03),
            (min(breakthrough_h, responsiveness_h, remaining_l), 0.99),
            (min(breakthrough_m, responsiveness_h, remaining_l), 0.93),
            (min(breakthrough_h, responsiveness_h), 0.99),
            (min(stagnation_h, responsiveness_l), 0.04),
            (capacity_l, 0.10),
            (quality_l, 0.12),
            (min(expected_h, max(responsiveness_h, challenger_h)), 0.96),
            (min(in_family_h, info_l, stagnation_h), 0.02),
            (min(escape_h, global_stagnation_h, saturation_h), 0.05),
            (min(global_stagnation_h, global_gain_l, supported_dominance_h, residual_l, responsiveness_l), 0.03),
        ],
        0.35,
        "refine",
        trace_rules,
    )
    stop = _weighted(
        [
            # A low-weight always-on rule prevents one weakly activated
            # consequent from creating a near-binary 0.05 -> 0.9 jump.
            (0.35, 0.05),
            (min(p_h, rc_h, u_l, maturity_h, quality_h, e_h, challenger_l, unresolved_l), 0.90),
            (min(p_h, rc_m, u_l, maturity_h, quality_h, e_h, challenger_l, unresolved_l), 0.70),
            (min(p_l, max(maturity_l, stability_l)), 0.05),
            (u_h, 0.08),
            (min(rc_l, e_h, maturity_l), 0.08),
            (min(p_h, e_h, rc_h, maturity_h, quality_h, challenger_l, unresolved_l, max(u_l, u_m)), 0.88),
            (min(p_m, u_m), 0.50),
            (min(stability_h, maturity_h, quality_h, e_h, max(stagnation_l, stagnation_m), max(u_l, min(u_m, coverage_h)), challenger_l, unresolved_l), 0.88),
            (min(stability_h, maturity_h, quality_h, e_h, coverage_h, u_l, challenger_l, unresolved_l), 0.92),
            (min(maturity_l, quality_h), 0.03),
            (min(u_h, coverage_l), 0.02),
            (quality_l, 0.06),
            (min(max(e_l, medium_input(e)), max(u_m, u_h)), 0.08),
            (challenger_h, 0.04),
            (min(max(challenger_m, challenger_h), max(u_m, u_h)), 0.08),
            (urgency_h, 0.03),
            (urgency_m, 0.18),
            (residual_h, 0.04),
            (model_residual_h, 0.03),
            (min(info_l, maturity_h, u_l, preprocessing_l), 0.82),
            (min(info_l, challenger_l, residual_l), 0.88),
            (min(global_stagnation_h, global_gain_l, residual_l, stability_h, challenger_l, urgency_l, absolute_quality_h), 0.98),
            (min(global_gain_l, stability_h, challenger_l, residual_l, absolute_quality_h), 0.90),
            (min(supported_dominance_h, quality_h, stability_h, urgency_l, challenger_l, global_stagnation_h, residual_l), 0.96),
            (min(supported_dominance_m, quality_h, stability_h, urgency_l, challenger_l, residual_l), 0.72),
            (min(dominance_h, supported_dominance_l, remaining_h), 0.08),
            (min(coverage_l, e_l, remaining_h), 0.06),
            (min(escape_h, global_stagnation_h, saturation_h), 0.04),
            (min(breakthrough_h, responsiveness_h), 0.03),
            (min(responsiveness_h, max(maturity_l, maturity_m)), 0.20),
            (min(exhaustion_h, quality_h, stability_h), 0.98),
            (min(exhaustion_h, global_stagnation_h, global_gain_l, residual_l, model_residual_l, e_h), 0.99),
            (exhaustion_l, 0.03),
            (min(dominance_l, residual_h), 0.02),
            (min(remaining_l, supported_dominance_h, challenger_l, urgency_l, exhaustion_h), 0.99),
            (min(expected_l, quality_h, stability_h, challenger_l, residual_l), 0.94),
        ],
        0.05,
        "stop",
        trace_rules,
    )
    dominant_rules = {
        "EXPLORE": max(
            (
                (u_h, "unexplored_potential_high"),
                (urgency_h, "analytical_evidence_urgency_high"),
                (min(global_stagnation_h, global_gain_l, residual_h), "global_stagnation_high_residual_value_high"),
                (min(global_gain_l, residual_h), "global_information_low_residual_value_high"),
                (escape_h, "controlled_family_escape"),
                (exhaustion_l, "search_exhaustion_low"),
                (in_family_h, "explore_model_in_family"),
                (uncertain, "evidence_uncertain"),
            ),
            key=lambda item: item[0],
        )[1],
        "REFINE": max(
            (
                (min(challenger_h, challenger_maturity_l), "effective_challenger_high_maturity_low"),
                (min(quality_h, maturity_h, preprocessing_h), "preprocessing_potential_high"),
                (min(breakthrough_h, responsiveness_h, remaining_l), "late_breakthrough_high"),
                (min(responsiveness_h, max(capacity_m, capacity_h)), "responsive_model_has_capacity"),
                (min(info_h, global_stagnation_l), "model_information_high_global_stagnation_low"),
                (min(info_l, stagnation_h), "model_information_low_model_stagnation_high"),
            ),
            key=lambda item: item[0],
        )[1],
        "STOP": max(
            (
                (min(supported_dominance_h, quality_h, stability_h, urgency_l, challenger_l), "evidence_supported_dominant_incumbent"),
                (min(exhaustion_h, quality_h, stability_h), "search_exhaustion_high"),
                (min(exhaustion_h, global_stagnation_h, global_gain_l, residual_l, model_residual_l, e_h), "exhausted_unsuccessful_search"),
                (min(global_stagnation_h, global_gain_l, residual_l, urgency_l, absolute_quality_h), "global_stagnation_low_residual_value"),
                (min(info_l, challenger_l, residual_l), "redundant_local_search_low_residual_value"),
                (min(info_l, maturity_h, u_l, preprocessing_l), "search_mature_information_gain_low"),
                (min(p_h, rc_h, u_l, maturity_h, quality_h), "plateau_complete_quality_high"),
                (urgency_h, "analytical_urgency_high_blocks_stop"),
                (residual_h, "family_residual_high_blocks_stop"),
            ),
            key=lambda item: item[0],
        )[1],
    }

    fuzzy_inference_trace: dict[str, Any] = {}
    if capture_full_trace:
        memberships = {
            **evidence_memberships,
            "empirical_strength": _membership_vector(e_l, e_m, e_h),
            "plateau_strength": _membership_vector(p_l, p_m, p_h),
            "refinement_completeness": _membership_vector(rc_l, rc_m, rc_h),
            "unexplored_potential": _membership_vector(u_l, u_m, u_h),
            "incumbent_stability": _membership_vector(stability_l, stability_m, stability_h),
            "model_stagnation": _membership_vector(stagnation_l, stagnation_m, stagnation_h),
            "global_recent_information_gain": _membership_vector(global_gain_l, global_gain_m, global_gain_h),
            "global_search_stagnation": _membership_vector(global_stagnation_l, global_stagnation_m, global_stagnation_h),
            "family_saturation": _membership_vector(saturation_l, saturation_m, saturation_h),
            "model_refinement_maturity": _membership_vector(maturity_l, maturity_m, maturity_h),
            "incumbent_quality": _membership_vector(quality_l, quality_m, quality_h),
            "absolute_incumbent_quality": _membership_vector(absolute_quality_l, absolute_quality_m, absolute_quality_h),
            "family_coverage": _membership_vector(coverage_l, coverage_m, coverage_h),
            "semantic_empirical_misalignment": _membership_vector(misalignment_l, misalignment_m, misalignment_h),
            "recent_relative_improvement": _membership_vector(improvement_l, improvement_m, improvement_h),
            "effective_challenger_potential": _membership_vector(challenger_l, challenger_m, challenger_h),
            "challenger_refinement_maturity": _membership_vector(challenger_maturity_l, challenger_maturity_m, challenger_maturity_h),
            "refinement_responsiveness": _membership_vector(responsiveness_l, responsiveness_m, responsiveness_h),
            "remaining_refinement_capacity": _membership_vector(capacity_l, capacity_m, capacity_h),
            "unresolved_analytical_evidence": _membership_vector(unresolved_l, unresolved_m, unresolved_h),
            "analytical_evidence_urgency": _membership_vector(urgency_l, urgency_m, urgency_h),
            "model_refinement_information_gain": _membership_vector(info_l, info_m, info_h),
            "intra_family_model_coverage": _membership_vector(intra_l, intra_m, intra_h),
            "residual_search_value": _membership_vector(residual_l, residual_m, residual_h),
            "global_model_residual_potential": _membership_vector(model_residual_l, model_residual_m, model_residual_h),
            "preprocessing_refinement_potential": _membership_vector(preprocessing_l, preprocessing_m, preprocessing_h),
            "incumbent_dominance": _membership_vector(dominance_l, dominance_m, dominance_h),
            "evidence_supported_dominance": _membership_vector(supported_dominance_l, supported_dominance_m, supported_dominance_h),
            "search_exhaustion": _membership_vector(exhaustion_l, exhaustion_m, exhaustion_h),
            "expected_search_value": _membership_vector(expected_l, expected_m, expected_h),
            "late_breakthrough": _membership_vector(breakthrough_l, breakthrough_m, breakthrough_h),
            "remaining_budget_ratio": _membership_vector(remaining_l, remaining_m, remaining_h),
            "explore_model_in_family_strength": _membership_vector(in_family_l, in_family_m, in_family_h),
            "explore_new_family_strength": _membership_vector(escape_l, escape_m, escape_h),
        }
        direct_inputs = {
            "reliability": r, "conflict": c,
            "semantic_discriminability": d, "discriminability": d,
            "empirical_strength": e, "plateau_strength": p,
            "refinement_completeness": completeness, "unexplored_potential": unexplored,
            "incumbent_stability": stability, "model_stagnation": stagnation,
            "global_recent_information_gain": global_gain,
            "global_search_stagnation": global_stagnation,
            "family_saturation": saturation, "model_refinement_maturity": maturity,
            "incumbent_quality": quality, "absolute_incumbent_quality": absolute_quality,
            "family_coverage": coverage, "semantic_empirical_misalignment": misalignment,
            "recent_relative_improvement": recent_improvement,
            "effective_challenger_potential": challenger,
            "challenger_refinement_maturity": challenger_maturity,
            "refinement_responsiveness": responsiveness,
            "remaining_refinement_capacity": remaining_capacity,
            "unresolved_analytical_evidence": unresolved,
            "analytical_evidence_urgency": urgency,
            "model_refinement_information_gain": information_gain,
            "intra_family_model_coverage": intra_coverage,
            "residual_search_value": residual_value,
            "global_model_residual_potential": model_residual,
            "preprocessing_refinement_potential": preprocessing_potential,
            "incumbent_dominance": dominance,
            "evidence_supported_dominance": supported_dominance,
            "search_exhaustion": exhaustion, "expected_search_value": expected_value,
            "late_breakthrough": breakthrough, "remaining_budget_ratio": remaining_ratio,
            "explore_model_in_family_strength": explore_model_in_family,
            "explore_new_family_strength": explore_new_family,
        }
        fuzzy_inference_trace = _build_inference_trace(
            inputs=direct_inputs,
            memberships=memberships,
            rule_groups=trace_rules or {},
            raw_action_strengths={
                "EXPLORE": explore, "REFINE": refine, "STOP": stop,
            },
        )
        fuzzy_inference_trace["defuzzified_outputs"] = {
            "GUIDANCE": guidance,
            "BREADTH": breadth,
            "EXPLORE": explore,
            "REFINE": refine,
            "STOP": stop,
        }

    return SearchControl(
        mode="fuzzy",
        guidance_weight=guidance,
        search_breadth=breadth,
        explore_strength=explore,
        refine_strength=refine,
        stop_strength=stop,
        plateau_strength=p,
        refinement_completeness=completeness,
        unexplored_potential=unexplored,
        reliability=r,
        conflict=c,
        semantic_discriminability=d,
        empirical_strength=e,
        incumbent_stability=stability,
        model_stagnation=stagnation,
        global_recent_information_gain=global_gain,
        global_search_stagnation=global_stagnation,
        family_saturation=saturation,
        model_refinement_maturity=maturity,
        incumbent_quality=quality,
        absolute_incumbent_quality=absolute_quality,
        relative_incumbent_quality=relative_quality,
        family_coverage=coverage,
        distinct_model_coverage=model_coverage,
        cross_family_diversity=family_diversity,
        semantic_empirical_misalignment=misalignment,
        recent_relative_improvement=recent_improvement,
        budget_coverage=budget,
        challenger_potential=challenger,
        challenger_refinement_maturity=challenger_maturity,
        refinement_responsiveness=responsiveness,
        remaining_refinement_capacity=remaining_capacity,
        unresolved_analytical_evidence=unresolved,
        analytical_evidence_urgency=urgency,
        model_refinement_information_gain=information_gain,
        refinement_information_gain=information_gain,
        intra_family_model_coverage=intra_coverage,
        family_residual_potential=family_residual,
        global_model_residual_potential=model_residual,
        residual_search_value=residual_value,
        preprocessing_refinement_potential=preprocessing_potential,
        raw_challenger_potential=raw_challenger,
        effective_challenger_potential=challenger,
        incumbent_dominance=dominance,
        dominance_confidence=dominance_support,
        evidence_supported_dominance=supported_dominance,
        search_exhaustion=exhaustion,
        expected_search_value=expected_value,
        explore_new_family_strength=explore_new_family,
        explore_model_in_family_strength=explore_model_in_family,
        late_breakthrough=breakthrough,
        remaining_budget_ratio=remaining_ratio,
        dominant_rules=dominant_rules,
        fuzzy_inference_trace=fuzzy_inference_trace,
    )


def build_search_control(
    mode: str,
    evidence: Mapping[str, Any],
    empirical_strength: float,
    *,
    plateau_strength: float = 0.0,
    refinement_completeness: float = 0.0,
    unexplored_potential: float = 1.0,
    incumbent_stability: float = 0.0,
    model_stagnation: float = 0.0,
    global_recent_information_gain: float = 1.0,
    global_search_stagnation: float = 0.0,
    family_saturation: float = 0.0,
    model_refinement_maturity: float = 1.0,
    incumbent_quality: float = 0.0,
    family_coverage: float = 0.0,
    distinct_model_coverage: float = 0.0,
    cross_family_diversity: float = 0.0,
    semantic_empirical_misalignment: float = 0.0,
    recent_relative_improvement: float = 0.0,
    absolute_incumbent_quality: float = 0.0,
    relative_incumbent_quality: float = 0.0,
    budget_coverage: float = 0.0,
    challenger_potential: float = 0.0,
    challenger_refinement_maturity: float = 0.0,
    refinement_responsiveness: float = 0.0,
    remaining_refinement_capacity: float = 0.0,
    unresolved_analytical_evidence: float = 0.0,
    analytical_evidence_urgency: float | None = None,
    refinement_information_gain: float = 1.0,
    model_refinement_information_gain: float | None = None,
    intra_family_model_coverage: float = 0.0,
    family_residual_potential: float = 0.0,
    global_model_residual_potential: float = 0.0,
    residual_search_value: float | None = None,
    preprocessing_refinement_potential: float = 0.0,
    raw_challenger_potential: float = 0.0,
    effective_challenger_potential: float | None = None,
    incumbent_dominance: float = 0.0,
    dominance_confidence: float | None = None,
    evidence_supported_dominance: float | None = None,
    search_exhaustion: float | None = None,
    expected_search_value: float = 1.0,
    explore_model_in_family_potential: float = 0.0,
    late_breakthrough: float = 0.0,
    remaining_budget_ratio: float = 1.0,
    capture_full_trace: bool = False,
) -> SearchControl:
    normalized = str(mode).strip().lower()
    if normalized != "fuzzy":
        raise ValueError("T-FUSE controller accepts only fuzzy influence")
    return fuzzy_control(
        reliability=float(evidence.get("fusion_reliability", 0.0)),
        conflict=float(evidence.get("conflict", 0.5)),
        semantic_discriminability=float(
            evidence.get(
                "semantic_discriminability", evidence.get("discriminability", 0.0)
            )
        ),
        empirical_strength=empirical_strength,
        plateau_strength=plateau_strength,
        refinement_completeness=refinement_completeness,
        unexplored_potential=unexplored_potential,
        incumbent_stability=incumbent_stability,
        model_stagnation=model_stagnation,
        global_recent_information_gain=global_recent_information_gain,
        global_search_stagnation=global_search_stagnation,
        family_saturation=family_saturation,
        model_refinement_maturity=model_refinement_maturity,
        incumbent_quality=incumbent_quality,
        family_coverage=family_coverage,
        distinct_model_coverage=distinct_model_coverage,
        cross_family_diversity=cross_family_diversity,
        semantic_empirical_misalignment=semantic_empirical_misalignment,
        recent_relative_improvement=recent_relative_improvement,
        absolute_incumbent_quality=absolute_incumbent_quality,
        relative_incumbent_quality=relative_incumbent_quality,
        budget_coverage=budget_coverage,
        challenger_potential=challenger_potential,
        challenger_refinement_maturity=challenger_refinement_maturity,
        refinement_responsiveness=refinement_responsiveness,
        remaining_refinement_capacity=remaining_refinement_capacity,
        unresolved_analytical_evidence=unresolved_analytical_evidence,
        analytical_evidence_urgency=analytical_evidence_urgency,
        refinement_information_gain=refinement_information_gain,
        model_refinement_information_gain=model_refinement_information_gain,
        intra_family_model_coverage=intra_family_model_coverage,
        family_residual_potential=family_residual_potential,
        global_model_residual_potential=global_model_residual_potential,
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
        remaining_budget_ratio=remaining_budget_ratio,
        capture_full_trace=capture_full_trace,
    )


__all__ = [
    "SearchControl",
    "build_search_control",
    "fuzzy_control",
]
