"""Planner node: prerequisites, strategic decision and safety validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from agents.planner_agent import PlannerAgent, PlannerDecision
from evaluation.search_trace import append_search_event
from search.safety import SearchSafetyGuard
from utils.training_budget import count_valid_trainings
from utils.configuration_identity import log_exploration_status
from utils.stop_policy import stop_eligibility
from utils.model_family_state import MODEL_TO_FAMILY


_MAX_PLANNER_CALLS = 200


@dataclass
class PlannerNode:
    planner: PlannerAgent
    safety_guard: SearchSafetyGuard = field(default_factory=SearchSafetyGuard)
    max_replanning_attempts: int = 3

    @staticmethod
    def _prerequisite_result(
        *, action: str, planner_call_count: int, iteration: int
    ) -> Dict[str, Any]:
        signals = {
            "next_action": action,
            "reason": f"{action}_prerequisite",
            "confidence": 1.0,
            "action": None,
        }
        return {
            "planner_signals": signals,
            "next_node": action,
            "stop": False,
            "done": False,
            "orchestration_reason": signals["reason"],
            "iteration": iteration,
            "planner_call_count": planner_call_count,
            "history": [{"iteration": iteration, "node": "planner", "decision": dict(signals)}],
        }

    @staticmethod
    def _request_id(decision: PlannerDecision, planner_call_count: int) -> str | None:
        if not decision.selected_model:
            return None
        return f"planner-request-{planner_call_count}-{decision.selected_model}"

    @classmethod
    def _candidate_request(
        cls, decision: PlannerDecision, state: Mapping[str, Any], planner_call_count: int
    ) -> dict[str, Any]:
        technical_controller_support = dict(decision.controller_support)
        technical_controller_support.pop("fuzzy_inference_trace", None)
        request = {
            "request_id": cls._request_id(decision, planner_call_count),
            "action": decision.action,
            "model": decision.selected_model,
            "family": decision.selected_family,
            "configuration_strategy": decision.configuration_strategy,
            "allowed_models": [decision.selected_model],
            "reachable_models": list(state.get("model_catalog", {}).get("available_models", [])),
            "priority_score": decision.decision_score,
            "priority_components": {
                "controller_support": technical_controller_support,
                "empirical_support": dict(decision.empirical_support),
            },
            "planner_decision": decision.to_dict(),
        }
        required_transform = decision.supporting_evidence.get(
            "required_target_transform"
        )
        if required_transform:
            request["required_target_transform"] = required_transform
        required_model = decision.supporting_evidence.get(
            "required_refinement_model"
        )
        if required_model:
            request["required_refinement_model"] = required_model
        return request

    @staticmethod
    def _fuzzy_snapshot(
        state: Mapping[str, Any],
        decision: PlannerDecision,
        planner_call_count: int,
        best: Mapping[str, Any] | None,
        accepted: bool,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        inference = decision.controller_support.get("fuzzy_inference_trace", {})
        if not isinstance(inference, Mapping) or not inference:
            return None, None
        snapshot = dict(inference)
        registry_value = snapshot.pop("rule_registry", [])
        registry = [dict(item) for item in registry_value if isinstance(item, Mapping)]
        raw = dict(snapshot.get("raw_action_strengths", {}))
        effective = dict(
            decision.supporting_evidence.get("effective_action_scores", raw)
        )
        actions = ("EXPLORE", "REFINE", "STOP")
        tie_priority = {"EXPLORE": 2, "REFINE": 3, "STOP": 1}
        raw_argmax = max(
            actions, key=lambda action: (float(raw.get(action, 0.0)), tie_priority[action])
        )
        effective_argmax = max(
            actions,
            key=lambda action: (
                float(effective.get(action, 0.0)), tie_priority[action]
            ),
        )
        executed = str(decision.action).upper() if accepted else None
        stop_eligible = bool(decision.supporting_evidence.get("stop_eligible", False))
        hard_stop = bool(decision.supporting_evidence.get("hard_stop_required", False))
        retry_override = accepted and executed == "RETRY"
        override = accepted and executed != raw_argmax
        reason = None
        if override and hard_stop:
            reason = "hard_budget_stop"
        elif override and retry_override:
            reason = "technical_retry_override"
        elif override and decision.supporting_evidence.get("stop_blocked", False):
            reason = decision.supporting_evidence.get("stop_block_reason")
        elif override:
            reason = decision.reason_code
        snapshot.update({
            "effective_action_strengths": effective,
            "selected_fuzzy_action": raw_argmax,
            "executed_planner_action": executed,
            "action_arbitration": {
                "raw_argmax_action": raw_argmax,
                "effective_argmax_action": effective_argmax,
                "executed_action": executed,
                "override_applied": override,
                "override_reason": reason,
                "stop_eligible": stop_eligible,
                "hard_stop_required": hard_stop,
                "technical_retry_override": retry_override,
                "decision_accepted_by_safety_guard": accepted,
                "downstream_strengths_modified": raw != effective,
            },
            "upstream_context": {
                "valid_training_index": int(
                    decision.supporting_evidence.get(
                        "valid_trainings",
                        state.get(
                            "valid_training_count",
                            state.get(
                                "iteration",
                                count_valid_trainings(state.get("performance_history", [])),
                            ),
                        ),
                    )
                ),
                "planner_step": planner_call_count,
                "current_best_model": best.get("model") if best else None,
                "current_best_family": (
                    MODEL_TO_FAMILY.get(str(best.get("model"))) if best else None
                ),
                "current_best_validation_mase": best.get("mase") if best else None,
                "target_mase_reached": bool(
                    decision.controller_support.get(
                        "target_mase_reached",
                        state.get("target_mase_reached", False),
                    )
                ),
                "remaining_budget": decision.supporting_evidence.get("remaining_budget"),
                "remaining_budget_ratio": decision.controller_support.get(
                    "remaining_budget_ratio"
                ),
                "selected_family": decision.selected_family,
                "selected_model": decision.selected_model,
                "stop_reason": decision.supporting_evidence.get("stop_reason"),
            },
            "metadata": {
                **dict(snapshot.get("metadata", {})),
                "planner_arbitration_attached": True,
                "dataset": state.get("dataset_name") or state.get("series_id"),
                "run_id": state.get("run_id"),
                "seed": state.get("seed"),
            },
        })
        already_registered = any(
            isinstance(event, Mapping) and event.get("fuzzy_rule_registry")
            for event in state.get("search_trace", [])
        )
        return snapshot, None if already_registered else registry
    def _trace(
        self,
        state: Dict[str, Any],
        *,
        decision: PlannerDecision,
        accepted: bool,
        guard_reason: str | None,
        planner_call_count: int,
    ) -> None:
        history = self.planner.policy.history(state)
        best = min(history, key=lambda item: float(item["mase"]), default=None)
        control = decision.controller_support
        log_status = log_exploration_status(state)
        fuzzy_snapshot, fuzzy_registry = self._fuzzy_snapshot(
            state,
            decision,
            planner_call_count,
            best,
            accepted,
        )
        append_search_event(
            state,
            event_type="planner_decision",
            payload={
                "request_id": self._request_id(decision, planner_call_count),
                **({"fuzzy_controller_snapshot": fuzzy_snapshot} if fuzzy_snapshot else {}),
                **({"fuzzy_rule_registry": fuzzy_registry} if fuzzy_registry else {}),
                "influence_mode": state.get("influence_mode", "fuzzy"),
                "action": decision.action,
                "selected_action": decision.action,
                "selected_family": decision.selected_family,
                "selected_model": decision.selected_model,
                "decision_score": decision.decision_score,
                "guidance_weight": control.get("guidance_weight", 0.0),
                "search_breadth": control.get("search_breadth", 1.0),
                "fusion_reliability": control.get("fusion_reliability", 0.0),
                "conflict": control.get("conflict", 0.0),
                "discriminability": control.get("discriminability", 0.0),
                "semantic_discriminability": control.get(
                    "semantic_discriminability",
                    control.get("discriminability", 0.0),
                ),
                "preferred_families": list(control.get("preferred_families", [])),
                "target_family_coverage": control.get("target_family_coverage", 0),
                "families_seen_before": list(
                    decision.supporting_evidence.get("families_seen_before", [])
                ),
                "selection_reason": decision.supporting_evidence.get(
                    "selection_reason", decision.reason_code
                ),
                "action_degrees": dict(control.get("action_degrees", {})),
                "valid_trainings": decision.supporting_evidence.get(
                    "valid_trainings", len(history)
                ),
                "remaining_budget": decision.supporting_evidence.get(
                    "remaining_budget"
                ),
                "minimum_valid_trainings_before_stop": (
                    decision.supporting_evidence.get(
                        "minimum_valid_trainings_before_stop", 5
                    )
                ),
                "stop_eligible": decision.supporting_evidence.get(
                    "stop_eligible", False
                ),
                "plateau_detected": decision.supporting_evidence.get(
                    "plateau_detected", False
                ),
                "required_target_transform": decision.supporting_evidence.get(
                    "required_target_transform"
                ),
                "required_refinement_model": decision.supporting_evidence.get(
                    "required_refinement_model"
                ),
                "mandatory_refinement_pending": (
                    decision.supporting_evidence.get(
                        "mandatory_refinement_pending", False
                    )
                ),
                "reasons_blocking_stop": list(
                    decision.supporting_evidence.get(
                        "reasons_blocking_stop", []
                    )
                ),
                "explore_score": decision.supporting_evidence.get(
                    "explore_score"
                ),
                "refine_score": decision.supporting_evidence.get(
                    "refine_score"
                ),
                "stop_score": decision.supporting_evidence.get("stop_score"),
                "stop_reason": decision.supporting_evidence.get("stop_reason"),
                "raw_fuzzy_stop_strength": (
                    decision.supporting_evidence.get(
                        "raw_fuzzy_stop_strength"
                    )
                ),
                "effective_stop_strength": (
                    decision.supporting_evidence.get(
                        "effective_stop_strength"
                    )
                ),
                "stop_blocked": decision.supporting_evidence.get(
                    "stop_blocked", False
                ),
                "stop_block_reason": decision.supporting_evidence.get(
                    "stop_block_reason"
                ),
                "family_escape_triggered": decision.supporting_evidence.get(
                    "family_escape_triggered", False
                ),
                "current_family": decision.supporting_evidence.get(
                    "current_family"
                ),
                "selected_escape_family": decision.supporting_evidence.get(
                    "selected_escape_family"
                ),
                "family_escape_reason": decision.supporting_evidence.get(
                    "family_escape_reason"
                ),
                "incumbent_stability": control.get("incumbent_stability", 0.0),
                "model_stagnation": control.get("model_stagnation", 0.0),
                "global_recent_information_gain": control.get(
                    "global_recent_information_gain", 1.0
                ),
                "global_search_stagnation": control.get(
                    "global_search_stagnation", 0.0
                ),
                "model_refinement_information_gain": control.get(
                    "model_refinement_information_gain",
                    control.get("refinement_information_gain", 0.0),
                ),
                "family_residual_potential": control.get(
                    "family_residual_potential", 0.0
                ),
                "model_coverage_global": control.get(
                    "distinct_model_coverage", 0.0
                ),
                "model_coverage_within_family": control.get(
                    "model_coverage_by_family", {}
                ),
                "family_model_residual_potential": control.get(
                    "family_model_residual_potential", {}
                ),
                "global_model_residual_potential": control.get(
                    "global_model_residual_potential", 0.0
                ),
                "residual_search_value": control.get(
                    "residual_search_value", 0.0
                ),
                "incumbent_quality": control.get("incumbent_quality", 0.0),
                "incumbent_dominance": control.get("incumbent_dominance", 0.0),
                "effective_challenger_potential": control.get(
                    "effective_challenger_potential", 0.0
                ),
                "analytical_evidence_urgency": control.get(
                    "analytical_evidence_urgency", 0.0
                ),
                "search_exhaustion": control.get("search_exhaustion", 0.0),
                "late_breakthrough": control.get("late_breakthrough", 0.0),
                "remaining_budget_ratio": control.get(
                    "remaining_budget_ratio", 0.0
                ),
                "family_saturation": control.get("family_saturation", 0.0),
                "model_refinement_maturity": control.get("model_refinement_maturity", 0.0),
                "trials_since_last_improvement": control.get("trials_since_last_improvement", 0),
                "current_family_attempts": control.get("current_family_attempts", 0),
                "current_model_refinements": control.get("current_model_refinements", 0),
                "recent_relative_improvement": control.get("recent_relative_improvement", 0.0),
                "unexplored_potential": control.get("unexplored_potential", 0.0),
                "explore_strength": control.get("explore_strength", 0.0),
                "refine_strength": control.get("refine_strength", 0.0),
                "stop_strength": control.get("stop_strength", 0.0),
                "exploration_scope": decision.supporting_evidence.get(
                    "exploration_scope"
                ),
                "family_model_coverage_before": (
                    decision.supporting_evidence.get(
                        "family_model_coverage_before"
                    )
                ),
                "family_model_residual_before": (
                    decision.supporting_evidence.get(
                        "family_model_residual_before"
                    )
                ),
                "target_mase_reached": bool(
                    state.get("target_mase_reached", False)
                ),
                "fuzzy_outputs": dict(
                    decision.supporting_evidence.get("fuzzy_outputs", {})
                ),
                "fuzzy_inputs": dict(
                    decision.supporting_evidence.get("fuzzy_inputs", {})
                ),
                "hard_constraints": dict(
                    decision.supporting_evidence.get("hard_constraints", {})
                ),
                "effective_action_scores": dict(
                    decision.supporting_evidence.get(
                        "effective_action_scores", {}
                    )
                ),
                "current_best_model": best.get("model") if best else None,
                "current_best_mase": best.get("mase") if best else None,
                "safety_guard_accepted": accepted,
                "safety_guard_reason": guard_reason,
                **log_status,
            },
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        iteration = count_valid_trainings(state.get("performance_history", []))
        planner_call_count = int(state.get("planner_call_count", 0)) + 1

        if not state.get("data_summary"):
            return self._prerequisite_result(
                action="analytical", planner_call_count=planner_call_count, iteration=iteration
            )
        if state.get("semantic_enabled") and not state.get("semantic_completed"):
            return self._prerequisite_result(
                action="semantic", planner_call_count=planner_call_count, iteration=iteration
            )

        rejections: list[dict[str, Any]] = []
        if planner_call_count >= _MAX_PLANNER_CALLS:
            decision = self.planner.replanning_exhausted(
                state, [{"reason": "planner_call_limit", "model": None}]
            )
            guard = self.safety_guard.validate(decision, state, planner_input=self.planner.build_input(state))
        else:
            decision = None
            guard = None
            attempts = max(1, int(state.get("max_replanning_attempts", self.max_replanning_attempts)))
            for _ in range(attempts):
                decision = self.planner.plan(state, safety_rejections=rejections)
                guard = self.safety_guard.validate(decision, state, planner_input=self.planner.build_input(state))
                self._trace(
                    state,
                    decision=decision,
                    accepted=guard.accepted,
                    guard_reason=guard.reason,
                    planner_call_count=planner_call_count,
                )
                if guard.accepted:
                    break
                rejections.append({"reason": guard.reason, "model": decision.selected_model})
            if guard is not None and not guard.accepted:
                decision = self.planner.replanning_exhausted(state, rejections)
                guard = self.safety_guard.validate(decision, state, planner_input=self.planner.build_input(state))

        assert decision is not None and guard is not None
        if not guard.accepted:
            raise RuntimeError(
                "SearchSafetyGuard rejected the Planner decision: "
                f"reason={guard.reason}; replanning_rejections={rejections}"
            )

        is_stop = decision.action == "stop"
        candidate_request = None if is_stop else self._candidate_request(
            decision, state, planner_call_count
        )
        request_history = list(state.get("candidate_request_history", []) or [])
        if candidate_request:
            request_history.append(candidate_request)

        signals = {
            **decision.to_dict(),
            "next_action": "END" if is_stop else "technical",
            "technical_mode": decision.action,
            "confidence": decision.decision_confidence,
            "reason": decision.reason_code,
            "candidate_request": candidate_request,
        }
        stop_metadata = (
            {
                "code": decision.reason_code,
                "message": decision.reason_code,
                "step": planner_call_count,
                "source": "planner",
                "supporting_evidence": dict(decision.supporting_evidence),
            }
            if is_stop
            else None
        )
        search_state = self.planner.policy.evaluate(state).to_dict()
        search_state.pop("fuzzy_inference_trace", None)
        log_status = log_exploration_status(state)
        search_state.update(log_status)
        search_state.update(stop_eligibility(state, self.planner.policy_config))
        return {
            "planner_signals": signals,
            "planner_decision": decision.to_dict(),
            "search_decision_state": search_state,
            **log_status,
            "candidate_request": candidate_request,
            "candidate_request_history": request_history,
            "next_node": signals["next_action"],
            "stop": is_stop,
            "done": is_stop,
            "orchestration_reason": decision.reason_code,
            "stop_reason": decision.reason_code if is_stop else None,
            "stop_metadata": stop_metadata,
            "iteration": iteration,
            "valid_training_count": iteration,
            "planner_call_count": planner_call_count,
            "technical_mode": decision.action,
            "safety_guard_result": guard.to_dict(),
            "replanning_attempts": len(rejections),
            "search_trace": state.get("search_trace", []),
            "history": [{"iteration": iteration, "node": "planner", "decision": decision.to_dict()}],
            "technical_status": None,
            "request_planner_replan": False,
            "technical_escalation": None,
            "technical_error": bool(
                decision.action == "retry"
                and state.get("technical_error", False)
            ),
            "non_retryable_candidate_failure": False,
            "last_technical_error": (
                state.get("last_technical_error")
                if decision.action == "retry"
                else None
            ),
            "technical_retry_count": (
                int(state.get("technical_retry_count", 0))
                if decision.action == "retry"
                else 0
            ),
            "technical_exhausted_models": list(state.get("technical_exhausted_models", []) or []),
        }


__all__ = ["PlannerNode"]
