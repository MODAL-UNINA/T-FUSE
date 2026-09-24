"""Deterministic validation boundary for Planner and Technical decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from agents.planner_agent import ALLOWED_ACTIONS, PlannerDecision
from evaluation.holdout import FORBIDDEN_SEARCH_KEYS
from utils.configuration_identity import canonical_configuration_signature
from utils.stop_policy import stop_eligibility
from utils.model_family_state import MODEL_TO_FAMILY


@dataclass(frozen=True)
class SafetyGuardResult:
    accepted: bool
    reason: str | None
    required_replanning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchSafetyGuard:
    @staticmethod
    def _reject(reason: str) -> SafetyGuardResult:
        return SafetyGuardResult(False, reason, True)

    @staticmethod
    def _contains_test_field(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key = str(key).lower()
                # Audit metadata emitted by leakage-safe preprocessing. This
                # carries no test values and is safe only in its negative form.
                if key == "test_target_modified" and item is False:
                    continue
                if key in FORBIDDEN_SEARCH_KEYS or key.startswith("test_"):
                    return True
                if SearchSafetyGuard._contains_test_field(item):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(SearchSafetyGuard._contains_test_field(item) for item in value)
        return False

    @staticmethod
    def _blocked_signatures(state: Mapping[str, Any]) -> set[str]:
        blocked: set[str] = set()
        for source in (
            state.get("performance_history", []) or [],
            state.get("candidate_ledger", []) or [],
            state.get("rejected_technical_candidates", []) or [],
        ):
            for item in source:
                if isinstance(item, Mapping) and item.get("configuration_signature"):
                    blocked.add(str(item["configuration_signature"]))
        memory = state.get("memory", {})
        if isinstance(memory, Mapping):
            for item in memory.get("failed_combinations", []) or []:
                if not isinstance(item, Mapping):
                    continue
                signature = item.get("configuration_signature")
                if not signature:
                    signature = canonical_configuration_signature(
                        str(item.get("model", "")),
                        item.get("hyperparameters", {}),
                        item.get("preprocessing", []),
                    )
                blocked.add(str(signature))
        return blocked

    def validate_configuration(
        self,
        decision: PlannerDecision | Mapping[str, Any],
        configuration: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> SafetyGuardResult:
        value = decision.to_dict() if isinstance(decision, PlannerDecision) else dict(decision)
        if self._contains_test_field(configuration):
            return self._reject("test_access_forbidden")
        model = configuration.get("model_type", configuration.get("model"))
        if model != value.get("selected_model"):
            return self._reject("candidate_substitution_forbidden")
        preprocessing = configuration.get("preprocessing", {})
        transformations = preprocessing.get("transformations", []) if isinstance(preprocessing, Mapping) else preprocessing
        signature = canonical_configuration_signature(
            str(model), configuration.get("hyperparameters", {}), transformations or []
        )
        if signature in self._blocked_signatures(state):
            return self._reject("duplicate_candidate")
        return SafetyGuardResult(True, None, False)

    def validate(
        self,
        decision: PlannerDecision | Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        planner_input: Mapping[str, Any] | None = None,
    ) -> SafetyGuardResult:
        value = decision.to_dict() if isinstance(decision, PlannerDecision) else dict(decision)
        if not value or value.get("action") not in ALLOWED_ACTIONS:
            return self._reject("invalid_decision_format")
        if self._contains_test_field(value) or self._contains_test_field(planner_input or {}):
            return self._reject("test_access_forbidden")

        action = str(value["action"])
        if action == "stop":
            if value.get("selected_model") is not None:
                return self._reject("invalid_stop_candidate")
            reason = str(value.get("reason_code", ""))
            if reason not in {
                "budget_exhausted",
                "stable_validation_plateau",
                "planner_converged",
                "fuzzy_controller_converged",
                "fuzzy_search_exhausted_no_target",
                "stable_best_candidate",
                "search_space_exhausted",
                "replanning_limit_exhausted",
            }:
                return self._reject("invalid_stop_request")
            status = stop_eligibility(state)
            if status["hard_stop_required"]:
                if reason != "budget_exhausted":
                    return self._reject("hard_budget_requires_budget_exhausted_stop")
                return SafetyGuardResult(True, None, False)
            if not status["stop_eligible"]:
                rejection_by_blocker = {
                    "minimum_valid_trainings_not_reached": (
                        "stop_rejected_minimum_budget"
                    ),
                    "required_raw_candidate_pending": (
                        "stop_rejected_required_raw_candidate"
                    ),
                    "required_log_refinement_pending": (
                        "stop_rejected_required_log_refinement"
                    ),
                    "required_clip_refinement_pending": (
                        "stop_rejected_required_clip_refinement"
                    ),
                    "current_best_refinement_incomplete": (
                        "stop_rejected_current_best_refinement"
                    ),
                }
                blockers = status["reasons_blocking_stop"]
                blocker = next(
                    (item for item in blockers if item in rejection_by_blocker),
                    blockers[0] if blockers else "stop_not_eligible",
                )
                return self._reject(
                    rejection_by_blocker.get(blocker, "stop_not_eligible")
                )
            return SafetyGuardResult(True, None, False)

        expected_strategy = {
            "explore": "default_then_local_refinement",
            "refine": "local_refinement",
            "retry": "recovery_configuration",
        }[action]
        if value.get("configuration_strategy") != expected_strategy:
            return self._reject("invalid_configuration_strategy")

        model = value.get("selected_model")
        family = value.get("selected_family")
        if not isinstance(model, str) or model not in MODEL_TO_FAMILY:
            return self._reject("model_not_in_catalog")
        if family != MODEL_TO_FAMILY[model]:
            return self._reject("model_family_mismatch")

        catalog = state.get("model_catalog", {})
        available = set(catalog.get("available_models", []) or []) if isinstance(catalog, Mapping) else set()
        unavailable = set(catalog.get("unavailable_models", []) or []) if isinstance(catalog, Mapping) else set()
        disabled = set(catalog.get("disabled_models", []) or []) if isinstance(catalog, Mapping) else set()
        failed = set(catalog.get("failed_models", []) or []) if isinstance(catalog, Mapping) else set()
        exhausted = set(state.get("technical_exhausted_models", []) or [])
        if model not in available or model in unavailable or model in disabled or model in failed or model in exhausted:
            return self._reject("candidate_unavailable")

        remaining = int((planner_input or {}).get("remaining_budget", 1))
        if remaining <= 0:
            return self._reject("budget_exhausted")
        if action == "retry" and int(state.get("technical_retry_count", 0)) >= int(state.get("max_technical_retries", 0)):
            return self._reject("retry_limit_exhausted")
        return SafetyGuardResult(True, None, False)


__all__ = ["SafetyGuardResult", "SearchSafetyGuard"]
