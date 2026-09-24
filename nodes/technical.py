"""Technical worker node.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict

from agents.technical_agent import TechnicalAgent
from core.candidate import (
    CandidateRecord,
    candidate_error_counts,
    cumulative_retry_count,
    increment_candidate_error,
)
from core.candidate_catalog import (
    DEFAULT_CANDIDATE_CATALOG,
    derive_candidate_seed,
    set_deterministic_seed,
)
from utils.configuration_identity import canonical_configuration_signature
from utils.preprocessing_registry import (
    FrameworkConfigurationError,
    is_non_retryable_preprocessing_error,
)
from evaluation.search_trace import append_search_event, resolve_influence_mode
from memory.memory_manager import MemoryManager
from search.safety import SearchSafetyGuard


@dataclass
class TechnicalNode:
    """Execute technical model/configuration selection and return partial updates."""

    agent: TechnicalAgent
    memory_manager: MemoryManager
    force_guidance: str | None = None
    count_iteration: bool = True
    safety_guard: SearchSafetyGuard = field(default_factory=SearchSafetyGuard)

    @staticmethod
    def _reset_candidate_transaction(state: Dict[str, Any]) -> None:
        """Remove every candidate-scoped artifact before a new selection."""
        for key in (
            "current_candidate_config",
            "candidate_record",
            "model_rationale",
            "model_confidence",
            "training_error",
            "last_candidate_failure",
            "preprocessed_train_df",
            "preprocessed_val_df",
            "preprocessing_log",
            "preprocessing_candidate_id",
            "candidate_seed",
            "determinism_audit",
        ):
            state.pop(key, None)


    @staticmethod
    def _requested_model(state: Dict[str, Any]) -> str | None:
        request = state.get("candidate_request", {})
        if isinstance(request, dict):
            model = request.get("model") or request.get("model_type")
            if model:
                return str(model)
        decision = state.get("planner_decision", {})
        if isinstance(decision, dict) and decision.get("selected_model"):
            return str(decision.get("selected_model"))
        return None

    @classmethod
    def _mark_model_exhausted(
        cls, state: Dict[str, Any], model: str | None = None
    ) -> list[str]:
        model = model or cls._requested_model(state)
        exhausted = list(state.get("technical_exhausted_models", []) or [])
        if model and model not in exhausted:
            exhausted.append(model)
        state["technical_exhausted_models"] = exhausted
        return exhausted

    @staticmethod
    def _canonical_hyperparameters(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        try:
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _catalog_hyperparameter_exhaustion(
        cls, state: Dict[str, Any], model: str | None
    ) -> dict[str, Any]:
        """Return deterministic exhaustion only for a genuinely finite model space.

        Ranged constraints are intentionally non-enumerable and therefore cannot
        be marked exhausted by a Cartesian-product check.

        Preprocessing is deliberately not used to multiply the catalog here: the
        runtime search contract treats catalog hyperparameter combinations as the
        atomic refinement choices for a model, while preprocessing is attached to
        the chosen candidate.  This prevents repeated LLM calls after the finite
        model catalog has been consumed.
        """
        if not model:
            return {"exhausted": False, "model": model, "total": 0, "attempted": 0}
        schema = DEFAULT_CANDIDATE_CATALOG.finite_values(str(model))
        if not isinstance(schema, dict) or not schema:
            return {
                "exhausted": False,
                "finite": False,
                "model": model,
                "total": 0,
                "attempted": 0,
            }

        expected_keys = set(schema)
        total = math.prod(len(values) for values in schema.values())
        attempted: set[str] = set()

        sources: list[Any] = []
        sources.extend(state.get("performance_history", []) or [])
        sources.extend(state.get("candidate_ledger", []) or [])
        sources.extend(state.get("rejected_technical_candidates", []) or [])
        memory = state.get("memory", {})
        if isinstance(memory, dict):
            sources.extend(memory.get("failed_combinations", []) or [])

        allowed_by_key = {
            key: {
                json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                for item in values
            }
            for key, values in schema.items()
        }
        for record in sources:
            if not isinstance(record, dict):
                continue
            record_model = record.get("model") or record.get("model_type")
            if str(record_model or "") != str(model):
                continue
            hp = record.get("hyperparameters", {})
            if not isinstance(hp, dict) or set(hp) != expected_keys:
                continue
            valid = True
            for key in expected_keys:
                try:
                    encoded = json.dumps(
                        hp[key], sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    )
                except (TypeError, ValueError):
                    valid = False
                    break
                if encoded not in allowed_by_key[key]:
                    valid = False
                    break
            if valid:
                canonical = cls._canonical_hyperparameters(hp)
                if canonical:
                    attempted.add(canonical)

        attempted_count = len(attempted)
        return {
            "exhausted": total > 0 and attempted_count >= total,
            "finite": True,
            "model": str(model),
            "total": int(total),
            "attempted": int(attempted_count),
            "remaining": max(0, int(total) - int(attempted_count)),
        }

    @staticmethod
    def _guidance_to_mode(guidance: str | None) -> str:
        if guidance is None:
            return "explore"
        normalized = str(guidance).strip().lower()
        if normalized == "retry":
            return "retry"
        if normalized in {"refine", "refine_model"}:
            return "refine"
        if normalized in {
            "explore",
            "explore_new_models",
            "explore_new_model",
            "run_technical",
        }:
            return "explore"
        return "explore"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = "explore"
        planner_signals = state.get("planner_signals", {})
        if not isinstance(planner_signals, dict):
            planner_signals = {}
        if self.force_guidance is not None:
            mode = self._guidance_to_mode(self.force_guidance)
        else:
            mode = self._guidance_to_mode(planner_signals.get("technical_mode"))
            if mode not in {"explore", "refine", "retry"}:
                mode = self._guidance_to_mode(state.get("technical_mode", "explore"))

        iteration = int(state.get("planner_call_count", 0))
        state["step"] = iteration
        state["technical_mode"] = mode

        retry_count_before = int(state.get("technical_retry_count", 0))
        previous_error = state.get("last_technical_error")

        is_retry = bool(state.get("technical_error", False)) or retry_count_before > 0
        node_label = "retry_technical" if is_retry else "technical"

        if bool(state.get("technical_error", False)) and previous_error:
            planner_signals["technical_retry"] = True
            planner_signals["technical_retry_count"] = retry_count_before
            planner_signals["last_technical_error"] = str(previous_error)
        else:
            planner_signals["technical_retry"] = False

        state["planner_signals"] = planner_signals

        # Deterministic preflight applies only to fully finite categorical spaces.
        requested_model = self._requested_model(state)
        catalog_exhaustion = self._catalog_hyperparameter_exhaustion(
            state, requested_model
        )
        if catalog_exhaustion.get("exhausted"):
            exhausted_models = self._mark_model_exhausted(state, requested_model)
            append_search_event(
                state,
                event_type="technical_scope_exhausted",
                payload={
                    "request_id": (
                        state.get("candidate_request", {}).get("request_id")
                        if isinstance(state.get("candidate_request"), dict)
                        else None
                    ),
                    "influence_mode": resolve_influence_mode(state),
                    "model": requested_model,
                    "status": "exhausted",
                    "reason_code": "candidate_catalog_exhausted",
                    "catalog_exhaustion": catalog_exhaustion,
                },
            )
            return {
                "technical_status": "NO_VALID_NEW_CONFIGURATION",
                "request_planner_replan": True,
                "technical_error": False,
                "last_technical_error": None,
                "technical_retry_count": 0,
                "technical_exhausted_models": exhausted_models,
                "technical_escalation": {
                    "status": "NO_VALID_NEW_CONFIGURATION",
                    "reason": "CandidateCatalog hyperparameter space exhausted before TechnicalAgent call.",
                    "request_planner_replan": True,
                    "exhausted_scope": "same_model",
                    "exhausted_model": requested_model,
                    "source": "candidate_catalog_preflight",
                    "catalog_exhaustion": catalog_exhaustion,
                },
                "cumulative_technical_retry_count": cumulative_retry_count(state),
                "candidate_error_counts": candidate_error_counts(state),
                "technical_mode": mode,
                "current_candidate_config": None,
                "planner_signals": planner_signals,
                "history": [{
                    "iteration": iteration,
                    "node": "technical_preflight",
                    "status": "catalog_exhausted",
                    "model": requested_model,
                    "catalog_exhaustion": catalog_exhaustion,
                }],
                "last_action": "technical_preflight",
                "memory": state.get("memory", {}),
                "search_trace": state.get("search_trace", []),
            }

        self._reset_candidate_transaction(state)
        technical_state: Dict[str, Any] = {}
        rejection_count_before = len(
            state.get("rejected_technical_candidates", []) or []
        )
        try:
 
            technical_state = dict(state)
            for forbidden in (
                "semantic_context", "semantic_signals",
                "semantic_raw_signals", "semantic_operational_signals",
                "diagnostics",
            ):
                technical_state.pop(forbidden, None)

            technical_state.pop("inferred_domain", None)
            technical_state["analysis_history"] = [
                item for item in state.get("analysis_history", [])
                if isinstance(item, dict) and item.get("task") != "semantic"
            ]
            technical_state["planner_guidance"] = state.get("planner_instruction")
            technical_state["technical_agent_uses_fusion_output"] = False
            agent_result = self.agent.run(technical_state)
            for key in ("current_candidate_config", "model_rationale", "model_confidence", "memory", "rejected_technical_candidates", "technical_status", "technical_escalation", "request_planner_replan"):
                if key in technical_state:
                    state[key] = technical_state[key]


            if (
                isinstance(agent_result, dict)
                and agent_result.get("technical_status")
                == "NO_VALID_NEW_CONFIGURATION"
            ):
                escalation = agent_result.get("technical_escalation", {})
                exhausted_model = self._requested_model(state)
                exhausted_models = self._mark_model_exhausted(
                    state, exhausted_model
                )
                return {
                    "technical_status": "NO_VALID_NEW_CONFIGURATION",
                    "request_planner_replan": True,
                    "technical_escalation": escalation,

                    "technical_error": False,
                    "last_technical_error": previous_error,
                    "technical_retry_count": 0,
                    "technical_exhausted_models": exhausted_models,
                    "cumulative_technical_retry_count": cumulative_retry_count(
                        state
                    ),
                    "candidate_error_counts": candidate_error_counts(state),
                    "technical_mode": mode,
                    "current_candidate_config": None,
                    "planner_signals": planner_signals,
                    "history": [{
                        "iteration": iteration,
                        "node": node_label,
                        "status": "no_valid_new_configuration",
                        "technical_mode": mode,
                        "technical_escalation": escalation,
                        "exhausted_model": exhausted_model,
                    }],
                    "last_action": node_label,
                    "memory": state.get("memory", {}),
                }
            if not (
                isinstance(state.get("current_candidate_config"), dict)
                and state["current_candidate_config"].get("model_type")
            ):
                raise RuntimeError(
                    "TechnicalAgent returned without a concrete candidate"
                )

        except FrameworkConfigurationError:
            raise
        except Exception as exc:
            err_msg = str(exc)
            non_retryable = is_non_retryable_preprocessing_error(exc)
            rejected = technical_state.get("rejected_technical_candidates", [])
            if isinstance(rejected, list):
                state["rejected_technical_candidates"] = list(rejected)
            new_rejections = (
                rejected[rejection_count_before:]
                if isinstance(rejected, list)
                else []
            )
            latest_rejection = new_rejections[-1] if new_rejections else None
            if isinstance(latest_rejection, dict):
                rejected_candidate = {
                    "model_type": latest_rejection.get("model_type", ""),
                    "hyperparameters": latest_rejection.get(
                        "hyperparameters", {}
                    ),
                    "preprocessing": {
                        "transformations": latest_rejection.get(
                            "preprocessing", []
                        )
                    },
                }
                pending = CandidateRecord.pending(
                    rejected_candidate,
                    seed=int(
                        state.get(
                            "candidate_seed", state.get("experiment_seed", 0)
                        )
                    ),
                    occurrence=len(state.get("candidate_ledger", [])),
                )
                error_type = (
                    "duplicate_candidate"
                    if latest_rejection.get("rejection_code")
                    == "duplicate_candidate"
                    else "invalid_configuration"
                )
                self.memory_manager.commit_candidate(
                    state, pending.fail(error_type)
                )
            else:
                error_type = "technical_generation_failure"
                self.memory_manager.sync_state_memory(state)

            counts = increment_candidate_error(state, error_type)
            retry_count = int(state.get("technical_retry_count", 0)) + 1
            cumulative_count = cumulative_retry_count(state)
            rejected_record = state.get("candidate_record", {})
            append_search_event(
                state,
                event_type="candidate_rejected",
                payload={
                    "request_id": (
                        state.get("candidate_request", {}).get("request_id")
                        if isinstance(state.get("candidate_request"), dict)
                        else None
                    ),
                    "influence_mode": resolve_influence_mode(state),
                    "candidate_id": (
                        rejected_record.get("candidate_id")
                        if isinstance(rejected_record, dict)
                        else None
                    ),
                    "candidate_signature": (
                        rejected_record.get("configuration_signature")
                        if isinstance(rejected_record, dict)
                        else None
                    ),
                    "model": (
                        latest_rejection.get("model_type")
                        if isinstance(latest_rejection, dict)
                        else None
                    ),
                    "status": "rejected",
                    "rejection_code": (
                        latest_rejection.get("rejection_code")
                        if isinstance(latest_rejection, dict)
                        else "technical_generation_failure"
                    ),
                    "error_type": error_type,
                    "error_message": err_msg,
                },
            )


            state.pop("current_candidate_config", None)
            state.pop("model_rationale", None)
            state.pop("model_confidence", None)

            history_update = {
                "iteration": iteration,
                "node": node_label,
                "status": "error",
                "error": err_msg,
                "technical_mode": mode,
                "technical_retry_count": retry_count,
                "cumulative_technical_retry_count": cumulative_count,
                "candidate_error_type": error_type,
                "candidate_error_counts": counts,
                "rejected_technical_candidates": state.get("rejected_technical_candidates", []),
            }

            max_retries = int(state.get("max_technical_retries", 10))
            is_exhausted = (
                not non_retryable and retry_count >= max_retries
            )

            result = {
                "technical_error": not non_retryable,
                "non_retryable_candidate_failure": non_retryable,
                "request_planner_replan": non_retryable,
                "last_technical_error": err_msg,
                "last_technical_error_type": type(exc).__name__,
                "technical_retry_count": retry_count,
                "cumulative_technical_retry_count": cumulative_count,
                "candidate_error_counts": counts,
                "rejected_technical_candidates": state.get("rejected_technical_candidates", []),
                "technical_mode": mode,
                "planner_signals": planner_signals,
                "current_candidate_config": None,
                "model_rationale": None,
                "model_confidence": None,
                "training_error": None,
                "preprocessed_train_df": None,
                "preprocessed_val_df": None,
                "preprocessing_log": {},
                "preprocessing_candidate_id": None,
                "history": [history_update],
                "stop_reason": None,
                "last_action": node_label,
                "memory": state.get("memory", {}),
            }

            if is_exhausted:
                exhausted_model = (
                    latest_rejection.get("model_type")
                    if isinstance(latest_rejection, dict)
                    else self._requested_model(state)
                )
                exhausted_models = self._mark_model_exhausted(
                    state, exhausted_model
                )

                result.update({
                    "technical_status": "NO_VALID_NEW_CONFIGURATION",
                    "request_planner_replan": True,
                    "technical_error": False,
                    "technical_retry_count": 0,
                    "technical_exhausted_models": exhausted_models,
                    "technical_escalation": {
                        "status": "NO_VALID_NEW_CONFIGURATION",
                        "reason": (
                            f"Max technical retries ({retry_count}) reached due "
                            "to repeated invalid/failed configurations."
                        ),
                        "request_planner_replan": True,
                        "exhausted_scope": "same_model",
                        "exhausted_model": exhausted_model,
                        "step": iteration,
                        "source": "technical_node",
                    },
                })

            return result

        candidate = state.get("current_candidate_config", {})
        if isinstance(candidate, dict) and candidate.get("model_type"):
            configuration_guard = self.safety_guard.validate_configuration(
                state.get("planner_decision", {}), candidate, state
            )
            if not configuration_guard.accepted:
                rejection_code = str(configuration_guard.reason or "invalid_configuration")
                if rejection_code == "candidate_already_evaluated":
                    # Legacy alias normalized at the node boundary.
                    rejection_code = "duplicate_candidate"
                error_type = (
                    "duplicate_candidate"
                    if rejection_code == "duplicate_candidate"
                    else "invalid_configuration"
                )
                configuration_signature = canonical_configuration_signature(
                    candidate["model_type"],
                    candidate.get("hyperparameters", {}),
                    candidate.get("preprocessing", {}),
                )
                rejected = list(state.get("rejected_technical_candidates", []) or [])
                rejected.append({
                    "rejected_candidate_id": f"rejected-{len(rejected) + 1}",
                    "step": iteration,
                    "attempt": int(state.get("technical_retry_count", 0)) + 1,
                    "request_id": (
                        state.get("candidate_request", {}).get("request_id")
                        if isinstance(state.get("candidate_request"), dict)
                        else None
                    ),
                    "model_type": candidate.get("model_type"),
                    "hyperparameters": dict(candidate.get("hyperparameters", {}) or {}),
                    "preprocessing": (
                        list(candidate.get("preprocessing", {}).get("transformations", []) or [])
                        if isinstance(candidate.get("preprocessing"), dict)
                        else list(candidate.get("preprocessing", []) or [])
                    ),
                    "configuration_signature": configuration_signature,
                    "rejection_stage": "search_safety_guard",
                    "rejection_code": rejection_code,
                    "error_type": error_type,
                    "reason": f"Technical configuration rejected by SearchSafetyGuard: {rejection_code}",
                    "was_sent_to_training": False,
                })
                state["rejected_technical_candidates"] = rejected
                append_search_event(
                    state,
                    event_type="candidate_rejected",
                    payload={
                        "request_id": (
                            state.get("candidate_request", {}).get("request_id")
                            if isinstance(state.get("candidate_request"), dict)
                            else None
                        ),
                        "influence_mode": resolve_influence_mode(state),
                        "candidate_signature": configuration_signature,
                        "model": candidate.get("model_type"),
                        "status": "rejected",
                        "rejection_code": rejection_code,
                        "error_type": error_type,
                    },
                )
                retry_count = int(state.get("technical_retry_count", 0)) + 1
                max_retries = max(1, int(state.get("max_technical_retries", 1)))
                exhausted = retry_count >= max_retries
                counts = increment_candidate_error(state, error_type)
                exhausted_models = (
                    self._mark_model_exhausted(
                        state, candidate.get("model_type")
                    )
                    if exhausted
                    else list(state.get("technical_exhausted_models", []) or [])
                )
                error_message = (
                    "Technical configuration rejected by SearchSafetyGuard: "
                    f"{rejection_code}"
                )
                return {
                    "technical_status": (
                        "NO_VALID_NEW_CONFIGURATION"
                        if exhausted
                        else "CONFIGURATION_REJECTED"
                    ),
                    "request_planner_replan": True,
                    "technical_error": not exhausted,
                    "last_technical_error": error_message,
                    "technical_retry_count": 0 if exhausted else retry_count,
                    "technical_exhausted_models": exhausted_models,
                    "cumulative_technical_retry_count": cumulative_retry_count(
                        state
                    ),
                    "candidate_error_counts": counts,
                    "current_candidate_config": None,
                    "safety_guard_result": configuration_guard.to_dict(),
                    "technical_mode": mode,
                    "planner_signals": planner_signals,
                    "search_trace": state.get("search_trace", []),
                    "history": [{
                        "iteration": iteration,
                        "node": node_label,
                        "status": "configuration_rejected",
                        "reason": configuration_guard.reason,
                        "technical_retry_count": retry_count,
                        "model_exhausted": exhausted,
                    }],
                }
            configuration_signature = canonical_configuration_signature(
                candidate["model_type"],
                candidate.get("hyperparameters", {}),
                candidate.get("preprocessing", {}),
            )
            state["candidate_seed"] = derive_candidate_seed(
                str(state.get("dataset_fingerprint", "")),
                int(state.get("experiment_seed", 0)),
                configuration_signature,
            )
            state["determinism_audit"] = set_deterministic_seed(
                state["candidate_seed"]
            )
            state["candidate_record"] = CandidateRecord.pending(
                candidate,
                seed=int(
                    state.get("candidate_seed", state.get("experiment_seed", 0))
                ),
                occurrence=len(state.get("candidate_ledger", [])),
            ).to_dict()
        history_update = {
            "iteration": iteration,
            "node": node_label,
            "action": "model_selected",
            "selected_model": candidate.get("model_type", "") if isinstance(candidate, dict) else "",
            "technical_mode": mode,
            "status": "ok",
        }

        return {
            "current_candidate_config": state.get("current_candidate_config", {}),
            "model_rationale": state.get("model_rationale", ""),
            "model_confidence": state.get("model_confidence", 0.5),
            "memory": state.get("memory", {}),
            "technical_error": False,
            "non_retryable_candidate_failure": False,
            "last_technical_error": None,
            "last_technical_error_type": None,
            "technical_status": "CANDIDATE_SELECTED",
            "request_planner_replan": False,
            "technical_escalation": None,
            "technical_retry_count": 0,
            "technical_exhausted_models": list(
                state.get("technical_exhausted_models", []) or []
            ),
            "cumulative_technical_retry_count": cumulative_retry_count(state),
            "candidate_error_counts": candidate_error_counts(state),
            "rejected_technical_candidates": state.get("rejected_technical_candidates", []),
            "technical_mode": mode,
            "planner_signals": planner_signals,
            "history": [history_update],
            "last_action": node_label,
            "analysis_history": state.get("analysis_history", []),
            "candidate_record": state.get("candidate_record"),
            "candidate_seed": state.get("candidate_seed"),
            "determinism_audit": state.get("determinism_audit", {}),
            "training_error": None,
            "preprocessed_train_df": None,
            "preprocessed_val_df": None,
            "preprocessing_log": {},
            "preprocessing_candidate_id": None,
            "last_candidate_failure": None,
            "search_trace": state.get("search_trace", []),
        }
