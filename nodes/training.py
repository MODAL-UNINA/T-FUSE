"""LangGraph node for model training.

"""

from __future__ import annotations

from typing import Any, Dict

from agents.training_agent import TrainingAgent
from core.candidate import (
    CandidateRecord,
    candidate_error_counts,
    cumulative_retry_count,
    increment_candidate_error,
)
from memory.memory_manager import MemoryManager
from utils.training_budget import count_valid_trainings
from utils.configuration_identity import log_exploration_status
from evaluation.search_trace import append_search_event, resolve_influence_mode
from utils.preprocessing_registry import canonical_failure_fingerprint
import time


class TrainingNode:
    """Node that wraps TrainingAgent for LangGraph integration.
    """

    def __init__(
        self, agent: TrainingAgent, memory_manager: MemoryManager | None = None
    ) -> None:
        self._agent = agent
        self._memory_manager = memory_manager

    @staticmethod
    def _record_failed_candidate(state: Dict[str, Any], err_msg: str) -> None:
        candidate = state.get("current_candidate_config", {})
        if not isinstance(candidate, dict) or not candidate.get("model_type"):
            return
        preprocessing = candidate.get("preprocessing", {"transformations": []})
        if isinstance(preprocessing, dict):
            preprocessing_value = preprocessing.get("transformations", [])
        else:
            preprocessing_value = preprocessing
        failed_cand_record = {
            "model": candidate.get("model_type"),
            "hyperparameters": candidate.get("hyperparameters", {}),
            "preprocessing": preprocessing_value,
            "mase": float("inf"),
            "rmse": float("inf"),
            "error": err_msg,
            "retry_reason": f"Training Node crash: {err_msg}",
        }
        failed_cand_record["failure_fingerprint"] = canonical_failure_fingerprint(
            str(candidate.get("model_type", "")),
            candidate.get("hyperparameters", {}),
            preprocessing_value,
            "execution_failure",
            err_msg,
        )
        state["last_candidate_failure"] = failed_cand_record

    @staticmethod
    def _pending_record(state: Dict[str, Any]) -> CandidateRecord:
        payload = state.get("candidate_record")
        if isinstance(payload, dict) and payload.get("status") == "pending":
            return CandidateRecord.from_dict(payload)
        return CandidateRecord.pending(
            state.get("current_candidate_config", {}),
            seed=int(state.get("candidate_seed", state.get("experiment_seed", 0))),
            occurrence=len(state.get("candidate_ledger", [])),
        )


    def _commit_terminal(
        self, state: Dict[str, Any], *, failed: bool
    ) -> CandidateRecord:
        pending = self._pending_record(state)
        if failed:
            terminal = pending.fail("execution_failure")
        else:
            latest = (
                state.get("performance_history", [])[-1]
                if state.get("performance_history")
                else {}
            )
            mase = float(latest.get("mase", float("inf")))
            terminal = pending.complete(
                {
                    key: latest.get(key)
                    for key in ("mase", "rmse", "mse", "mae", "mape", "r2")
                    if latest.get(key) is not None
                },
                finite_underperformance=mase > float(state.get("target_mase", 0.85)),
            )
        state["candidate_record"] = terminal.to_dict()
        if self._memory_manager is not None:
            self._memory_manager.commit_candidate(state, terminal)
            return terminal
        ledger = list(state.get("candidate_ledger", []))
        if not any(
            item.get("candidate_id") == terminal.candidate_id
            for item in ledger
            if isinstance(item, dict)
        ):
            ledger.append(terminal.to_dict())
        state["candidate_ledger"] = ledger
        return terminal

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training on preprocessed data and return partial state."""
        if state.get("technical_error"):
            return {}

        iteration = int(state.get("planner_call_count", 0))
        state["step"] = iteration
        training_started = time.perf_counter()

        try:
            self._agent.run(state)
        except Exception as exc:

            err_msg = f"Training failed: {str(exc)}"
            state["training_error"] = err_msg
            state["last_technical_error"] = err_msg
            self._record_failed_candidate(state, err_msg)
            terminal = self._commit_terminal(state, failed=True)
            counts = increment_candidate_error(state, "execution_failure")
            cumulative_count = cumulative_retry_count(state)
            append_search_event(
                state,
                event_type="candidate_completed",
                payload={
                    "request_id": (
                        state.get("candidate_request", {}).get("request_id")
                        if isinstance(state.get("candidate_request"), dict)
                        else None
                    ),
                    "influence_mode": resolve_influence_mode(state),
                    "candidate_id": terminal.candidate_id,
                    "candidate_signature": terminal.configuration_signature,
                    "candidate_seed": terminal.seed,
                    "model": terminal.model,
                    "status": "failed",
                    "failure_stage": "training_preflight",
                    "error_type": "execution_failure",
                    "runtime_seconds": time.perf_counter() - training_started,
                },
            )
            return {
                "training_error": err_msg,
                "technical_error": True,
                "non_retryable_candidate_failure": False,
                "last_technical_error": err_msg,
                "technical_retry_count": int(state.get("technical_retry_count", 0)) + 1,
                "cumulative_technical_retry_count": cumulative_count,
                "candidate_error_counts": counts,
                "memory": state.get("memory", {}),
                "performance_history": state.get("performance_history", []),
                "tested_models": state.get("tested_models", []),
                "candidate_record": state.get("candidate_record"),
                "candidate_ledger": state.get("candidate_ledger", []),
                "search_trace": state.get("search_trace", []),
                "history": [
                    {
                        "iteration": iteration,
                        "node": "training",
                        "status": "error",
                        "error": err_msg,
                    }
                ],
                "last_action": "training",
            }

        training_error = state.get("training_error")
        if training_error:
            state["last_technical_error"] = str(training_error)
        terminal = self._commit_terminal(state, failed=bool(training_error))
        if training_error:
            latest_perf: Dict[str, Any] = {}
            counts = increment_candidate_error(state, "execution_failure")
        else:
            latest_perf = (
                state.get("performance_history", [])[-1]
                if state.get("performance_history")
                else {}
            )
            counts = candidate_error_counts(state)
        previous_scores = [
            float(item["mase"])
            for item in state.get("performance_history", [])[:-1]
            if isinstance(item, dict)
            and item.get("mase") is not None
            and float(item["mase"]) < float("inf")
        ]
        latest_score = latest_perf.get("mase")
        improved = (
            not training_error
            and latest_score is not None
            and (
                not previous_scores
                or float(latest_score) < min(previous_scores)
            )
        )
        append_search_event(
            state,
            event_type="candidate_completed",
            payload={
                "request_id": (
                    state.get("candidate_request", {}).get("request_id")
                    if isinstance(state.get("candidate_request"), dict)
                    else None
                ),
                "influence_mode": resolve_influence_mode(state),
                "candidate_id": terminal.candidate_id,
                "candidate_signature": terminal.configuration_signature,
                "candidate_seed": terminal.seed,
                "model": terminal.model,
                "status": "failed" if training_error else "completed",
                "failure_stage": "training" if training_error else None,
                "validation_metrics": (
                    {}
                    if training_error
                    else {
                        key: latest_perf.get(key)
                        for key in ("mase", "rmse", "mae", "mse", "mape")
                    }
                ),
                "improved_incumbent": (
                    bool(improved) if not training_error else None
                ),
                "runtime_seconds": time.perf_counter() - training_started,
                "error_type": (
                    "execution_failure"
                    if training_error
                    else state.get("candidate_record", {}).get("error_type")
                ),
            },
        )
        status = "error" if training_error else "ok"
        valid_training_count = count_valid_trainings(
            state.get("performance_history", [])
        )
        state["valid_training_count"] = valid_training_count
        log_status = log_exploration_status(state)
        state.update(log_status)

        updates: Dict[str, Any] = {
            "last_action": "training",
            "tested_models": state.get("tested_models", []),
            "performance_history": state.get("performance_history", []),
            "best_model": state.get("best_model"),
            "best_score": float(
                state.get("best_model", {}).get("validation_score", float("inf"))
                if isinstance(state.get("best_model"), dict)
                else float("inf")
            ),
            "model_results": state.get("model_results", []),
            "target_mase_reached": state.get("target_mase_reached", False),
            "done": state.get("done", False),
            "stop_reason": state.get("stop_reason"),
            "stop_metadata": state.get("stop_metadata"),
            "valid_training_count": valid_training_count,
            "iteration": valid_training_count,
            "memory": state.get("memory", {}),
            "candidate_record": state.get("candidate_record"),
            "candidate_ledger": state.get("candidate_ledger", []),
            "non_retryable_candidate_failure": False,
            "candidate_error_counts": counts,
            "cumulative_technical_retry_count": cumulative_retry_count(state),
            "search_trace": state.get("search_trace", []),
            **log_status,
            "history": [
                {
                    "iteration": iteration,
                    "node": "training",
                    "action": "train_and_evaluate",
                    "model": terminal.model,
                    "rmse": latest_perf.get("rmse"),
                    "mase": latest_perf.get("mase"),
                    "status": status,
                }
            ],
        }

        if training_error:
            updates["training_error"] = training_error
            updates["technical_error"] = True
            updates["last_technical_error"] = training_error
            updates["technical_retry_count"] = int(state.get("technical_retry_count", 0)) + 1
            updates["cumulative_technical_retry_count"] = cumulative_retry_count(
                state
            )
        else:
            updates["training_error"] = None
            updates["technical_error"] = False
            updates["non_retryable_candidate_failure"] = False
            updates["last_technical_error"] = None

        return updates
