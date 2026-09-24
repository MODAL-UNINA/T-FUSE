"""Preprocessing worker node.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

from agents.preprocessing_agent import PreprocessingAgent
from core.candidate import (
    CandidateRecord,
    candidate_error_counts,
    cumulative_retry_count,
    increment_candidate_error,
)
from evaluation.search_trace import append_search_event, resolve_influence_mode
from memory.memory_manager import MemoryManager
from utils.preprocessing_registry import (
    FrameworkConfigurationError,
    canonical_failure_fingerprint,
)


@dataclass
class PreprocessingNode:
    """Execute the preprocessing agent and return partial state updates."""

    agent: PreprocessingAgent
    memory_manager: MemoryManager

    @staticmethod
    def _record_preprocessing_failure(state: Dict[str, Any], err_msg: str) -> None:
        candidate = state.get("current_candidate_config", {})
        if not isinstance(candidate, dict) or not candidate.get("model_type"):
            return

        failed_cand_record = {
            "model": candidate.get("model_type"),
            "hyperparameters": candidate.get("hyperparameters", {}),
            "preprocessing": candidate.get("preprocessing", {"transformations": []}),
            "mase": float("inf"),
            "rmse": float("inf"),
            "error": err_msg,
            "retry_reason": f"Preprocessing crash: {err_msg}",
        }
        failed_cand_record["failure_fingerprint"] = canonical_failure_fingerprint(
            str(candidate.get("model_type", "")),
            candidate.get("hyperparameters", {}),
            candidate.get("preprocessing", {}),
            "execution_failure",
            err_msg,
        )
        state["last_candidate_failure"] = failed_cand_record

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("technical_error"):
            return {}

        state["step"] = int(state.get("planner_call_count", 0))
        started = time.perf_counter()

        # Fail closed against stale artifacts even if a caller bypassed the
        # TechnicalNode's transaction reset.
        state["processed_train_df"] = None
        state["replace_outliers_log"] = {}
        state["preprocessed_train_df"] = None
        state["preprocessed_val_df"] = None
        state["preprocessing_log"] = {}
        state["preprocessing_candidate_id"] = None

        try:
            self.agent.run(state)
            payload = state.get("candidate_record")
            if (
                not isinstance(payload, dict)
                or payload.get("status") != "pending"
                or not payload.get("candidate_id")
            ):
                raise ValueError(
                    "Preprocessing requires a pending candidate transaction"
                )
            log = state.get("preprocessing_log")
            if not isinstance(log, dict):
                raise ValueError("PreprocessingAgent did not produce a log")
            log = dict(log)
            log["candidate_id"] = payload["candidate_id"]
            log["configuration_signature"] = payload[
                "configuration_signature"
            ]
            log["candidate_seed"] = payload["seed"]
            state["preprocessing_log"] = log
            state["preprocessing_candidate_id"] = payload["candidate_id"]
        except FrameworkConfigurationError:
            raise
        except Exception as exc:
            err_msg = f"Preprocessing failed: {str(exc)}"
            state["last_technical_error"] = err_msg
            self._record_preprocessing_failure(state, err_msg)
            payload = state.get("candidate_record")
            pending = (
                CandidateRecord.from_dict(payload)
                if isinstance(payload, dict) and payload.get("status") == "pending"
                else CandidateRecord.pending(
                    state.get("current_candidate_config", {}),
                    seed=int(
                        state.get(
                            "candidate_seed", state.get("experiment_seed", 0)
                        )
                    ),
                    occurrence=len(state.get("candidate_ledger", [])),
                )
            )
            self.memory_manager.commit_candidate(
                state, pending.fail("execution_failure")
            )
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
                    "candidate_id": state.get("candidate_record", {}).get(
                        "candidate_id"
                    ),
                    "candidate_signature": state.get(
                        "candidate_record", {}
                    ).get("configuration_signature"),
                    "model": state.get("candidate_record", {}).get("model"),
                    "status": "failed",
                    "failure_stage": "preprocessing",
                    "error_type": "execution_failure",
                    "runtime_seconds": time.perf_counter() - started,
                },
            )

            return {
                "technical_error": True,
                "non_retryable_candidate_failure": False,
                "last_technical_error": err_msg,
                "technical_retry_count": int(state.get("technical_retry_count", 0)) + 1,
                "cumulative_technical_retry_count": cumulative_count,
                "candidate_error_counts": counts,
                "processed_train_df": None,
                "replace_outliers_log": {},
                "preprocessed_train_df": None,
                "preprocessed_val_df": None,
                "preprocessing_log": {},
                "preprocessing_candidate_id": None,
                "history": [
                    {
                        "iteration": state["step"],
                        "node": "preprocessing",
                        "status": "error",
                        "error": err_msg,
                    }
                ],
                "last_action": "preprocessing",
                "memory": state.get("memory", {}),
                "candidate_record": state.get("candidate_record"),
                "candidate_ledger": state.get("candidate_ledger", []),
                "search_trace": state.get("search_trace", []),
            }

        return {
            "processed_train_df": state.get("processed_train_df"),
            "replace_outliers_log": state.get(
                "replace_outliers_log", {}
            ),
            "preprocessed_train_df": state.get("preprocessed_train_df"),
            "preprocessed_val_df": state.get("preprocessed_val_df"),
            "preprocessing_log": state.get("preprocessing_log", {}),
            "preprocessing_candidate_id": state.get(
                "preprocessing_candidate_id"
            ),
            "candidate_record": state.get("candidate_record"),
            "analysis_history": state.get("analysis_history", []),
            "memory": state.get("memory", {}),
            "technical_error": False,
            "non_retryable_candidate_failure": False,
            "last_technical_error": None,
            "training_error": None,
            "candidate_error_counts": candidate_error_counts(state),
            "cumulative_technical_retry_count": cumulative_retry_count(state),
            "last_action": "preprocessing",
        }
