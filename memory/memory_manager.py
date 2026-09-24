"""State-facing memory adapter for LangGraph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from core.candidate import CandidateRecord
from core.memory import AgentMemory

@dataclass
class MemoryManager:
    """Synchronize AgentMemory with the shared graph state."""

    memory: AgentMemory

    def register_dataset_signature(self, state: Dict[str, Any]) -> None:
        series_id = str(state.get("series_id", ""))
        train_df = state.get("train_df")
        if not series_id or train_df is None:
            return
        self.memory.register_dataset_signature(series_id=series_id, train_df=train_df)
        self.sync_state_memory(state)

    def commit_candidate(
        self, state: Dict[str, Any], record: CandidateRecord
    ) -> bool:
        """Commit one terminal candidate exactly once."""
        if not record.is_terminal:
            raise ValueError("Only terminal CandidateRecord instances can be committed")
        if any(
            item.get("candidate_id") == record.candidate_id
            for item in self.memory.candidate_records
            if isinstance(item, dict)
        ):
            self.sync_state_memory(state)
            return False

        payload = record.to_dict()
        self.memory.candidate_records.append(payload)
        metrics = payload["validation_metrics"]
        self.memory.record_model_result(
            model_name=record.model,
            rmse=float(metrics.get("rmse", float("inf"))),
            mase=float(metrics.get("mase", float("inf"))),
            strategy=str(state.get("strategy", "exploration")),
            regime=str(state.get("context_regime", "normal")),
            hyperparameters=dict(payload["hyperparameters"]),
            preprocessing_signature=list(payload["preprocessing_signature"]),
            target_rmse=float(state.get("target_rmse", 1.5)),
            target_mase=float(state.get("target_mase", 0.85)),
            error_message=str(state.get("last_technical_error", "")),
            outcome_type=record.error_type or record.status,
        )
        state["candidate_record"] = payload
        state["candidate_ledger"] = list(self.memory.candidate_records)
        self.sync_state_memory(state)
        return True

    def sync_state_memory(self, state: Dict[str, Any]) -> None:
        from utils.configuration_identity import deduplicate_failed_combinations
        state_mem = state.get("memory", {})
        if isinstance(state_mem, dict):
            for m in state_mem.get("failed_models", []):
                if m not in self.memory.failed_models:
                    self.memory.failed_models.append(m)
            
            for combo in deduplicate_failed_combinations(state_mem.get("failed_combinations", [])):
                if combo not in self.memory.failed_combinations:
                    self.memory.failed_combinations.append(combo)
            
            for pat in state_mem.get("successful_patterns", []):
                if pat not in self.memory.successful_patterns:
                    self.memory.successful_patterns.append(pat)

            for sid, sig in state_mem.get("dataset_signatures", {}).items():
                self.memory.dataset_signatures[sid] = sig
            for record in state_mem.get("candidate_records", []):
                if (
                    isinstance(record, dict)
                    and not any(
                        existing.get("candidate_id") == record.get("candidate_id")
                        for existing in self.memory.candidate_records
                    )
                ):
                    self.memory.candidate_records.append(record)

        self.memory.failed_combinations = deduplicate_failed_combinations(self.memory.failed_combinations)
        state["memory"] = {
            "failed_models": list(self.memory.failed_models),
            "failed_combinations": list(self.memory.failed_combinations),
            "successful_patterns": list(self.memory.successful_patterns),
            "dataset_signatures": dict(self.memory.dataset_signatures),
            "candidate_records": list(self.memory.candidate_records),
        }
