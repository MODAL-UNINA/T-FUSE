"""Semantic node and evidence-integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import copy

import pandas as pd

from agents.semantic_agent import SemanticAgent
from memory.memory_manager import MemoryManager
from soft_computing import AnalyticalFamilyCompatibilityMapper
from soft_computing.evidence_fusion import fuse_evidence
from utils.semantic_agent_input import build_semantic_agent_state


@dataclass
class SemanticNode:
    """Run semantic analysis, then combine it with analytical evidence.

    """

    agent: SemanticAgent
    memory_manager: MemoryManager
    mode: str = "fuzzy"

    @staticmethod
    def _validate_training_document_boundary(
        semantic_train_df: pd.DataFrame, train_df: pd.DataFrame
    ) -> None:
        """Reject non-canonical or post-training semantic evidence."""
        if "date" not in semantic_train_df:
            raise RuntimeError(
                "Semantic timestamp policy violation: canonical date is required."
            )
        document_dates = pd.to_datetime(
            semantic_train_df["date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        invalid_count = int(len(document_dates) - document_dates.notna().sum())
        if invalid_count:
            raise RuntimeError(
                "Semantic timestamp policy violation: "
                f"{invalid_count} training documents have invalid dates."
            )
        if not isinstance(train_df, pd.DataFrame) or train_df.empty or "date" not in train_df:
            raise RuntimeError(
                "The numerical training boundary is required before semantic analysis."
            )
        train_end = pd.to_datetime(
            train_df["date"], errors="coerce", utc=True
        ).max()
        if pd.isna(train_end):
            raise RuntimeError("The numerical training boundary is not parseable.")
        train_end = train_end.tz_localize(None)
        later_count = int((document_dates > train_end).sum())
        if later_count:
            raise RuntimeError(
                "Semantic leakage guard rejected "
                f"{later_count} documents after the training boundary."
            )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["step"] = int(state.get("planner_call_count", 0))
        mode = str(self.mode).strip().lower()
        if mode != "fuzzy":
            raise ValueError("T-FUSE semantic node accepts only fuzzy influence")
        semantic_train_df = state.get("semantic_docs_train_df")
        if not isinstance(semantic_train_df, pd.DataFrame) or semantic_train_df.empty:
            raise RuntimeError("T-FUSE requires non-empty training-time semantic documents")
        self._validate_training_document_boundary(
            semantic_train_df, state.get("train_df")
        )

        semantic_state = build_semantic_agent_state(state)
        self.agent.run(semantic_state)
        semantic_result = semantic_state
        semantic_context = copy.deepcopy(semantic_result["semantic_context"])
        state["semantic_raw_signals"] = copy.deepcopy(semantic_result["semantic_raw_signals"])
        state["semantic_operational_signals"] = copy.deepcopy(
            semantic_result["semantic_operational_signals"]
        )
        state["semantic_excluded_signals"] = copy.deepcopy(
            semantic_result["semantic_excluded_signals"]
        )
        state["semantic_signals"] = copy.deepcopy(semantic_result["semantic_signals"])

        mapping = AnalyticalFamilyCompatibilityMapper().map(
            {
                **dict(state.get("data_summary", {}) or {}),
                "forecast_horizon": state.get("forecast_horizon"),
            }
        )
        evidence = fuse_evidence(mapping, semantic_context, mode=mode)
        evidence["analytical_mapping"] = mapping

        state["semantic_context"] = semantic_context
        state["semantic_completed"] = True
        state["semantic_agent_uses_analytical_output"] = False
        state.setdefault("diagnostics", {})["evidence_fusion"] = evidence
        state["planner_guidance"] = evidence

        inferred_domain = semantic_context.get("inferred_domain", "unknown")
        state["inferred_domain"] = inferred_domain

        return {
            "semantic_signals": state.get("semantic_signals", []),
            "semantic_raw_signals": state.get("semantic_raw_signals", []),
            "semantic_operational_signals": state.get("semantic_operational_signals", []),
            "semantic_excluded_signals": state.get("semantic_excluded_signals", []),
            "semantic_context": semantic_context,
            "semantic_completed": True,
            "inferred_domain": inferred_domain,
            "last_action": "semantic",
            "semantic_agent_uses_analytical_output": False,
            "diagnostics": state.get("diagnostics", {}),
            "planner_guidance": evidence,
        }
