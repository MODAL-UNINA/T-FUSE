"""Analytical worker node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from agents.analytical_agent import AnalyticalAgent
from core.framework_settings import SETTINGS
from utils.data_loader import select_recent_balanced_semantic_docs
from memory.memory_manager import MemoryManager


@dataclass
class AnalyticalNode:
    """Execute the analytical agent and return partial state updates."""

    agent: AnalyticalAgent
    memory_manager: MemoryManager

    @staticmethod
    def _split_semantic_documents(state: Dict[str, Any]) -> None:
        """Create timestamp-only semantic train/validation partitions.

        This belongs to the orchestration boundary rather than AnalyticalAgent:
        the analytical worker remains blind to document contents, while the
        node may use the already-established numeric split dates to partition
        the independent semantic source. Every document must already expose a
        valid canonical assignment timestamp.
        """
        if not bool(state.get("semantic_enabled", False)):
            return

        docs = state.get("semantic_docs_df")
        train_df = state.get("train_df")
        val_df = state.get("val_df")
        if not isinstance(docs, pd.DataFrame) or docs.empty:
            raise RuntimeError(
                "Semantic mode requires a non-empty semantic_docs_df before "
                "the analytical/semantic orchestration boundary."
            )
        if (
            not isinstance(train_df, pd.DataFrame)
            or train_df.empty
            or "date" not in train_df
            or not isinstance(val_df, pd.DataFrame)
            or val_df.empty
            or "date" not in val_df
        ):
            raise RuntimeError(
                "Semantic documents cannot be partitioned before the numeric "
                "train/validation split is available."
            )

        documents = docs.copy(deep=True)
        if "date" not in documents:
            raise RuntimeError(
                "Semantic timestamp policy violation: canonical date is required."
            )
        parsed_dates = pd.to_datetime(
            documents["date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        invalid_document_dates = int(len(parsed_dates) - parsed_dates.notna().sum())
        if invalid_document_dates:
            raise RuntimeError(
                "Semantic timestamp policy violation: "
                f"{invalid_document_dates} document rows have invalid dates."
            )
        train_end = pd.to_datetime(train_df["date"], errors="coerce", utc=True).max()
        val_end = pd.to_datetime(val_df["date"], errors="coerce", utc=True).max()
        if pd.isna(train_end) or pd.isna(val_end):
            raise RuntimeError("Numeric train/validation dates are not parseable.")
        train_end = train_end.tz_localize(None)
        val_end = val_end.tz_localize(None)

        # Validation documents are retained for audit only and are never
        # exposed to SemanticAgent. Later documents are excluded from search.
        train_mask = parsed_dates <= train_end
        validation_mask = (parsed_dates > train_end) & (parsed_dates <= val_end)
        after_validation = parsed_dates > val_end

        semantic_train_available = documents.loc[train_mask].reset_index(drop=True)
        semantic_val_df = documents.loc[validation_mask].reset_index(drop=True)
        semantic_train_df = select_recent_balanced_semantic_docs(
            semantic_train_available, int(state.get("semantic_max_docs", 0) or 0)
        )
        if semantic_train_df.empty:
            raise RuntimeError(
                "Semantic mode has no documents available on or before the "
                "training boundary; using validation documents would leak future evidence."
            )

        state["semantic_docs"] = semantic_train_df.to_dict(orient="records")
        state["semantic_docs_train_df"] = semantic_train_df
        state["semantic_docs_val_df"] = semantic_val_df
        state["semantic_split_meta"] = {
            "strategy": "timestamped_documents_only",
            "train_docs_available": int(len(semantic_train_available)),
            "train_docs": int(len(semantic_train_df)),
            "train_docs_selected": int(len(semantic_train_df)),
            "validation_docs": int(len(semantic_val_df)),
            "documents_after_validation_excluded": int(after_validation.sum()),
            "test_docs_exposed": 0,
            "train_end": train_end.isoformat(),
            "validation_start": pd.to_datetime(
                val_df["date"], errors="coerce", utc=True
            ).min().tz_localize(None).isoformat(),
            "validation_end": val_end.isoformat(),
        }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["step"] = int(state.get("planner_call_count", 0))
        analytical_state = dict(state)
        for forbidden in (
            "semantic_docs_df", "semantic_docs", "semantic_docs_train_df",
            "semantic_docs_val_df", "semantic_docs_test_df", "semantic_context",
            "semantic_signals", "semantic_raw_signals", "semantic_operational_signals",
            "planner_guidance", "diagnostics",
        ):
            analytical_state.pop(forbidden, None)
        analytical_state["analytical_agent_uses_semantic_output"] = False
        self.agent.run(analytical_state)
        raw_train = analytical_state.get("raw_train_df")
        summary = analytical_state.get("data_summary", {})
        if isinstance(raw_train, pd.DataFrame) and "target" in raw_train:
            from utils.metrics import build_mase_reference

            base = summary.get("base_metrics", {}) if isinstance(summary, dict) else {}
            analytical_state["mase_reference"] = build_mase_reference(
                raw_train["target"].to_numpy(dtype=float),
                inferred_freq=base.get("inferred_freq") or base.get("frequency"),
                seasonality_detected=bool(
                    summary.get("seasonality_detected", base.get("seasonality_detected", False))
                ),
                seasonality_period=summary.get("seasonal_periods"),
                suggested_lag=summary.get("suggested_lag"),
            )
        for key in ("train_df", "val_df", "raw_train_df", "raw_validation_df", "split_metadata", "data_summary", "dataset_length", "target_mase", "mase_reference", "analysis_history", "extreme_outliers_detected", "log_transform_recommended", "log_candidate_required", "log_transform_supported"):
            if key in analytical_state:
                state[key] = analytical_state[key]
        self._split_semantic_documents(state)
        return {
            "train_df": state.get("train_df"),
            "val_df": state.get("val_df"),
            "raw_train_df": state.get("raw_train_df"),
            "raw_validation_df": state.get("raw_validation_df"),
            "split_metadata": state.get("split_metadata", {}),
            "semantic_docs_df": state.get("semantic_docs_df"),
            "semantic_docs": state.get("semantic_docs"),
            "semantic_docs_train_df": state.get("semantic_docs_train_df"),
            "semantic_docs_val_df": state.get("semantic_docs_val_df"),
            "semantic_split_meta": state.get("semantic_split_meta"),
            "target_mase": state.get("target_mase", SETTINGS.finalize_mase_threshold),
            "mase_reference": state.get("mase_reference"),
            "data_summary": state.get("data_summary", {}),
            "extreme_outliers_detected": state.get(
                "extreme_outliers_detected",
                state.get("data_summary", {}).get(
                    "extreme_outliers_detected", {"found": False}
                ),
            ),
            "log_transform_recommended": bool(
                state.get("log_transform_recommended", False)
            ),
            "log_candidate_required": bool(
                state.get("log_candidate_required", False)
            ),
            "log_transform_supported": bool(
                state.get("log_transform_supported", False)
            ),
            "dataset_length": state.get("dataset_length", 0),
            "analytical_features": state.get("data_summary", {}),
            "model_prescription": state.get("model_prescription", {}),
            "model_selection_signals": state.get("model_selection_signals", {}),
            "analysis_history": state.get("analysis_history", []),
            "memory": state.get("memory", {}),
            "last_action": "analytical",
        }
