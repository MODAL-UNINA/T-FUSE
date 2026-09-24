"""Shared memory module used by planner and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class AgentMemory:
    """Cross-step memory for adaptive planning decisions."""

    failed_models: List[str] = field(default_factory=list)

    failed_combinations: List[Dict[str, Any]] = field(default_factory=list)
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    dataset_signatures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    candidate_records: List[Dict[str, Any]] = field(default_factory=list)

    def register_dataset_signature(
        self, series_id: str, train_df: pd.DataFrame
    ) -> None:
        """Store compact dataset metadata used by the planner."""
        if train_df.empty:
            signature = {"n_obs": 0, "target_mean": 0.0, "target_std": 0.0}
        else:
            target = train_df["target"].astype(float)
            signature = {
                "n_obs": int(len(train_df)),
                "target_mean": float(target.mean()),
                "target_std": float(target.std()),
                "date_min": str(train_df["date"].min()),
                "date_max": str(train_df["date"].max()),
            }
        self.dataset_signatures[series_id] = signature

    def record_model_result(
        self,
        model_name: str,
        rmse: float,
        strategy: str,
        regime: str,
        hyperparameters: Dict[str, Any] | None = None,
        preprocessing_signature: List[str] | None = None,
        target_rmse: float = 1.5,
        target_mase: float = 0.85,
        mase: float = 0.0,
        error_message: str = "",
        outcome_type: str = "completed",
    ) -> None:
        """Persist one terminal outcome without converting poor scores to faults."""
        hp = dict(hyperparameters or {})
        combo = {
            "model": model_name,
            "hyperparameters": hp,
            "preprocessing": list(preprocessing_signature or []),
            "rmse": float(rmse),
            "mase": float(mase),
            "error_message": error_message,
        }

        if outcome_type == "execution_failure":
            combo["failure_type"] = "execution_failure"
            if combo not in self.failed_combinations:
                self.failed_combinations.append(combo)
            return

        if outcome_type in {"invalid_configuration", "duplicate_candidate"}:
            return

        if mase <= target_mase or (mase == 0.0 and rmse <= target_rmse):
            self.successful_patterns.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "regime": regime,
                    "rmse": float(rmse),
                    "mase": float(mase),
                    "hyperparameters": hp,
                    "preprocessing": list(preprocessing_signature or []),
                    "error_message": error_message,
                    "failure_type": "",
                }
            )
