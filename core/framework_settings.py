"""Fixed methodological settings and explicit inputs for T-FUSE experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class DataSplit:
    """Chronological train, validation, and test proportions."""

    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not 0.0 < value < 1.0 for value in values):
            raise ValueError("Split values must be strictly between 0 and 1.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Split values must sum to 1.0.")

    def sizes(
        self, total: int, forecast_horizon: int
    ) -> tuple[int, int, int]:
        """Return contiguous sizes with complete validation and test windows.

        Requested validation and test sizes are rounded down independently to
        multiples of ``forecast_horizon``. Every residual observation is kept
        in the training partition.
        """
        if total < 3:
            raise ValueError("At least three observations are required for a split.")
        horizon = int(forecast_horizon)
        if horizon < 1:
            raise ValueError("forecast_horizon must be a positive integer.")
        requested_validation = int(total * self.validation)
        requested_test = int(total * self.test)
        validation = (requested_validation // horizon) * horizon
        test = (requested_test // horizon) * horizon
        train = total - validation - test
        if validation < horizon or test < horizon:
            raise ValueError(
                "Split cannot provide at least one complete forecast-horizon "
                "window in both validation and test partitions."
            )
        return train, validation, test


@dataclass(frozen=True)
class ExperimentInputs:
    """The only method inputs that vary from one forecasting experiment to another."""

    dataset_path: Path
    forecast_horizon: int
    split: DataSplit

    def __post_init__(self) -> None:
        if int(self.forecast_horizon) < 1:
            raise ValueError("forecast_horizon must be a positive integer.")
        object.__setattr__(self, "dataset_path", Path(self.dataset_path).expanduser())

    @property
    def series_id(self) -> str:
        return self.dataset_path.stem

    @property
    def domain(self) -> str | None:
        parent = self.dataset_path.parent.name.strip()
        return parent or None


@dataclass(frozen=True)
class FrameworkSettings:
    """Immutable parameters that define the T-FUSE method."""

    seed: int = 42
    minimum_training_observations: int = 30
    max_replanning_attempts: int = 3
    max_iterations: int = 80
    maximum_valid_trainings: int = 30
    minimum_valid_trainings_before_stop: int = 5
    enforce_full_valid_training_budget: bool = False
    finalize_mase_threshold: float = 0.85
    max_technical_retries: int = 2
    output_root: str = "outputs"
    isolate_runs: bool = True
    planner_policy: Mapping[str, float | int | bool] = field(default_factory=lambda: MappingProxyType({
        "empirical_full_strength_trials": 6,
        "stopping_patience": 5,
        "minimum_relative_improvement": 0.002,
        "exploration_bonus": 0.12,
        "guidance_exploration_bonus": 0.10,
        "cost_penalty": 0.06,
        "failure_penalty": 0.16,
        "max_refinement_trials_per_model": 4,
        "global_information_window": 3,
        "capture_full_trace": True,
    }))
    replace_outliers: Mapping[str, float | int | str | bool] = field(default_factory=lambda: MappingProxyType({
        "enabled": True,
        "direction": "upper",
        "detection_method": "robust_mad",
        "threshold": 8.0,
        "max_replacements": 2,
        "replacement_method": "rolling_median",
        "replacement_window": 3,
    }))
    semantic: Mapping[str, float | int | bool] = field(default_factory=lambda: MappingProxyType({
        "max_docs": 0,
        "max_chunks": 0,
        "chunk_size": 100,
        "max_chars_per_doc": 2000,
        "max_total_chars": 24000,
        "max_prompt_tokens": 12000,
        "deduplicate_facts": True,
        "near_duplicate_threshold": 0.90,
        "include_prediction_text": True,
    }))
    semantic_aggregation: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({
        "domain_label_weight": 0.6,
        "domain_context_weight": 0.4,
        "min_domain_concept_overlap": 0.15,
        "alternative_domain_support_threshold": 0.1,
    }))
    semantic_target_resolution: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({
        "material_alternative_threshold": 0.15,
        "quantity_similarity_threshold": 0.8,
    }))
    llm_model_name: str = "Qwen/Qwen3.6-27B"
    llm_max_new_tokens: int = 3000
    llm_temperature: float = 0.5
    llm_do_sample: bool = False
    llm_device: str = "auto"


SETTINGS = FrameworkSettings()


__all__ = ["DataSplit", "ExperimentInputs", "FrameworkSettings", "SETTINGS"]
