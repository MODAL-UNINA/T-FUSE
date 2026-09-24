"""Minimal experiment manifest for reproducible T-FUSE runs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class ExperimentManifest:
    dataset_fingerprint: str
    candidate_catalog_fingerprint: str
    seed: int
    influence_mode: str
    maximum_valid_trainings: int
    forecast_horizon: int
    semantic_document_filtering: dict[str, Any]

    @classmethod
    def create(cls, **kwargs: Any) -> "ExperimentManifest":
        allowed = set(cls.__dataclass_fields__)
        data = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

__all__ = ["ExperimentManifest"]
