"""Immutable lifecycle record for one concrete search candidate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

from utils.configuration_identity import canonical_configuration_signature
from utils.model_family_state import MODEL_TO_FAMILY


ERROR_TYPES = {
    "invalid_configuration",
    "duplicate_candidate",
    "execution_failure",
    "finite_underperformance",
}
ERROR_COUNTER_TYPES = (
    "invalid_configuration",
    "duplicate_candidate",
    "execution_failure",
    "technical_generation_failure",
)
TERMINAL_STATUSES = {"completed", "failed", "rejected", "duplicate"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def candidate_error_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Return normalized, non-negative counters for distinct failure classes."""
    raw = state.get("candidate_error_counts", {})
    if not isinstance(raw, Mapping):
        raw = {}
    normalized: dict[str, int] = {}
    for error_type in ERROR_COUNTER_TYPES:
        try:
            value = int(raw.get(error_type, 0))
        except (TypeError, ValueError):
            value = 0
        normalized[error_type] = max(0, value)
    return normalized


def increment_candidate_error(
    state: dict[str, Any], error_type: str
) -> dict[str, int]:
    """Increment exactly one error class and persist the normalized counters."""
    if error_type not in ERROR_COUNTER_TYPES:
        raise ValueError(f"Unsupported candidate error counter: {error_type}")
    counts = candidate_error_counts(state)
    counts[error_type] += 1
    state["candidate_error_counts"] = counts
    return counts


def cumulative_retry_count(state: Mapping[str, Any]) -> int:
    """Compatibility total; never use this aggregate as execution-failure count."""
    return sum(candidate_error_counts(state).values())


@dataclass(frozen=True)
class CandidateRecord:
    """Deeply immutable snapshot of a candidate and its terminal outcome."""

    candidate_id: str
    model: str
    family: str
    hyperparameters: Mapping[str, Any]
    preprocessing_signature: tuple[Any, ...]
    seed: int
    status: str
    validation_metrics: Mapping[str, Any]
    error_type: str | None
    started_at: str
    ended_at: str | None
    configuration_signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperparameters", _freeze(self.hyperparameters))
        object.__setattr__(
            self, "preprocessing_signature", tuple(_freeze(self.preprocessing_signature))
        )
        object.__setattr__(
            self, "validation_metrics", _freeze(self.validation_metrics)
        )
        if self.error_type is not None and self.error_type not in ERROR_TYPES:
            raise ValueError(f"Unsupported candidate error_type: {self.error_type}")

    @classmethod
    def pending(
        cls,
        candidate: Mapping[str, Any],
        *,
        seed: int,
        occurrence: int = 0,
        started_at: str | None = None,
    ) -> "CandidateRecord":
        model = str(candidate.get("model_type", ""))
        hp = dict(candidate.get("hyperparameters", {}) or {})
        preprocessing = candidate.get("preprocessing", {})
        if isinstance(preprocessing, Mapping):
            transformations = list(preprocessing.get("transformations", []) or [])
        else:
            transformations = list(preprocessing or [])
        signature = canonical_configuration_signature(model, hp, transformations)
        digest = hashlib.sha256(
            f"{signature}|{int(seed)}|{int(occurrence)}".encode("utf-8")
        ).hexdigest()[:20]
        return cls(
            candidate_id=f"candidate-{digest}",
            model=model,
            family=MODEL_TO_FAMILY.get(model, "Unknown"),
            hyperparameters=hp,
            preprocessing_signature=tuple(transformations),
            seed=int(seed),
            status="pending",
            validation_metrics={},
            error_type=None,
            started_at=started_at or _utc_now(),
            ended_at=None,
            configuration_signature=signature,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            model=str(payload["model"]),
            family=str(payload.get("family", "Unknown")),
            hyperparameters=dict(payload.get("hyperparameters", {}) or {}),
            preprocessing_signature=tuple(
                payload.get("preprocessing_signature", []) or []
            ),
            seed=int(payload.get("seed", 0)),
            status=str(payload.get("status", "pending")),
            validation_metrics=dict(payload.get("validation_metrics", {}) or {}),
            error_type=payload.get("error_type"),
            started_at=str(payload.get("started_at") or _utc_now()),
            ended_at=payload.get("ended_at"),
            configuration_signature=str(payload["configuration_signature"]),
        )

    def complete(
        self,
        validation_metrics: Mapping[str, Any],
        *,
        finite_underperformance: bool = False,
        ended_at: str | None = None,
    ) -> "CandidateRecord":
        return replace(
            self,
            status="completed",
            validation_metrics=dict(validation_metrics),
            error_type=(
                "finite_underperformance" if finite_underperformance else None
            ),
            ended_at=ended_at or _utc_now(),
        )

    def fail(
        self,
        error_type: str,
        *,
        ended_at: str | None = None,
    ) -> "CandidateRecord":
        if error_type not in ERROR_TYPES:
            raise ValueError(f"Unsupported candidate error_type: {error_type}")
        status = "duplicate" if error_type == "duplicate_candidate" else (
            "rejected" if error_type == "invalid_configuration" else "failed"
        )
        return replace(
            self,
            status=status,
            error_type=error_type,
            ended_at=ended_at or _utc_now(),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "family": self.family,
            "hyperparameters": _thaw(self.hyperparameters),
            "preprocessing_signature": _thaw(self.preprocessing_signature),
            "seed": self.seed,
            "status": self.status,
            "validation_metrics": _thaw(self.validation_metrics),
            "error_type": self.error_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "configuration_signature": self.configuration_signature,
        }
