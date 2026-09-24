"""Fixed fuzzy runtime contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SUPPORTED_INFLUENCE_MODES = ("fuzzy",)


@dataclass(frozen=True)
class InfluenceComponents:
    influence_mode: str
    semantic_enabled: bool
    controller_mode: str
    reliability_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_influence_components(mode: str = "fuzzy") -> InfluenceComponents:
    if str(mode).strip().lower() != "fuzzy":
        raise ValueError("T-FUSE accepts only fuzzy influence")
    return InfluenceComponents("fuzzy", True, "fuzzy", "fuzzy")


def validate_experiment_config(config: Mapping[str, Any]) -> InfluenceComponents:
    experiment = config.get("experiment", {})
    configured = experiment.get("influence_mode", "fuzzy") if isinstance(experiment, Mapping) else "fuzzy"
    return derive_influence_components(str(configured))


__all__ = [
    "InfluenceComponents",
    "SUPPORTED_INFLUENCE_MODES",
    "derive_influence_components",
    "validate_experiment_config",
]
