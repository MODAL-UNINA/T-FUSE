"""Central runtime model catalog.

"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Iterable

from utils.model_family_state import MODEL_TO_FAMILY


OPTIONAL_MODEL_DEPENDENCIES = {
    "Prophet": "prophet",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "Chronos": "chronos",
    "Reformer": "transformers",
    "Autoformer": "neuralforecast",
    "Informer": "neuralforecast",
    "FEDformer": "neuralforecast",
    "N-HiTS": "neuralforecast",
    "PatchTST": "neuralforecast",
    "TimesNet": "neuralforecast",
    "TiDE": "neuralforecast",
}


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def build_model_catalog(
    *,
    configured_models: Iterable[str] | None = None,
    disabled_models: Iterable[str] = (),
    failed_models: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the single model-status source used by Planner and SafetyGuard."""
    models = [
        str(model)
        for model in (configured_models or MODEL_TO_FAMILY)
        if str(model) in MODEL_TO_FAMILY
    ]
    disabled = {str(model) for model in disabled_models}
    failed = {str(model) for model in failed_models}
    unavailable = {
        model
        for model in models
        if model in OPTIONAL_MODEL_DEPENDENCIES
        and not _installed(OPTIONAL_MODEL_DEPENDENCIES[model])
    }
    available = [
        model
        for model in models
        if model not in unavailable
        and model not in disabled
        and model not in failed
    ]
    return {
        "available_models": available,
        "unavailable_models": sorted(
            unavailable, key=models.index
        ),
        "disabled_models": [
            model for model in models if model in disabled
        ],
        "failed_models": [
            model for model in models if model in failed
        ],
    }




__all__ = [
    "OPTIONAL_MODEL_DEPENDENCIES",
    "build_model_catalog",
]
