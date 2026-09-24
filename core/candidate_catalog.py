"""Immutable candidate-space contract and deterministic per-candidate seeds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from models.defaults import HYPERPARAMETER_CONSTRAINTS, HYPERPARAMETER_SUGGESTIONS


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Dataset fingerprint requires a non-empty DataFrame")
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


@dataclass(frozen=True)
class CandidateCatalog:
    """Frozen runtime constraints plus non-binding candidate suggestions."""

    _constraints: Mapping[str, Mapping[str, Any]]
    _suggestions: Mapping[str, Mapping[str, Any]]
    fingerprint: str

    @classmethod
    def from_contracts(
        cls,
        constraints: Mapping[str, Mapping[str, Mapping[str, Any]]],
        suggestions: Mapping[str, Mapping[str, list[Any]]] | None = None,
    ) -> "CandidateCatalog":
        normalized_constraints: dict[str, dict[str, Any]] = {}
        for model, schema in sorted(constraints.items()):
            if not isinstance(schema, Mapping) or not schema:
                raise ValueError(f"Empty hyperparameter constraints for {model}")
            normalized_constraints[str(model)] = {}
            for key, rule in sorted(schema.items()):
                if not isinstance(rule, Mapping) or not (
                    isinstance(rule.get("type"), str)
                    or isinstance(rule.get("any_of"), list)
                ):
                    raise ValueError(f"Invalid constraint for {model}.{key}")
                normalized_constraints[str(model)][str(key)] = _thaw(_freeze(rule))

        normalized_suggestions: dict[str, dict[str, Any]] = {}
        for model, schema in sorted((suggestions or {}).items()):
            if model not in normalized_constraints:
                raise ValueError(f"Suggestions declared for unknown model {model}")
            normalized_suggestions[str(model)] = {}
            for key, values in sorted(schema.items()):
                if key not in normalized_constraints[model]:
                    raise ValueError(
                        f"Suggestions declared for unknown parameter {model}.{key}"
                    )
                if not isinstance(values, list) or not values:
                    raise ValueError(f"Empty suggestions for {model}.{key}")
                for value in values:
                    _validate_rule(
                        value,
                        normalized_constraints[model][key],
                        f"{model}.{key}",
                    )
                normalized_suggestions[str(model)][str(key)] = list(values)

        payload = {
            "constraints": normalized_constraints,
            "suggestions": normalized_suggestions,
        }
        fingerprint = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return cls(
            _freeze(normalized_constraints),
            _freeze(normalized_suggestions),
            fingerprint,
        )

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self._constraints)

    def validate(self, model: str, hyperparameters: Mapping[str, Any]) -> None:
        if model not in self._constraints:
            raise ValueError(f"Model {model!r} is outside CandidateCatalog")
        if not isinstance(hyperparameters, Mapping) or not hyperparameters:
            raise ValueError(f"{model} requires explicit hyperparameters")
        schema = self._constraints[model]
        unknown = sorted(set(hyperparameters) - set(schema))
        if unknown:
            raise ValueError(
                f"Hyperparameters outside CandidateCatalog for {model}: {unknown}"
            )
        missing = sorted(
            key
            for key, rule in schema.items()
            if bool(rule.get("required", True)) and key not in hyperparameters
        )
        if missing:
            raise ValueError(
                f"Missing required hyperparameters for {model}: {missing}"
            )
        for key, value in hyperparameters.items():
            _validate_rule(value, schema[key], f"{model}.{key}")

        if model == "ETS":
            seasonal = hyperparameters.get("seasonal")
            periods = hyperparameters.get("seasonal_periods")
            if seasonal is None and periods is not None:
                raise ValueError("ETS with seasonal=None requires seasonal_periods=None")
            if seasonal is not None and periods is None:
                raise ValueError("Seasonal ETS requires an integer seasonal_periods")
        if model == "SARIMA":
            seasonal_order = hyperparameters.get("seasonal_order")
            if isinstance(seasonal_order, (list, tuple)) and len(seasonal_order) == 4:
                p_seasonal, d_seasonal, q_seasonal, period = seasonal_order
                if period == 0 and any((p_seasonal, d_seasonal, q_seasonal)):
                    raise ValueError(
                        "SARIMA seasonal_order with m=0 requires P=D=Q=0"
                    )
                if period == 1:
                    raise ValueError("SARIMA seasonal period m must be 0 or at least 2")
        if model == "Reformer":
            nhead = hyperparameters.get("nhead")
            if isinstance(nhead, int) and 32 % nhead != 0:
                raise ValueError("Reformer nhead must divide the fixed model dimension 32")
        if model in {"Autoformer", "Informer", "FEDformer", "PatchTST"}:
            hidden_size = hyperparameters.get("hidden_size")
            n_head = hyperparameters.get("n_head")
            if (
                isinstance(hidden_size, int)
                and isinstance(n_head, int)
                and hidden_size % n_head != 0
            ):
                raise ValueError(f"{model} hidden_size must be divisible by n_head")

    def constraints_dict(self) -> dict[str, dict[str, Any]]:
        return _thaw(self._constraints)

    def suggestions_dict(self) -> dict[str, dict[str, list[Any]]]:
        return _thaw(self._suggestions)

    def finite_values(self, model: str) -> dict[str, list[Any]] | None:
        """Return a complete finite domain, or None when any rule is ranged."""
        schema = self._constraints.get(model)
        if not isinstance(schema, Mapping):
            return None
        finite: dict[str, list[Any]] = {}
        for key, rule in schema.items():
            values = _finite_rule_values(rule)
            if values is None:
                return None
            finite[str(key)] = values
        return finite


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_rule(value: Any, rule: Mapping[str, Any], path: str) -> None:
    alternatives = rule.get("any_of")
    if isinstance(alternatives, (list, tuple)):
        for alternative in alternatives:
            try:
                _validate_rule(value, alternative, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}={value!r} does not satisfy any allowed constraint")

    kind = rule.get("type")
    if kind == "enum":
        values = rule.get("values", ())
        if not any(_canonical(value) == _canonical(item) for item in values):
            raise ValueError(f"{path}={value!r} must be one of {_thaw(values)!r}")
        return
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{path}={value!r} must be a boolean")
        return
    if kind == "integer":
        if type(value) is not int:
            raise ValueError(f"{path}={value!r} must be an integer")
        _validate_bounds(value, rule, path)
        return
    if kind == "number":
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValueError(f"{path}={value!r} must be a finite number")
        _validate_bounds(float(value), rule, path)
        return
    if kind == "tuple":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}={value!r} must be a list or tuple")
        items = rule.get("items", ())
        if len(value) != len(items):
            raise ValueError(f"{path} must contain exactly {len(items)} values")
        labels = rule.get("labels", ())
        for index, (item, item_rule) in enumerate(zip(value, items)):
            label = labels[index] if index < len(labels) else str(index)
            _validate_rule(item, item_rule, f"{path}.{label}")
        return
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}={value!r} must be a list or tuple")
        minimum = int(rule.get("min_items", 0))
        maximum = int(rule.get("max_items", len(value)))
        if not minimum <= len(value) <= maximum:
            raise ValueError(
                f"{path} must contain between {minimum} and {maximum} values"
            )
        item_rule = rule.get("items")
        for index, item in enumerate(value):
            _validate_rule(item, item_rule, f"{path}[{index}]")
        return
    raise ValueError(f"Unsupported constraint type {kind!r} for {path}")


def _validate_bounds(value: float | int, rule: Mapping[str, Any], path: str) -> None:
    minimum = rule.get("min")
    maximum = rule.get("max")
    if minimum is not None:
        invalid = value <= minimum if rule.get("min_exclusive") else value < minimum
        if invalid:
            operator = ">" if rule.get("min_exclusive") else ">="
            raise ValueError(f"{path}={value!r} must be {operator} {minimum}")
    if maximum is not None:
        invalid = value >= maximum if rule.get("max_exclusive") else value > maximum
        if invalid:
            operator = "<" if rule.get("max_exclusive") else "<="
            raise ValueError(f"{path}={value!r} must be {operator} {maximum}")


def _finite_rule_values(rule: Mapping[str, Any]) -> list[Any] | None:
    alternatives = rule.get("any_of")
    if isinstance(alternatives, (list, tuple)):
        combined: list[Any] = []
        for alternative in alternatives:
            values = _finite_rule_values(alternative)
            if values is None:
                return None
            for value in values:
                if not any(_canonical(value) == _canonical(item) for item in combined):
                    combined.append(value)
        return combined
    if rule.get("type") == "enum":
        return _thaw(rule.get("values", ()))
    if rule.get("type") == "boolean":
        return [False, True]
    return None


DEFAULT_CANDIDATE_CATALOG = CandidateCatalog.from_contracts(
    HYPERPARAMETER_CONSTRAINTS,
    HYPERPARAMETER_SUGGESTIONS,
)


def derive_candidate_seed(
    dataset_fingerprint: str,
    experiment_seed: int,
    configuration_signature: str,
) -> int:
    payload = (
        f"{dataset_fingerprint}|{int(experiment_seed)}|{configuration_signature}"
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big"
    )


def set_deterministic_seed(seed: int) -> dict[str, Any]:
    """Seed supported runtimes and report determinism limitations."""
    seed = int(seed)
    numpy_seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(numpy_seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    audit: dict[str, Any] = {
        "seed": seed,
        "effective_python_seed": seed,
        "effective_numpy_seed": numpy_seed,
        "effective_torch_seed": seed,
        "effective_cuda_seed": seed,
        "random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": False,
        "cuda_seeded": False,
        "nondeterministic_operations": [],
    }
    try:
        import torch
    except ImportError:
        audit["nondeterministic_operations"].append("torch_not_installed")
        return audit
    torch.manual_seed(seed)
    audit["torch_seeded"] = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        audit["cuda_seeded"] = True
    else:
        audit["nondeterministic_operations"].append("cuda_not_available")
    return audit


__all__ = [
    "CandidateCatalog",
    "DEFAULT_CANDIDATE_CATALOG",
    "dataframe_fingerprint",
    "derive_candidate_seed",
    "set_deterministic_seed",
]
