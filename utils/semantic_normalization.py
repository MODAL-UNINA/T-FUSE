import math
import re
from typing import Any, Mapping


def _finite_nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, number) if math.isfinite(number) else 0.0


def normalize_family_support(raw_support: Mapping[str, Any]) -> dict[str, float]:
    """Normalize accumulated semantic support across model families.

    Dividing every family by the same maximum preserves the relative evidence
    magnitude. Negative evidence may remain in the raw aggregation, while the
    preference passed downstream is a non-negative score in ``[0, 1]``.
    """
    values = {
        str(family): _finite_nonnegative(value)
        for family, value in raw_support.items()
    }
    maximum = max(values.values(), default=0.0)
    if maximum <= 0.0:
        return {family: 0.0 for family in values}
    return {family: value / maximum for family, value in values.items()}


def family_score_discriminability(scores: Mapping[str, Any]) -> float:
    """Measure cross-family score spread in ``[0, 1]``.

    Uniform scores yield zero. Since bounded scores have population standard
    deviation at most 0.5, dividing by 0.5 gives an interpretable unit scale.
    """
    values = [_finite_nonnegative(value) for value in scores.values()]
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return max(0.0, min(1.0, std / 0.5))

def normalize_semantic_text(value: str) -> str:
    """Normalize text for semantic comparison.
    
    - Convert to lowercase
    - Replace underscores and dashes with spaces
    - Remove insignificant punctuation
    - Remove multiple spaces
    - Return 'unknown' if empty
    """
    if not value or not isinstance(value, str):
        return "unknown"
        
    v = value.lower()
    v = v.replace("_", " ").replace("-", " ")
    # Keep alphanumeric and basic spaces
    v = re.sub(r'[^\w\s]', '', v)
    v = re.sub(r'\s+', ' ', v).strip()
    
    if not v or v == "unknown":
        return "unknown"
        
    return v
