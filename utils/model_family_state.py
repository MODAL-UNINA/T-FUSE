"""Canonical model-family registry and label normalization."""

from __future__ import annotations

from typing import Any, Dict, List

CANONICAL_FAMILY_ORDER: List[str] = [
    "Statistical",
    "Additive",
    "Tree-based",
    "Classical ML",
    "Neural",
    "Transformer/Foundation",
]

MODEL_TO_FAMILY: Dict[str, str] = {
    "ARIMA": "Statistical",
    "SARIMA": "Statistical",
    "ETS": "Statistical",
    "Prophet": "Additive",
    "XGBoost": "Tree-based",
    "LightGBM": "Tree-based",
    "CatBoost": "Tree-based",
    "RandomForest": "Tree-based",
    "ElasticNet": "Classical ML",
    "KNN": "Classical ML",
    "SVR": "Classical ML",
    "MLP": "Neural",
    "LSTM": "Neural",
    "DLinear": "Neural",
    "N-BEATS": "Neural",
    "N-HiTS": "Neural",
    "TiDE": "Neural",
    "Autoformer": "Transformer/Foundation",
    "Informer": "Transformer/Foundation",
    "FEDformer": "Transformer/Foundation",
    "PatchTST": "Transformer/Foundation",
    "TimesNet": "Neural",
    "Reformer": "Transformer/Foundation",
    "Chronos": "Transformer/Foundation",
}

_FAMILY_ALIASES = {
    "transformer": "Transformer/Foundation",
    "foundation": "Transformer/Foundation",
    "transformer/foundation": "Transformer/Foundation",
    "statistical": "Statistical",
    "additive": "Additive",
    "additive/decomposable": "Additive",
    "tree-based": "Tree-based",
    "tree based": "Tree-based",
    "classical ml": "Classical ML",
    "neural": "Neural",
    "statistical time series": "Statistical",
    "linear regression": "Classical ML",
    "linear models": "Classical ML",
    "gradient boosting": "Tree-based",
    "gradient boosting trees": "Tree-based",
    "tree based ensemble": "Tree-based",
    "deep learning sequence": "Neural",
}



def canonicalize_family_label(label: Any) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return ""
        
    lower_norm = normalized.lower()
    alias = _FAMILY_ALIASES.get(lower_norm)
    if alias:
        return alias
        
    if "statistical" in lower_norm or "arima" in lower_norm:
        return "Statistical"
    if "tree" in lower_norm or "boosting" in lower_norm or "forest" in lower_norm:
        return "Tree-based"
    if "deep learning" in lower_norm or "neural" in lower_norm or "sequence" in lower_norm:
        return "Neural"
    if "transformer" in lower_norm or "foundation" in lower_norm:
        return "Transformer/Foundation"
    if "additive" in lower_norm or "prophet" in lower_norm:
        return "Additive"
    if "classical" in lower_norm or "regression" in lower_norm:
        return "Classical ML"
        
    return normalized


def model_to_family(model_name: Any) -> str:
    model = str(model_name or "").strip()
    if not model:
        return ""
    return MODEL_TO_FAMILY.get(model, "Other")
