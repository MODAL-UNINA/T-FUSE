"""Shared hyperparameter contracts for all agents.

"""

from __future__ import annotations

from typing import Any, Dict


def _integer(minimum: int, maximum: int, *, required: bool = True) -> Dict[str, Any]:
    return {"type": "integer", "min": minimum, "max": maximum, "required": required}


def _number(
    minimum: float,
    maximum: float,
    *,
    required: bool = True,
    min_exclusive: bool = False,
    max_exclusive: bool = False,
) -> Dict[str, Any]:
    return {
        "type": "number",
        "min": minimum,
        "max": maximum,
        "min_exclusive": min_exclusive,
        "max_exclusive": max_exclusive,
        "required": required,
    }


def _enum(values: list[Any], *, required: bool = True) -> Dict[str, Any]:
    return {"type": "enum", "values": values, "required": required}


def _boolean(*, required: bool = True) -> Dict[str, Any]:
    return {"type": "boolean", "required": required}

DEFAULT_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "ARIMA": {"order": [2, 1, 2]},
    "SARIMA": {"order": [3, 0, 1], "seasonal_order": [0, 1, 1, 7]},
    "Prophet": {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
    "ETS": {"trend": "add", "seasonal": None, "seasonal_periods": None},
    "MLP": {"hidden_layer_sizes": [64, 32]},
    "LSTM": {
        "lag": 12,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "epochs": 100,
        "learning_rate": 0.001,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "RandomForest": {"n_estimators": 200, "max_depth": 10},
    "LightGBM": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
    "CatBoost": {"iterations": 300, "learning_rate": 0.05, "depth": 6},
    "N-BEATS": {"lookback": 20, "hidden_units": 256, "stacks": 2, "epochs": 100},
    "Chronos": {"model_size": "tiny"},
    "DLinear": {"lag": 24, "epochs": 100},
    "Reformer": {"lag": 30, "epochs": 100, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
    "Autoformer": {"input_size": 96, "hidden_size": 512, "max_steps": 10, "n_head": 8, "encoder_layers": 2, "decoder_layers": 1},
    "Informer": {"input_size": 96, "hidden_size": 512, "max_steps": 10, "n_head": 8, "encoder_layers": 2, "decoder_layers": 1},
    "FEDformer": {"input_size": 96, "hidden_size": 512, "max_steps": 10, "n_head": 8, "encoder_layers": 2, "decoder_layers": 1},
    "N-HiTS": {"input_size": 96, "max_steps": 10},
    "PatchTST": {"input_size": 96, "hidden_size": 512, "max_steps": 10, "n_head": 8, "encoder_layers": 2},
    "TimesNet": {"input_size": 96, "hidden_size": 512, "max_steps": 10},
    "TiDE": {"input_size": 96, "hidden_size": 512, "max_steps": 10, "num_decoder_layers": 1},
    "ElasticNet": {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "selection": "cyclic",
        "positive": False,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
    },
    "SVR": {
        "C": 1.0,
        "epsilon": 0.1,
        "kernel": "rbf",
        "gamma": "scale",
        "degree": 3,
        "coef0": 0.0,
        "shrinking": True,
        "tol": 1e-3,
        "cache_size": 200.0,
        "max_iter": -1,
    },
}


HYPERPARAMETER_CONSTRAINTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "ARIMA": {
        "order": {
            "type": "tuple",
            "items": [_integer(0, 5), _integer(0, 2), _integer(0, 5)],
            "labels": ["p", "d", "q"],
            "required": True,
        },
    },
    "SARIMA": {
        "order": {
            "type": "tuple",
            "items": [_integer(0, 5), _integer(0, 2), _integer(0, 5)],
            "labels": ["p", "d", "q"],
            "required": True,
        },
        "seasonal_order": {
            "type": "tuple",
            "items": [_integer(0, 2), _integer(0, 1), _integer(0, 2), _integer(0, 366)],
            "labels": ["P", "D", "Q", "m"],
            "required": True,
        },
    },
    "Prophet": {
        "seasonality_mode": _enum(["additive", "multiplicative"]),
        "changepoint_prior_scale": _number(0.0, 1.0, min_exclusive=True),
    },
    "ETS": {
        "trend": _enum(["add", "mul", None]),
        "seasonal": _enum(["add", "mul", None]),
        "seasonal_periods": {
            "any_of": [_enum([None]), _integer(2, 366)],
            "required": False,
        },
    },
    "MLP": {
        "hidden_layer_sizes": {
            "type": "list",
            "items": _integer(1, 4096),
            "min_items": 1,
            "max_items": 8,
            "required": True,
        },
    },
    "LSTM": {
        "lag": _integer(1, 512),
        "hidden_size": _integer(1, 4096),
        "num_layers": _integer(1, 16),
        "dropout": _number(0.0, 1.0, max_exclusive=True),
        "epochs": _integer(1, 2000),
        "learning_rate": _number(0.0, 1.0, min_exclusive=True),
    },
    "XGBoost": {
        "n_estimators": _integer(1, 10000),
        "max_depth": _integer(1, 64),
        "learning_rate": _number(0.0, 1.0, min_exclusive=True),
        "subsample": _number(0.0, 1.0, required=False, min_exclusive=True),
        "colsample_bytree": _number(0.0, 1.0, required=False, min_exclusive=True),
    },
    "RandomForest": {
        "n_estimators": _integer(1, 10000),
        "max_depth": {
            "any_of": [_enum([None]), _integer(1, 256)],
            "required": True,
        },
    },
    "LightGBM": {
        "n_estimators": _integer(1, 10000),
        "learning_rate": _number(0.0, 1.0, min_exclusive=True),
        "max_depth": {
            "any_of": [_enum([-1]), _integer(1, 256)],
            "required": True,
        },
    },
    "CatBoost": {
        "iterations": _integer(1, 10000),
        "learning_rate": _number(0.0, 1.0, min_exclusive=True),
        "depth": _integer(1, 16),
    },
    "N-BEATS": {
        "lookback": _integer(2, 512),
        "hidden_units": _integer(1, 4096),
        "stacks": _integer(1, 64),
        "epochs": _integer(1, 2000),
    },
    "Chronos": {"model_size": _enum(["tiny", "mini", "small"])},
    "DLinear": {"lag": _integer(2, 512), "epochs": _integer(1, 2000)},
    "Reformer": {
        "lag": _integer(2, 512),
        "epochs": _integer(1, 2000),
        "nhead": _integer(1, 16),
        "num_layers": _integer(1, 16),
        "dim_feedforward": _integer(1, 8192),
    },
    "Autoformer": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
        "n_head": _integer(1, 16),
        "encoder_layers": _integer(1, 16),
        "decoder_layers": _integer(1, 16),
    },
    "Informer": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
        "n_head": _integer(1, 16),
        "encoder_layers": _integer(1, 16),
        "decoder_layers": _integer(1, 16),
    },
    "FEDformer": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
        "n_head": _enum([8]),
        "encoder_layers": _integer(1, 16),
        "decoder_layers": _integer(1, 16),
    },
    "N-HiTS": {"input_size": _integer(2, 2048), "max_steps": _integer(1, 2000)},
    "PatchTST": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
        "n_head": _integer(1, 16),
        "encoder_layers": _integer(1, 16),
    },
    "TimesNet": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
    },
    "TiDE": {
        "input_size": _integer(2, 2048),
        "hidden_size": _integer(1, 4096),
        "max_steps": _integer(1, 2000),
        "num_decoder_layers": _integer(1, 16),
    },
    "ElasticNet": {
        "alpha": _number(0.0, 1000000.0, min_exclusive=True),
        "l1_ratio": _number(0.0, 1.0),
        "fit_intercept": _boolean(required=False),
        "max_iter": _integer(1, 1000000, required=False),
        "tol": _number(0.0, 1.0, required=False, min_exclusive=True),
        "selection": _enum(["cyclic", "random"], required=False),
        "positive": _boolean(required=False),
    },
    "KNN": {
        "n_neighbors": _integer(1, 10000),
        "weights": _enum(["uniform", "distance"], required=False),
        "algorithm": _enum(["auto", "ball_tree", "kd_tree", "brute"], required=False),
        "leaf_size": _integer(1, 10000, required=False),
        "p": _integer(1, 100, required=False),
        "metric": _enum(["minkowski", "euclidean", "manhattan"], required=False),
    },
    "SVR": {
        "C": _number(0.0, 1000000.0, min_exclusive=True),
        "epsilon": _number(0.0, 1000000.0),
        "kernel": _enum(["rbf", "linear", "poly", "sigmoid"]),
        "gamma": {
            "any_of": [_enum(["scale", "auto"]), _number(0.0, 1000000.0, min_exclusive=True)],
            "required": False,
        },
        "degree": _integer(1, 100, required=False),
        "coef0": _number(-1000000.0, 1000000.0, required=False),
        "shrinking": _boolean(required=False),
        "tol": _number(0.0, 1.0, required=False, min_exclusive=True),
        "cache_size": _number(0.0, 1000000.0, required=False, min_exclusive=True),
        "max_iter": {
            "any_of": [_enum([-1]), _integer(1, 1000000)],
            "required": False,
        },
    },
}



HYPERPARAMETER_SUGGESTIONS: Dict[str, Dict[str, list[Any]]] = {
    "ARIMA": {"order": [[1, 0, 0], [1, 1, 1], [2, 1, 2], [3, 1, 2]]},
    "SARIMA": {
        "order": [[1, 1, 1], [2, 1, 2], [3, 0, 1]],

        "seasonal_order": [
            [0, 0, 0, 0],
            [0, 1, 1, 3],
            [1, 1, 1, 3],
            [0, 1, 1, 4],
            [1, 1, 1, 4],
            [0, 1, 1, 6],
            [1, 1, 1, 6],
            [0, 1, 1, 7],
            [1, 1, 1, 7],
            [0, 1, 1, 12],
            [1, 1, 1, 12],
            [0, 1, 1, 13],
            [1, 1, 1, 13],
            [0, 1, 1, 26],
            [1, 1, 1, 26],
            [0, 1, 1, 52],
            [1, 1, 1, 52],
        ],
    },
    "Prophet": {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    },
    "ETS": {
        "trend": ["add", "mul", None],
        "seasonal": ["add", "mul", None],
        "seasonal_periods": [None, 4, 7, 12, 24, 52],
    },
    "MLP": {"hidden_layer_sizes": [[32], [64, 32], [128, 64]]},
    "LSTM": {
        "lag": [6, 7, 12, 14, 24, 30],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2],
        "epochs": [50, 100, 200],
        "learning_rate": [0.001, 0.005, 0.01],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, None],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 6, 10],
    },
    "CatBoost": {
        "iterations": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
    },
    "N-BEATS": {
        "lookback": [10, 20, 30],
        "hidden_units": [128, 256, 512],
        "stacks": [2, 4, 8],
        "epochs": [50, 100],
    },
    "Chronos": {"model_size": ["tiny", "mini", "small"]},
    "DLinear": {"lag": [12, 24, 48], "epochs": [50, 100, 200]},
    "Reformer": {
        "lag": [15, 30, 60],
        "epochs": [50, 100],
        "nhead": [2, 4, 8],
        "num_layers": [1, 2, 4],
        "dim_feedforward": [64, 128, 256],
    },
    "Autoformer": {
        "input_size": [48, 96, 192],
        "hidden_size": [256, 512, 1024],
        "max_steps": [10, 20],
        "n_head": [4, 8],
        "encoder_layers": [1, 2],
        "decoder_layers": [1, 2],
    },
    "Informer": {
        "input_size": [48, 96, 192],
        "hidden_size": [256, 512],
        "max_steps": [10, 20],
        "n_head": [4, 8],
        "encoder_layers": [1, 2],
        "decoder_layers": [1, 2],
    },
    "FEDformer": {
        "input_size": [48, 96, 192],
        "hidden_size": [256, 512],
        "max_steps": [10, 20],
        "n_head": [8],
        "encoder_layers": [1, 2],
        "decoder_layers": [1, 2],
    },
    "N-HiTS": {"input_size": [48, 96, 192], "max_steps": [10, 20]},
    "PatchTST": {
        "input_size": [48, 96, 192],
        "hidden_size": [256, 512],
        "max_steps": [10, 20],
        "n_head": [4, 8],
        "encoder_layers": [1, 2],
    },
    "TimesNet": {"input_size": [48, 96, 192], "hidden_size": [256, 512], "max_steps": [10, 20]},
    "TiDE": {
        "input_size": [48, 96, 192],
        "hidden_size": [256, 512],
        "max_steps": [10, 20],
        "num_decoder_layers": [1, 2],
    },
    "ElasticNet": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "fit_intercept": [True, False],
        "max_iter": [1000, 3000, 5000],
        "tol": [1e-4, 1e-3],
        "selection": ["cyclic", "random"],
        "positive": [False, True],
    },
    "KNN": {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2],
        "metric": ["minkowski", "euclidean", "manhattan"],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1, 0.2],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
        "coef0": [0.0, 0.1, 1.0],
        "shrinking": [True, False],
        "tol": [1e-4, 1e-3],
        "cache_size": [200.0, 500.0],
        "max_iter": [-1, 1000, 5000],
    },
}


HYPERPARAMETER_SEARCH_SPACE = HYPERPARAMETER_CONSTRAINTS


MODEL_HYPERPARAMETER_CONTRACT = {
    "ARIMA": {
        "order": "list[int, int, int] = [p, d, q]; 0 <= p,q <= 5 and 0 <= d <= 2."
    },
    "SARIMA": {
        "order": "list[int, int, int] = [p, d, q]; 0 <= p,q <= 5 and 0 <= d <= 2.",
        "seasonal_order": "list[int, int, int, int] = [P, D, Q, m]; 0 <= P,Q <= 2, 0 <= D <= 1, and either all seasonal terms including m are zero or 2 <= m <= 366."
    },
    "Prophet": {
        "seasonality_mode": "str enum: additive | multiplicative.",
        "changepoint_prior_scale": "float > 0; smaller values produce smoother trends, larger values allow more trend flexibility."
    },
    "ETS": {
        "trend": "str or null: add | mul | null.",
        "seasonal": "str or null: add | mul | null.",
        "seasonal_periods": "int or null; must match a plausible seasonal period for the frequency."
    },
    "MLP": {
        "hidden_layer_sizes": "list[int] or tuple[int, ...]; hidden layer widths."
    },
    "LSTM": {
        "lag": "int > 0; lookback length.",
        "hidden_size": "int > 0.",
        "num_layers": "int >= 1.",
        "dropout": "float in [0, 1).",
        "epochs": "int > 0.",
        "learning_rate": "float > 0."
    },
    "XGBoost": {
        "n_estimators": "int > 0.",
        "max_depth": "int > 0.",
        "learning_rate": "float > 0.",
        "subsample": "float in (0, 1].",
        "colsample_bytree": "float in (0, 1]."
    },
    "RandomForest": {
        "n_estimators": "int > 0.",
        "max_depth": "int > 0 or null."
    },
    "LightGBM": {
        "n_estimators": "int > 0.",
        "learning_rate": "float > 0.",
        "max_depth": "int; -1 means unlimited depth."
    },
    "CatBoost": {
        "iterations": "int > 0.",
        "learning_rate": "float > 0.",
        "depth": "int > 0."
    },
    "N-BEATS": {
        "lookback": "int > 0.",
        "hidden_units": "int > 0.",
        "stacks": "int > 0.",
        "epochs": "int > 0."
    },
    "Chronos": {
        "model_size": "str enum: tiny | mini | small."
    },
    "DLinear": {
        "lag": "int > 0.",
        "epochs": "int > 0."
    },
    "Reformer": {
        "lag": "int > 0.",
        "epochs": "int > 0.",
        "nhead": "int > 0.",
        "num_layers": "int >= 1.",
        "dim_feedforward": "int > 0."
    },
    "Autoformer": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0.",
        "n_head": "int > 0.",
        "encoder_layers": "int >= 1.",
        "decoder_layers": "int >= 1."
    },
    "Informer": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0.",
        "n_head": "int > 0.",
        "encoder_layers": "int >= 1.",
        "decoder_layers": "int >= 1."
    },
    "FEDformer": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0.",
        "n_head": "must be the integer 8 (NeuralForecast FEDformer constraint).",
        "encoder_layers": "int >= 1.",
        "decoder_layers": "int >= 1."
    },
    "N-HiTS": {
        "input_size": "int > 0.",
        "max_steps": "int > 0."
    },
    "PatchTST": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0.",
        "n_head": "int > 0.",
        "encoder_layers": "int >= 1."
    },
    "TimesNet": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0."
    },
    "TiDE": {
        "input_size": "int > 0.",
        "hidden_size": "int > 0.",
        "max_steps": "int > 0.",
        "num_decoder_layers": "int >= 1."
    },
    "ElasticNet": {
        "alpha": "float > 0.",
        "l1_ratio": "float in [0, 1].",
        "fit_intercept": "bool.",
        "max_iter": "int > 0.",
        "tol": "float > 0.",
        "selection": "str enum: cyclic | random.",
        "positive": "bool."
    },
    "KNN": {
        "n_neighbors": "int > 0.",
        "weights": "str enum: uniform | distance.",
        "algorithm": "str enum: auto | ball_tree | kd_tree | brute.",
        "leaf_size": "int > 0.",
        "p": "int, usually 1 or 2.",
        "metric": "str enum: minkowski | euclidean | manhattan."
    },
    "SVR": {
        "C": "float > 0.",
        "epsilon": "float >= 0.",
        "kernel": "str enum: rbf | linear | poly | sigmoid.",
        "gamma": "str or float: scale | auto | positive float.",
        "degree": "int >= 1; relevant mainly for poly kernel.",
        "coef0": "float; relevant for poly/sigmoid kernels.",
        "shrinking": "bool.",
        "tol": "float > 0.",
        "cache_size": "float > 0.",
        "max_iter": "int; -1 means no hard iteration limit."
    },
}
