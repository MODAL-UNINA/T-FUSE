"""Deterministic analytical agent for time-series diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict
import re 
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from core.candidate_catalog import dataframe_fingerprint
from core.framework_settings import SETTINGS
from utils.replace_outliers import detect_extreme_outliers

logger = logging.getLogger(__name__)


@dataclass
class AnalyticalAgent:
    """Extract statistical and temporal diagnostics without external inference."""

    @staticmethod
    def _autocorr(values: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at specified lag."""
        if len(values) <= lag or lag <= 0:
            return 0.0
        head = values[:-lag]
        tail = values[lag:]
        if np.std(head) < 1e-12 or np.std(tail) < 1e-12:
            return 0.0
        return float(np.corrcoef(head, tail)[0, 1])


    @staticmethod
    def infer_freq_tolerant(date_col, min_score=0.75, prefer_business=False):

        _WEEKDAYS = {
            0: "MON",
            1: "TUE",
            2: "WED",
            3: "THU",
            4: "FRI",
            5: "SAT",
            6: "SUN",
        }

        s = pd.to_datetime(date_col, errors="coerce")
        s = pd.Series(s).dropna().sort_values().drop_duplicates()

        if len(s) < 3:
            return "unknown", {"reason": "too_few_dates"}

        # 1. Standard inference
        strict_freq = pd.infer_freq(s)
        if strict_freq is not None:
            return strict_freq, {
                "method": "pd.infer_freq",
                "score": 1.0,
                "scores": {strict_freq: 1.0},
            }

        # Normalize by calendar date, which is useful for daily and weekly series.
        s_date = s.dt.normalize().drop_duplicates().sort_values()

        day_steps = s_date.diff().dropna().dt.days

        scores = {}

        # 2. Approximately daily
        if len(day_steps) > 0:
            scores["D"] = (day_steps == 1).mean()
        else:
            scores["D"] = 0.0

        # 3. Approximately business-daily, when applicable
        if prefer_business:
            bdays = pd.bdate_range(s_date.min(), s_date.max())
            observed = pd.DatetimeIndex(s_date)
            if len(bdays) > 0:
                bday_coverage = len(observed.intersection(bdays)) / len(bdays)
            else:
                bday_coverage = 0.0

            mostly_weekdays = (s_date.dt.weekday < 5).mean()
            scores["B"] = bday_coverage if mostly_weekdays >= 0.95 else 0.0

        # 4. Approximately weekly
        if len(day_steps) > 0:
            main_weekday = int(s_date.dt.weekday.mode().iloc[0])
            weekly_freq = f"W-{_WEEKDAYS[main_weekday]}"
            scores[weekly_freq] = (day_steps == 7).mean()

        # 5. Approximately monthly: calendar-aware rather than day-count based
        month_id = s.dt.year * 12 + s.dt.month
        month_steps = month_id.diff().dropna()
        scores["MS"] = (month_steps == 1).mean()

        # 6. Approximately quarterly
        quarter_id = s.dt.to_period("Q").astype("int64")
        quarter_steps = quarter_id.diff().dropna()
        scores["QS"] = (quarter_steps == 1).mean()

        # 7. Approximately yearly
        year_steps = s.dt.year.diff().dropna()
        scores["YS"] = (year_steps == 1).mean()

        best_freq = max(scores, key=scores.get)
        best_score = scores[best_freq]

        if best_score >= min_score:
            return best_freq, {
                "method": "tolerant",
                "score": float(best_score),
                "scores": {k: float(v) for k, v in scores.items()},
            }

        return "unknown", {
            "method": "tolerant",
            "score": float(best_score),
            "scores": {k: float(v) for k, v in scores.items()},
        }


    @staticmethod
    def resample_tolerant(df, date_col, min_score=0.75, agg="mean", freq=None):
        
        if freq is not None:
            freq_name = str(freq).strip().upper()
            supported = (
                freq_name
                in {
                    "D", "B", "W", "M", "MS", "ME", "Q", "QS", "QE",
                    "A", "Y", "YS", "YE",
                }
                or freq_name.startswith(
                    ("W-", "Q-", "QS-", "QE-", "A-", "Y-", "YS-", "YE-")
                )
            )
            if not supported:
                raise ValueError(f"Frequenza non supportata: {freq}")
        else:
            freq, _ = AnalyticalAgent.infer_freq_tolerant(
                df[date_col],
                min_score=min_score,
                prefer_business=True,
            )

        if freq == "unknown":
            raise ValueError(f"Frequenza non riconosciuta.")

        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).sort_values(date_col)
        out = out.set_index(date_col)

        if agg == "mean":
            resampled = out.resample(freq).mean(numeric_only=True)
        elif agg == "sum":
            resampled = out.resample(freq).sum(numeric_only=True)
        elif agg == "median":
            resampled = out.resample(freq).median(numeric_only=True)
        else:
            resampled = out.resample(freq).agg(agg)

        return resampled.reset_index(), freq


    @staticmethod
    def canonicalize_evaluation_partition(
        df: pd.DataFrame,
        *,
        freq: str,
        date_col: str = "date",
        agg: str = "mean",
        partition_name: str = "validation",
    ) -> pd.DataFrame:
        """Validate an evaluation partition without changing its observations."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        out = df.copy(deep=True)
        if date_col not in out.columns or "target" not in out.columns:
            raise ValueError(
                f"{partition_name} partition requires '{date_col}' and 'target' columns"
            )
        original_target = pd.to_numeric(out["target"], errors="coerce").to_numpy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        if out[date_col].isna().any():
            raise ValueError(f"{partition_name} contains invalid timestamps")
        if out["target"].isna().any():
            missing = int(out["target"].isna().sum())
            raise ValueError(
                f"{partition_name} contains {missing} missing target values; "
                "evaluation labels are never interpolated."
            )
        if not out[date_col].is_monotonic_increasing:
            raise ValueError(f"{partition_name} timestamps must be chronological")
        if out[date_col].duplicated().any():
            raise ValueError(f"{partition_name} timestamps must be unique")
        if not np.array_equal(
            original_target,
            out["target"].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise AssertionError(f"{partition_name} target was modified")
        out = out.reset_index(drop=True)
        out.attrs["evaluation_target_interpolation"] = "none"
        out.attrs["interpolated_target_count"] = 0
        out.attrs["interpolated_target_dates"] = []
        return out

    @staticmethod
    def _log_transform_diagnostics(values: np.ndarray) -> Dict[str, Any]:
        """Describe log1p suitability using training observations only."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {
                "detection_scope": "train_only",
                "log_transform_supported": False,
                "log_transform_recommended": False,
                "log_candidate_required": False,
                "strong_right_skew": False,
                "extreme_dynamic_range": False,
                "skewness": 0.0,
                "dynamic_range_ratio": None,
            }

        skewness = float(pd.Series(finite).skew()) if finite.size >= 3 else 0.0
        if not np.isfinite(skewness):
            skewness = 0.0
        positive = finite[finite > 0.0]
        positive_median = float(np.median(positive)) if positive.size else None
        dynamic_range_ratio = (
            float(np.max(finite) / positive_median)
            if positive_median is not None and positive_median > 0.0
            else None
        )
        supported = bool(np.min(finite) > -1.0)
        strong_right_skew = bool(skewness >= 1.5)
        extreme_dynamic_range = bool(
            dynamic_range_ratio is not None and dynamic_range_ratio >= 100.0
        )
        recommended = bool(
            supported and (strong_right_skew or extreme_dynamic_range)
        )
        return {
            "detection_scope": "train_only",
            "log_transform_supported": supported,
            "log_transform_recommended": recommended,
            "log_candidate_required": recommended,
            "strong_right_skew": strong_right_skew,
            "extreme_dynamic_range": extreme_dynamic_range,
            "skewness": skewness,
            "dynamic_range_ratio": dynamic_range_ratio,
            "minimum_training_target": float(np.min(finite)),
            "maximum_training_target": float(np.max(finite)),
        }

    def _compute_base_metrics(self, history: pd.DataFrame) -> tuple[Dict[str, Any], pd.DataFrame]:
        """Compute robust raw metrics and seasonality diagnostics."""

        try:
            df_resampled, inferred_freq = AnalyticalAgent.resample_tolerant(
                history,
                date_col="date",
                min_score=0.75,
                agg="mean",
            )
        except Exception:
            inferred_freq = "unknown"
            df_resampled = history.copy()

        df_resampled = df_resampled.copy()

        if "target" not in df_resampled.columns:
            raise ValueError("Missing required column: 'target'")

        df_resampled["target"] = pd.to_numeric(df_resampled["target"], errors="coerce")

        if df_resampled["target"].isna().any():
            df_resampled["target"] = (
                df_resampled["target"]
                .interpolate(limit_direction="both")
                .ffill()
                .bfill()
            )

        df_resampled = df_resampled.dropna(subset=["target"])

        values = df_resampled["target"].to_numpy(dtype=float)
        n = len(values)

        if n < 5:
            return {
                "n_obs": n,
                "time_range": "insufficient",
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "trend_strength": 0.0,
                "volatility_cv": 0.0,
                "autocorrelations": {},
                "delta_noise": 0.0,
                "inferred_freq": inferred_freq,
                "seasonal_periods": 1,
                "seasonality_detected": False,
                "seasonality_acf": 0.0,
                "seasonality_confidence": "low",
                "suggested_lag": 1,
                **self._log_transform_diagnostics(values),
            }, df_resampled

        date_col = df_resampled["date"] if "date" in df_resampled.columns else df_resampled.index

        try:
            time_range = f"{date_col.iloc[0]} to {date_col.iloc[-1]}"
        except Exception:
            time_range = f"{n} observations"

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        mean_abs = float(np.mean(np.abs(values)) + 1e-6)
        volatility_cv = float(std_val / mean_abs)

        t = np.arange(n, dtype=float)
        slope = float(np.polyfit(t, values, 1)[0])
        scale = float(std_val + 1e-6)
        trend_strength = float(abs(slope) * n / scale)

        if n >= 3 and std_val > 1e-12:
            trend = np.polyval(np.polyfit(t, values, 1), t)
            values_for_acf = values - trend
        else:
            values_for_acf = values.copy()

        def infer_frequency_config(freq: str, n_obs: int):
            freq_value = str(freq or "unknown").strip()

            try:
                offset = to_offset(freq_value)
                freq_name = offset.name.lower()
            except Exception:
                offset = None
                freq_name = freq_value.lower()

            step_seconds = None

            if offset is not None:
                try:
                    step_seconds = offset.nanos / 1e9
                except Exception:
                    step_seconds = None

            # Minute / sub-hourly data
            if "min" in freq_name or re.search(r"\d+t$", freq_name):
                if step_seconds and step_seconds > 0:
                    daily = max(1, int(round(86400 / step_seconds)))
                    weekly = max(1, int(round(7 * 86400 / step_seconds)))
                    return {
                        "lags_to_check": [daily, weekly],
                        "candidate_periods": [daily, weekly],
                        "default_period": daily,
                    }

            # Hourly data
            if "h" in freq_name:
                if step_seconds and step_seconds > 0:
                    daily = max(1, int(round(86400 / step_seconds)))
                    weekly = max(1, int(round(7 * 86400 / step_seconds)))
                else:
                    daily, weekly = 24, 168

                return {
                    "lags_to_check": [daily, weekly],
                    "candidate_periods": [daily, weekly],
                    "default_period": daily,
                }

            # Quarterly must be checked before daily because aliases like QE-DEC contain "d"
            if freq_name.startswith("q") or "qe" in freq_name or "qs" in freq_name:
                return {
                    "lags_to_check": [4, 8],
                    "candidate_periods": [4, 8],
                    "default_period": 4,
                }

            # Yearly must also be checked before daily because YE-DEC contains "d"
            if (
                freq_name.startswith("y")
                or freq_name.startswith("a")
                or "ye" in freq_name
                or "ys" in freq_name
            ):
                return {
                    "lags_to_check": [1, 2],
                    "candidate_periods": [1, 2],
                    "default_period": 1,
                }

            # Monthly
            if (
                freq_name.startswith("m")
                or "me" in freq_name
                or "ms" in freq_name
                or freq_name.endswith("m")
            ):
                return {
                    "lags_to_check": [6, 12, 24],
                    "candidate_periods": [6, 12, 24],
                    "default_period": 12,
                }

            # Weekly
            if freq_name.startswith("w"):
                return {
                    "lags_to_check": [4, 13, 26, 52],
                    "candidate_periods": [4, 13, 26, 52],
                    "default_period": 52,
                }

            # Daily / business daily
            if freq_name.startswith("d") or freq_name.startswith("b"):
                return {
                    "lags_to_check": [7, 14, 30, 90, 365],
                    "candidate_periods": [7, 14, 30, 90, 365],
                    "default_period": 7,
                }

            return {
                "lags_to_check": [7, 30],
                "candidate_periods": [7, 12, 24, 30, 52, 365],
                "default_period": 7,
            }

        config = infer_frequency_config(inferred_freq, n)

        lags_to_check = config["lags_to_check"]
        candidate_periods = config["candidate_periods"]
        default_period = config["default_period"]

        def safe_autocorr(x: np.ndarray, lag: int) -> float:
            if lag < 1 or lag >= len(x):
                return np.nan

            if np.nanstd(x) < 1e-12:
                return np.nan

            acf_val = self._autocorr(x, lag=lag)

            if acf_val is None or not np.isfinite(acf_val):
                return np.nan

            return float(acf_val)

        autocorrelations = {}

        for lag in lags_to_check:
            acf_val = safe_autocorr(values_for_acf, lag)
            autocorrelations[f"lag_{lag}"] = (
                round(acf_val, 6) if np.isfinite(acf_val) else None
            )

        diffs = np.diff(values)

        delta_noise = (
            float(np.std(diffs) / (std_val + 1e-6))
            if len(diffs) > 1
            else 0.0
        )

        max_seasonal_lag = min(n // 2, max(candidate_periods)) if n > 2 else 1

        # Significance threshold: combines practical and sample-size threshold.
        acf_threshold = 0.5

        best_lag = None
        best_acf = -np.inf

        for lag in candidate_periods:
            if 1 <= lag <= max_seasonal_lag:
                acf_val = safe_autocorr(values_for_acf, lag)

                if np.isfinite(acf_val) and acf_val >= acf_threshold and acf_val > best_acf:
                    best_lag = lag
                    best_acf = acf_val

        seasonality_detected = best_lag is not None

        frequency_default_window = int(default_period)
        if best_lag is None:
            usable_periods = [
                lag for lag in candidate_periods
                if 1 <= lag <= max_seasonal_lag
            ]

            if default_period <= max_seasonal_lag:
                best_lag = default_period
            elif usable_periods:
                best_lag = min(
                    usable_periods,
                    key=lambda lag: abs(lag - default_period),
                )
            else:
                best_lag = 1

            fallback_acf = safe_autocorr(values_for_acf, best_lag)
            best_acf = fallback_acf if np.isfinite(fallback_acf) else 0.0

        seasonal_periods = int(best_lag) if seasonality_detected else 1

        if seasonality_detected and best_acf >= 0.7:
            seasonality_confidence = "high"
        elif seasonality_detected and best_acf >= acf_threshold:
            seasonality_confidence = "medium"
        else:
            seasonality_confidence = "low"

        if n <= 2:
            suggested_lag = 1
        else:
            min_allowed_lag = 2

            max_allowed_lag = min(
                100,
                max(4, n // 4),
                n - 1,
            )

            max_allowed_lag = max(min_allowed_lag, max_allowed_lag)

            suggested_lag = (
                max(min_allowed_lag, seasonal_periods)
                if seasonality_detected
                else min_allowed_lag
            )

            max_check_lag = min(n // 3, 100, max_allowed_lag)
            acf_peak_threshold = max(0.20, 1.96 / np.sqrt(n))

            acf_values = {
                lag: safe_autocorr(values_for_acf, lag)
                for lag in range(1, max_check_lag + 1)
            }

            first_significant_peak = None

            for lag in range(2, max_check_lag):
                prev_acf = acf_values.get(lag - 1, np.nan)
                curr_acf = acf_values.get(lag, np.nan)
                next_acf = acf_values.get(lag + 1, np.nan)

                if not np.isfinite(curr_acf):
                    continue

                is_local_peak = (
                    np.isfinite(prev_acf)
                    and np.isfinite(next_acf)
                    and curr_acf >= prev_acf
                    and curr_acf >= next_acf
                )

                is_significant = curr_acf >= acf_peak_threshold

                if is_local_peak and is_significant:
                    first_significant_peak = lag
                    break

            if first_significant_peak is not None:
                suggested_lag = (
                    max(suggested_lag, first_significant_peak)
                    if seasonality_detected
                    else first_significant_peak
                )

            suggested_lag = int(
                max(
                    min_allowed_lag,
                    min(suggested_lag, max_allowed_lag),
                )
            )

        return {
            "n_obs": n,
            "time_range": time_range,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "trend_strength": trend_strength,
            "volatility_cv": volatility_cv,
            "autocorrelations": autocorrelations,
            "delta_noise": delta_noise,
            "inferred_freq": inferred_freq,
            "seasonal_periods": seasonal_periods,
            "seasonality_detected": bool(seasonality_detected),
            "seasonality_acf": round(float(best_acf), 6),
            "seasonality_confidence": seasonality_confidence,
            "frequency_default_window": frequency_default_window,
            "suggested_lag": suggested_lag,
            **self._log_transform_diagnostics(values),
        }, df_resampled

    

    @staticmethod
    def _detect_extreme_outliers(
        train_df: pd.DataFrame, config: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Detect robust extreme spikes using the training partition only."""
        return detect_extreme_outliers(train_df, config)

    def _autonomous_split(self, state: Dict[str, Any]) -> None:
        """Create the explicit chronological train/validation search partitions."""
        full_df = state.get("full_df")
        if full_df is None or "target" not in full_df.columns:
            return

        if "forecast_horizon" not in state:
            raise ValueError("AnalyticalAgent requires an explicit forecast_horizon.")
        forecast_horizon = int(state["forecast_horizon"])
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be a positive integer")

        if not bool(state.get("holdout_boundary_applied", False)):
            raise ValueError(
                "AnalyticalAgent requires a pre-separated TestHoldout; "
                "full_df must contain search data only."
            )
        pre_test_df = full_df.reset_index(drop=True)
        partition_sizes = state.get("partition_sizes", {})
        if not isinstance(partition_sizes, dict):
            raise ValueError("AnalyticalAgent requires explicit partition_sizes.")
        train_size = int(partition_sizes.get("train", 0))
        val_size = int(partition_sizes.get("validation", 0))
        test_size = int(partition_sizes.get("test", 0))
        total_size = len(pre_test_df) + test_size
        search_size = len(pre_test_df)
        if train_size + val_size != search_size:
            raise ValueError("Explicit train and validation sizes do not match search data.")
        minimum_train_size = SETTINGS.minimum_training_observations
        if train_size < minimum_train_size:
            raise ValueError(
                "Series is too short for the fixed T-FUSE training invariant: "
                f"search_size={search_size}, validation_size={val_size}, "
                f"minimum_train_size={minimum_train_size}."
            )
        if val_size % forecast_horizon or test_size % forecast_horizon:
            raise ValueError(
                "Validation and test sizes must be exact multiples of the "
                "forecast horizon."
            )
        effective_folds = val_size // forecast_horizon

        raw_train_df = pre_test_df.iloc[:train_size].reset_index(drop=True).copy(deep=True)
        raw_validation_df = pre_test_df.iloc[train_size:].reset_index(drop=True).copy(deep=True)
        train_df = raw_train_df.copy(deep=True)
        val_df = raw_validation_df.copy(deep=True)
        outlier_diagnostics = AnalyticalAgent._detect_extreme_outliers(
            raw_train_df, state.get("replace_outliers")
        )
        outlier_diagnostics["action"] = "detection_only_no_series_change"
        state["extreme_outliers_detected"] = outlier_diagnostics
        state["raw_train_df"] = raw_train_df
        state["raw_validation_df"] = raw_validation_df
        state["train_df"] = train_df
        state["val_df"] = val_df
        state["split_metadata"] = {
            "strategy": "explicit_chronological_split",
            "split_strategy": "chronological",
            "test_holdout_policy": "explicit_test_partition",
            "total_size": total_size,
            "filtered_n_observations": total_size,
            "train_size": len(train_df),
            "train_n_observations": len(train_df),
            "train_ratio_effective": len(train_df) / total_size,
            "validation_size": len(val_df),
            "validation_n_observations": len(val_df),
            "rolling_origin_count": effective_folds,
            "validation_ratio_effective": len(val_df) / total_size,
            "outlier_detection_scope": "train_only",
            "validation_target_modified": False,
            "validation_points_excluded_from_metrics": 0,
            "validation_size_before_outlier_adjustment": int(val_size),
            "outlier_split_action": outlier_diagnostics["action"],
            "test_size": test_size,
            "test_n_observations": test_size,
            "test_ratio_effective": test_size / total_size,
            "minimum_training_size": int(minimum_train_size),
            "actual_test_ratio": test_size / total_size if total_size else 0.0,
            "forecast_horizon": int(forecast_horizon),
            "validation_window_count": effective_folds,
            "split_uses_search_target_values": False,
            "validation_protocol": "rolling_origin_full_validation",
            "partition_window_alignment": "complete_forecast_horizon_windows",
            "validation_partial_window_size": 0,
            "test_partial_window_size": 0,
            "train_fingerprint": dataframe_fingerprint(train_df),
            "validation_fingerprint": dataframe_fingerprint(val_df),
            "test_holdout_fingerprint": str(
                state.get("holdout_metadata", {}).get("fingerprint", "")
            ),
            "train_start_date": str(train_df["date"].min()),
            "train_end_date": str(train_df["date"].max()),
            "validation_start_date": str(val_df["date"].min()),
            "validation_end_date": str(val_df["date"].max()),
            "train_end": str(train_df["date"].max()),
            "validation_start": str(val_df["date"].min()),
            "validation_end": str(val_df["date"].max()),
            "test_start_date": str(
                state.get("holdout_metadata", {}).get("start_date", "")
            ),
            "test_end_date": str(
                state.get("holdout_metadata", {}).get("end_date", "")
            ),
            "test_start": str(
                state.get("holdout_metadata", {}).get("start_date", "")
            ),
            "test_end": str(
                state.get("holdout_metadata", {}).get("end_date", "")
            ),
        }


    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run deterministic diagnostics and publish their measured output."""
        self._autonomous_split(state)

        series_df = state.get("train_df")
        if not isinstance(series_df, pd.DataFrame):
            raise ValueError("state['train_df'] must be a pandas DataFrame")
        step = int(state.get("step", 0))

        history = series_df.copy()

        if "target" not in history.columns and "value" in history.columns:
            history["target"] = history["value"]
        if "date" not in history.columns:
            history["date"] = history.index

        logger.info("AnalyticalAgent: computing deterministic diagnostics...")
        metrics, history = self._compute_base_metrics(history)

        validation = state.get("val_df")
        if isinstance(validation, pd.DataFrame) and not validation.empty:
            raw_validation = state.get("raw_validation_df")
            if not isinstance(raw_validation, pd.DataFrame):
                raise RuntimeError("raw_validation_df is required")
            validation = self.canonicalize_evaluation_partition(
                validation,
                freq=str(metrics.get("inferred_freq", "unknown")),
                partition_name="validation",
            )
            if not np.array_equal(
                pd.to_numeric(
                    raw_validation["target"], errors="coerce"
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    validation["target"], errors="coerce"
                ).to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise RuntimeError("Validation targets were modified")
            state["val_df"] = validation
            split_meta = dict(state.get("split_metadata", {}) or {})
            split_meta.update(
                {
                    "validation_size": len(validation),
                    "validation_fingerprint": dataframe_fingerprint(validation),
                    "validation_start_date": str(validation["date"].min()),
                    "validation_end_date": str(validation["date"].max()),
                    "canonical_frequency": metrics.get("inferred_freq", "unknown"),
                    "frequency_inferred_from": "train_only",
                    "evaluation_target_imputation": "none",
                    "evaluation_target_imputation_count": 0,
                    "evaluation_target_imputation_dates": [],
                    "validation_target_modified": False,
                    "validation_points_excluded_from_metrics": 0,
                }
            )
            state["split_metadata"] = split_meta

        state["train_df"] = history.copy()

        if "target_mase" not in state:
            state["target_mase"] = SETTINGS.finalize_mase_threshold

        result = {
            "base_metrics": metrics,
            "n_obs": metrics["n_obs"],
            "has_enough_history": metrics["n_obs"] >= 30,
            "trend_strength": metrics["trend_strength"],
            "volatility_cv": metrics["volatility_cv"],
            "autocorrelations": metrics.get("autocorrelations", {}),
            "delta_noise": metrics["delta_noise"],
            "inferred_freq": metrics["inferred_freq"],
            "seasonality_detected": bool(metrics.get("seasonality_detected", False)),
            "seasonality_acf": float(metrics.get("seasonality_acf", 0.0)),
            "seasonality_confidence": str(metrics.get("seasonality_confidence", "low")),
            "seasonal_periods": metrics.get("seasonal_periods"),
            "suggested_lag": metrics.get("suggested_lag"),
            "log_transform_recommended": bool(
                metrics.get("log_transform_recommended", False)
            ),
            "log_candidate_required": bool(
                metrics.get("log_candidate_required", False)
            ),
            "log_transform_supported": bool(
                metrics.get("log_transform_supported", False)
            ),
            "strong_right_skew": bool(
                metrics.get("strong_right_skew", False)
            ),
            "extreme_dynamic_range": bool(
                metrics.get("extreme_dynamic_range", False)
            ),
            "extreme_outliers_detected": state.get("extreme_outliers_detected", {"found": False}),
        }

        state["dataset_length"] = metrics["n_obs"]
        state["log_transform_recommended"] = bool(
            metrics.get("log_transform_recommended", False)
        )
        state["log_candidate_required"] = bool(
            metrics.get("log_candidate_required", False)
        )
        state["log_transform_supported"] = bool(
            metrics.get("log_transform_supported", False)
        )
        state["data_summary"] = result
        logger.info("AnalyticalAgent: deterministic diagnostics computed")
        return state
