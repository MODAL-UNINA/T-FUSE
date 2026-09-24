"""Hard boundary between online search and the final test holdout."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Mapping

import pandas as pd


FORBIDDEN_SEARCH_KEYS = frozenset(
    {
        "test_df",
        "raw_test_df",
        "preprocessed_test_df",
        "test_metrics",
        "test_score",
        "test_predictions",
        "y_true_test",
        "selected_model_test_rank",
        "test_regret",
    }
)


class TestLeakageError(RuntimeError):
    """Raised when an online component attempts to access final-test data."""

    __test__ = False


class SearchState(dict[str, Any]):
    """Runtime state that makes test fields structurally inaccessible."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SearchState":
        return cls(
            {
                key: value
                for key, value in values.items()
                if key not in FORBIDDEN_SEARCH_KEYS
            }
        )

    def _guard(self, key: object) -> None:
        if key in FORBIDDEN_SEARCH_KEYS:
            raise TestLeakageError(
                f"Search-time access to held-out field {key!r} is forbidden"
            )

    def __getitem__(self, key: str) -> Any:
        self._guard(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self._guard(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self._guard(key)
        return super().__contains__(key)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


@dataclass(frozen=True)
class TestHoldout:
    """Private chronological holdout revealed only to FinalEvaluator."""

    _frame: pd.DataFrame = field(repr=False, compare=False)
    fingerprint: str
    start_date: str
    end_date: str
    size: int

    @classmethod
    def separate(
        cls,
        full_df: pd.DataFrame,
        *,
        test_size: int,
    ) -> tuple[pd.DataFrame, "TestHoldout"]:
        if not isinstance(full_df, pd.DataFrame) or full_df.empty:
            raise ValueError("A non-empty full_df is required")
        if "target" not in full_df or "date" not in full_df:
            raise ValueError("full_df must contain date and target columns")
        size = int(test_size)
        if size < 1:
            raise ValueError("test_size must be a positive integer")
        if len(full_df) <= size + 1:
            raise ValueError(
                "Series is too short for the requested test holdout"
            )
        search_frame = full_df.iloc[:-size].reset_index(drop=True).copy(deep=True)
        test_frame = full_df.iloc[-size:].reset_index(drop=True).copy(deep=True)
        holdout = cls(
            _frame=test_frame,
            fingerprint=_frame_fingerprint(test_frame),
            start_date=str(test_frame["date"].min()),
            end_date=str(test_frame["date"].max()),
            size=len(test_frame),
        )
        return search_frame, holdout

    def _copy_for_final_evaluator(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


@dataclass
class FinalEvaluator:
    """One-shot gate that freezes the winner before exposing the holdout."""

    holdout: TestHoldout
    winning_candidate_id: str
    evaluation_count: int = 0

    def evaluate(
        self,
        evaluator: Callable[[pd.DataFrame], Any],
        *,
        winning_candidate_id: str,
    ) -> Any:
        if winning_candidate_id != self.winning_candidate_id:
            raise TestLeakageError("Winner changed after the final-evaluation freeze")
        if self.evaluation_count:
            raise TestLeakageError("The final test holdout may be evaluated only once")
        self.evaluation_count += 1
        result = evaluator(self.holdout._copy_for_final_evaluator())
        if winning_candidate_id != self.winning_candidate_id:
            raise TestLeakageError("Final evaluation modified winning_candidate_id")
        return result

    def acquire_once(self, *, winning_candidate_id: str) -> pd.DataFrame:
        """Acquire a protected copy for the existing final-evaluation pipeline."""
        return self.evaluate(
            lambda frame: frame, winning_candidate_id=winning_candidate_id
        )


__all__ = [
    "FORBIDDEN_SEARCH_KEYS",
    "FinalEvaluator",
    "SearchState",
    "TestHoldout",
    "TestLeakageError",
]
