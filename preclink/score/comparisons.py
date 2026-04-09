"""Comparison implementations for different data types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from rapidfuzz import distance as rf_distance

from preclink._typing import StringAlgorithm

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class StringComparison:
    """String similarity comparison using fuzzy matching algorithms.

    Args:
        column: Column name to compare.
        algorithm: Similarity algorithm to use.
        weight: Weight for this comparison in aggregate score.
    """

    column: str
    algorithm: StringAlgorithm = "jaro_winkler"
    weight: float = 1.0

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare string values using the configured algorithm.

        Args:
            left: Left series of string values.
            right: Right series of string values.

        Returns:
            Series of similarity scores between 0 and 1.
        """
        func: Callable[[Any, Any], float]
        if self.algorithm == "jaro_winkler":
            func = rf_distance.JaroWinkler.normalized_similarity
        elif self.algorithm == "levenshtein":
            func = rf_distance.Levenshtein.normalized_similarity
        elif self.algorithm == "damerau_levenshtein":
            func = rf_distance.DamerauLevenshtein.normalized_similarity
        else:
            msg = f"Unknown algorithm: {self.algorithm}"
            raise ValueError(msg)

        scores = []
        for l_val, r_val in zip(left, right, strict=True):
            if pd.isna(l_val) or pd.isna(r_val):
                scores.append(0.0)
            else:
                scores.append(func(str(l_val), str(r_val)))

        return pd.Series(scores, index=left.index)


@dataclass(frozen=True, slots=True)
class ExactComparison:
    """Exact match comparison.

    Args:
        column: Column name to compare.
        weight: Weight for this comparison in aggregate score.
    """

    column: str
    weight: float = 1.0

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare values for exact equality.

        Args:
            left: Left series of values.
            right: Right series of values.

        Returns:
            Series of 1.0 for matches, 0.0 for non-matches.
        """
        both_valid = ~(pd.isna(left) | pd.isna(right))
        matches = (left == right) & both_valid
        return matches.astype(float)


@dataclass(frozen=True, slots=True)
class NumericComparison:
    """Numeric comparison with tolerance.

    Args:
        column: Column name to compare.
        tolerance: Maximum allowed difference for a match.
        weight: Weight for this comparison in aggregate score.
    """

    column: str
    tolerance: float = 0.0
    weight: float = 1.0
    scale: Literal["linear", "gaussian"] = "linear"

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare numeric values with tolerance.

        Args:
            left: Left series of numeric values.
            right: Right series of numeric values.

        Returns:
            Series of similarity scores between 0 and 1.
        """
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")

        both_valid = ~(pd.isna(left_num) | pd.isna(right_num))
        diff = np.abs(left_num - right_num)

        if self.tolerance == 0:
            scores = (diff == 0).astype(float)
        elif self.scale == "linear":
            scores = np.maximum(0.0, 1.0 - diff / self.tolerance)
        else:
            sigma = self.tolerance / 2.0
            scores = np.exp(-0.5 * (diff / sigma) ** 2)

        scores_masked = scores.where(both_valid, 0.0)
        result: pd.Series[Any] = pd.Series(list(scores_masked), index=left.index)
        return result


@dataclass(frozen=True, slots=True)
class DateComparison:
    """Date comparison with day tolerance.

    Args:
        column: Column name to compare.
        tolerance_days: Maximum allowed difference in days.
        weight: Weight for this comparison in aggregate score.
    """

    column: str
    tolerance_days: int = 0
    weight: float = 1.0

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare date values with day tolerance.

        Args:
            left: Left series of date values.
            right: Right series of date values.

        Returns:
            Series of similarity scores between 0 and 1.
        """
        left_dt = pd.to_datetime(left, errors="coerce")
        right_dt = pd.to_datetime(right, errors="coerce")

        both_valid = ~(pd.isna(left_dt) | pd.isna(right_dt))
        diff_days = np.abs((left_dt - right_dt).dt.days)

        if self.tolerance_days == 0:
            scores = (diff_days == 0).astype(float)
        else:
            scores = np.maximum(0.0, 1.0 - diff_days / self.tolerance_days)

        scores_masked = scores.where(both_valid, 0.0)
        result: pd.Series[Any] = pd.Series(list(scores_masked), index=left.index)
        return result
