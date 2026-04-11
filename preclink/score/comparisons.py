"""Comparison implementations for different data types."""

from dataclasses import dataclass, field
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
        result: pd.Series = pd.Series(list(scores_masked), index=left.index)
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
        result: pd.Series = pd.Series(list(scores_masked), index=left.index)
        return result


@dataclass(slots=True)
class TFIDFStringComparison:
    """String comparison with TF-IDF weighting.

    Rare value matches provide stronger evidence than common value matches.
    For example, a match on "Jagmohan Trivikramji" is strong evidence of a
    true match because few people have that name, while a match on
    "John Smith" is weak evidence.

    The IDF weight is computed from both tables. The formula is
    ``idf(v) = log(N / df(v))`` where N is the total number of records
    and df(v) is the count of records containing value v. The final score
    is ``base_similarity * idf_weight``, normalized to the range [0, 1].

    Args:
        column: Column name to compare.
        algorithm: Similarity algorithm to use.
        weight: Weight for this comparison in aggregate score.
    """

    column: str
    algorithm: StringAlgorithm = "jaro_winkler"
    weight: float = 1.0
    _idf_weights: dict[str, float] = field(default_factory=dict, repr=False)
    _max_idf: float = field(default=1.0, repr=False)

    def set_idf_weights(self, left_values: pd.Series, right_values: pd.Series) -> None:
        """Compute IDF weights from both datasets.

        This should be called before compare() to set up the IDF weights.

        Args:
            left_values: All values from the left dataset column.
            right_values: All values from the right dataset column.
        """
        all_values = pd.concat([left_values, right_values], ignore_index=True)
        all_values = all_values.dropna().astype(str)

        n = len(all_values)
        if n == 0:
            self._idf_weights = {}
            self._max_idf = 1.0
            return

        value_counts = all_values.value_counts()
        idf_weights: dict[str, float] = {}
        for value, count in value_counts.items():
            idf_weights[str(value)] = float(np.log(n / count))

        self._idf_weights = idf_weights
        self._max_idf = max(idf_weights.values()) if idf_weights else 1.0

    def _get_idf_weight(self, value: str) -> float:
        """Get normalized IDF weight for a value.

        Args:
            value: The string value.

        Returns:
            Normalized IDF weight between 0 and 1.
        """
        if not self._idf_weights or self._max_idf == 0:
            return 1.0
        idf = self._idf_weights.get(value, self._max_idf)
        return idf / self._max_idf

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare string values with TF-IDF weighting.

        Args:
            left: Left series of string values.
            right: Right series of string values.

        Returns:
            Series of TF-IDF weighted similarity scores between 0 and 1.
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
                l_str, r_str = str(l_val), str(r_val)
                base_score = func(l_str, r_str)
                idf_left = self._get_idf_weight(l_str)
                idf_right = self._get_idf_weight(r_str)
                idf_weight = (idf_left + idf_right) / 2
                scores.append(base_score * idf_weight)

        return pd.Series(scores, index=left.index)
