"""Pairwise scoring engine."""

from typing import TYPE_CHECKING

import pandas as pd

from preclink._typing import column
from preclink.score.comparisons import TFIDFStringComparison

if TYPE_CHECKING:
    from preclink.score.protocols import Comparison


class PairwiseScorer:
    """Compute pairwise similarity scores for candidate pairs."""

    def __init__(self, comparisons: list["Comparison"]) -> None:
        """Initialize the scorer with comparison operations.

        Args:
            comparisons: List of comparison operations.
        """
        self.comparisons = comparisons

    def _setup_idf_weights(self, pairs: pd.DataFrame) -> None:
        """Set up IDF weights for TFIDFStringComparison instances.

        Args:
            pairs: DataFrame with candidate pairs.
        """
        for comparison in self.comparisons:
            if isinstance(comparison, TFIDFStringComparison):
                left_col = f"{comparison.column}_left"
                right_col = f"{comparison.column}_right"
                if left_col in pairs.columns and right_col in pairs.columns:
                    comparison.set_idf_weights(
                        column(pairs, left_col), column(pairs, right_col)
                    )

    def score(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Compute similarity scores for candidate pairs.

        Args:
            pairs: DataFrame with candidate pairs containing columns from both
                left and right DataFrames with _left and _right suffixes.

        Returns:
            DataFrame with original pairs plus score columns and aggregate score.
        """
        self._setup_idf_weights(pairs)

        result = pairs.copy()
        total_weight = sum(c.weight for c in self.comparisons)

        weighted_sum = pd.Series(0.0, index=pairs.index)

        for comparison in self.comparisons:
            left_col = f"{comparison.column}_left"
            right_col = f"{comparison.column}_right"
            score_col = f"{comparison.column}_score"

            if left_col in pairs.columns and right_col in pairs.columns:
                scores = comparison.compare(
                    column(pairs, left_col), column(pairs, right_col)
                )
                result[score_col] = scores
                weighted_sum += scores * comparison.weight

        result["score"] = weighted_sum / total_weight if total_weight > 0 else 0.0
        return result
