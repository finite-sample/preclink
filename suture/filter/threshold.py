"""Threshold-based filtering."""

import pandas as pd


class ThresholdFilter:
    """Filter pairs below a minimum score threshold."""

    def __init__(self, min_score: float) -> None:
        """Initialize threshold filter.

        Args:
            min_score: Minimum acceptable score.
        """
        self.min_score = min_score

    def filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Remove pairs below the threshold.

        Args:
            pairs: DataFrame with 'score' column.

        Returns:
            Filtered DataFrame with pairs meeting threshold.
        """
        return pairs[pairs["score"] >= self.min_score].copy()
