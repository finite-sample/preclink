"""Margin-based ambiguity filtering."""

import pandas as pd


class MarginFilter:
    """Remove ambiguous matches based on score margin.

    For each left record, removes the best match if the margin to the
    second-best match is below the threshold.
    """

    def __init__(self, margin: float) -> None:
        """Initialize margin filter.

        Args:
            margin: Minimum acceptable margin.
        """
        self.margin = margin

    def filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Remove ambiguous matches.

        Args:
            pairs: DataFrame with 'left_index' and 'score' columns.

        Returns:
            Filtered DataFrame with unambiguous matches.
        """
        if pairs.empty:
            return pairs

        keep_indices = []

        for left_idx in pairs["left_index"].unique():
            left_pairs = pairs[pairs["left_index"] == left_idx].sort_values(
                "score", ascending=False
            )

            if len(left_pairs) == 1:
                keep_indices.append(left_pairs.index[0])
            elif len(left_pairs) >= 2:
                best_score = left_pairs["score"].iloc[0]
                second_score = left_pairs["score"].iloc[1]

                if best_score - second_score >= self.margin:
                    keep_indices.append(left_pairs.index[0])

        return pairs.loc[keep_indices].copy()
