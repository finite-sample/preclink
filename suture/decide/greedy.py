"""Greedy matching algorithm."""

import pandas as pd


class GreedyDecision:
    """Greedy matching selecting best global pair first.

    Iteratively selects the highest-scoring unmatched pair until no
    valid pairs remain.
    """

    def decide(self, scored_pairs: pd.DataFrame) -> pd.DataFrame:
        """Select matches using greedy best-first approach.

        Args:
            scored_pairs: DataFrame with 'left_index', 'right_index', and 'score'.

        Returns:
            DataFrame with greedy matches.
        """
        if scored_pairs.empty:
            return scored_pairs

        sorted_pairs = scored_pairs.sort_values("score", ascending=False)

        matched_left: set[object] = set()
        matched_right: set[object] = set()
        matches = []

        for _, row in sorted_pairs.iterrows():
            left_idx = row["left_index"]
            right_idx = row["right_index"]

            if left_idx not in matched_left and right_idx not in matched_right:
                matches.append(row)
                matched_left.add(left_idx)
                matched_right.add(right_idx)

        if not matches:
            return scored_pairs.iloc[:0]

        return pd.DataFrame(matches)
