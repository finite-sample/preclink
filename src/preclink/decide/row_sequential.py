"""Row-sequential matching algorithm."""

import pandas as pd

from preclink._typing import column, rows


class RowSequentialDecision:
    """Row-sequential matching processing left records in order.

    For each left record (in index order), selects the best available
    right record. Simple baseline algorithm.
    """

    def decide(self, scored_pairs: pd.DataFrame) -> pd.DataFrame:
        """Select matches processing left records sequentially.

        Args:
            scored_pairs: DataFrame with 'left_index', 'right_index', and 'score'.

        Returns:
            DataFrame with row-sequential matches.
        """
        if scored_pairs.empty:
            return scored_pairs

        left_indices = sorted(column(scored_pairs, "left_index").unique())

        matched_right: set[object] = set()
        matches = []

        for left_idx in left_indices:
            candidates = rows(
                scored_pairs,
                (column(scored_pairs, "left_index") == left_idx)
                & (~column(scored_pairs, "right_index").isin(list(matched_right))),
            )

            if not candidates.empty:
                best = candidates.loc[column(candidates, "score").idxmax()]
                matches.append(best)
                matched_right.add(best["right_index"])

        if not matches:
            return scored_pairs.iloc[:0]

        return pd.DataFrame(matches)
