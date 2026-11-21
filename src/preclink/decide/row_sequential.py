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

        left_ids = column(scored_pairs, "left_index")
        left_indices = sorted(left_ids.unique())
        positions = scored_pairs.groupby(left_ids, sort=False, observed=True).indices

        right_ids = column(scored_pairs, "right_index")
        matched_right: set[object] = set()
        matched_missing = False
        matches = []

        for left_idx in left_indices:
            group_positions = positions.get(left_idx, [])
            group = scored_pairs.iloc[group_positions]
            if matched_missing:
                # pandas' membership treatment of missing values can depend on
                # the length of the Series. Preserve the full-column result.
                available = (~right_ids.isin(list(matched_right))).iloc[group_positions]
            else:
                available = ~column(group, "right_index").isin(list(matched_right))
            candidates = rows(group, available)

            if not candidates.empty:
                best = candidates.loc[column(candidates, "score").idxmax()]
                matches.append(best)
                target = best["right_index"]
                matched_right.add(target)
                matched_missing = matched_missing or pd.isna(target)

        if not matches:
            return scored_pairs.iloc[:0]

        return pd.DataFrame(matches)
