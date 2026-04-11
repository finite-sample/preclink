"""Hungarian algorithm for optimal assignment."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class HungarianDecision:
    """Optimal assignment using maximum-weight bipartite matching.

    Maximizes total matching score while ensuring each record is matched
    at most once.

    Implementation note:
        Uses scipy.optimize.linear_sum_assignment which internally implements
        the Jonker-Volgenant algorithm (since scipy 1.4.0), an improvement
        over the original Hungarian algorithm with better worst-case complexity.
        See: https://doi.org/10.1007/BF02278710
    """

    def decide(self, scored_pairs: pd.DataFrame) -> pd.DataFrame:
        """Select optimal matches using Hungarian algorithm.

        Args:
            scored_pairs: DataFrame with 'left_index', 'right_index', and 'score'.

        Returns:
            DataFrame with optimal matches.
        """
        if scored_pairs.empty:
            return scored_pairs

        left_indices = scored_pairs["left_index"].unique()
        right_indices = scored_pairs["right_index"].unique()

        left_map = {v: i for i, v in enumerate(left_indices)}
        right_map = {v: i for i, v in enumerate(right_indices)}

        n_left = len(left_indices)
        n_right = len(right_indices)

        if n_left == 0 or n_right == 0:
            return scored_pairs.iloc[:0]

        large_cost = 1e10
        cost_matrix = np.full((n_left, n_right), large_cost)

        for _, row in scored_pairs.iterrows():
            i = left_map[row["left_index"]]
            j = right_map[row["right_index"]]
            cost_matrix[i, j] = 1.0 - row["score"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind, strict=True):
            if cost_matrix[i, j] < large_cost:
                left_idx = left_indices[i]
                right_idx = right_indices[j]
                match_row = scored_pairs[
                    (scored_pairs["left_index"] == left_idx)
                    & (scored_pairs["right_index"] == right_idx)
                ]
                if not match_row.empty:
                    matches.append(match_row.iloc[0])

        if not matches:
            return scored_pairs.iloc[:0]

        return pd.DataFrame(matches)
