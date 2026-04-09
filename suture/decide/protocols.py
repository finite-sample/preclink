"""Protocols for decision rules."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DecisionRule(Protocol):
    """Protocol for matching decision algorithms."""

    def decide(self, scored_pairs: pd.DataFrame) -> pd.DataFrame:
        """Select final matches from scored candidate pairs.

        Args:
            scored_pairs: DataFrame with candidate pairs and scores.
                Must contain 'left_index', 'right_index', and 'score' columns.

        Returns:
            DataFrame with selected matches.
        """
        ...
