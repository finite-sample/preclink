"""Protocols for filtering operations."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Filter(Protocol):
    """Protocol for pair filtering strategies."""

    def filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Filter candidate pairs.

        Args:
            pairs: DataFrame with candidate pairs and scores.

        Returns:
            Filtered DataFrame.
        """
        ...
