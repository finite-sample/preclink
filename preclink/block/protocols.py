"""Protocols for blocking operations."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Blocker(Protocol):
    """Protocol for blocking strategies."""

    def block(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Generate candidate pairs from two DataFrames.

        Args:
            left: Left DataFrame.
            right: Right DataFrame.

        Returns:
            DataFrame with candidate pairs containing columns from both
            DataFrames with _left and _right suffixes.
        """
        ...
