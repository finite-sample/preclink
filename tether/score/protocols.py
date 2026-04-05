"""Protocols for comparison operations."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Comparison(Protocol):
    """Protocol for field comparison operations."""

    column: str
    weight: float

    def compare(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compare two series and return similarity scores.

        Args:
            left: Left series of values.
            right: Right series of values.

        Returns:
            Series of similarity scores between 0 and 1.
        """
        ...
