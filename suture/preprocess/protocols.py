"""Protocols for preprocessing operations."""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for preprocessing operations."""

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Preprocessed DataFrame.
        """
        ...
