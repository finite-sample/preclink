"""Missing value handling."""

import pandas as pd

from tether._typing import MissingPolicy


class MissingHandler:
    """Handle missing values in DataFrames."""

    def __init__(
        self,
        policy: MissingPolicy = "skip",
        fill_value: str = "",
        columns: list[str] | None = None,
    ) -> None:
        """Initialize missing value handler.

        Args:
            policy: Missing value policy.
            fill_value: Fill value for 'fill' policy.
            columns: Columns to process.
        """
        self.policy = policy
        self.fill_value = fill_value
        self.columns = columns

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        result = df.copy()

        cols_to_process = self.columns if self.columns is not None else result.columns.tolist()

        if self.policy == "skip":
            pass
        elif self.policy == "zero":
            for col in cols_to_process:
                if col in result.columns:
                    result[col] = result[col].fillna(self.fill_value)
        elif self.policy == "penalize":
            pass

        return result
