"""Blocking implementations."""

import pandas as pd

from tether.block.crosswalk import Crosswalk


class FieldBlocker:
    """Block on one or more fields with optional crosswalk."""

    def __init__(
        self,
        on: str | list[str],
        crosswalk: dict[str, str] | Crosswalk | None = None,
    ) -> None:
        """Initialize field blocker.

        Args:
            on: Field name(s) to block on.
            crosswalk: Optional crosswalk for value normalization.
        """
        self.on = [on] if isinstance(on, str) else on
        if crosswalk is None:
            self.crosswalk: Crosswalk | None = None
        elif isinstance(crosswalk, dict):
            self.crosswalk = Crosswalk(crosswalk)
        else:
            self.crosswalk = crosswalk

    def block(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Generate candidate pairs by blocking on specified fields.

        Args:
            left: Left DataFrame.
            right: Right DataFrame.

        Returns:
            DataFrame with candidate pairs.
        """
        left_copy = left.copy()
        right_copy = right.copy()

        left_copy["left_index"] = left_copy.index
        right_copy["right_index"] = right_copy.index

        for field in self.on:
            if self.crosswalk is not None:
                left_copy[f"_block_{field}"] = self.crosswalk.apply(left_copy[field])
                right_copy[f"_block_{field}"] = self.crosswalk.apply(right_copy[field])
            else:
                left_copy[f"_block_{field}"] = left_copy[field]
                right_copy[f"_block_{field}"] = right_copy[field]

        block_cols = [f"_block_{field}" for field in self.on]

        left_blocked = left_copy.dropna(subset=block_cols)
        right_blocked = right_copy.dropna(subset=block_cols)

        pairs = left_blocked.merge(right_blocked, on=block_cols, suffixes=("_left", "_right"))

        cols_to_drop = block_cols
        pairs = pairs.drop(columns=[c for c in cols_to_drop if c in pairs.columns])

        return pairs


class FullBlocker:
    """Generate all possible pairs (no blocking).

    Use with caution - creates n*m pairs.
    """

    def block(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Generate all possible pairs.

        Args:
            left: Left DataFrame.
            right: Right DataFrame.

        Returns:
            DataFrame with all pairs.
        """
        left_copy = left.copy()
        right_copy = right.copy()

        left_copy["left_index"] = left_copy.index
        right_copy["right_index"] = right_copy.index

        left_copy["_key"] = 1
        right_copy["_key"] = 1

        pairs = left_copy.merge(right_copy, on="_key", suffixes=("_left", "_right"))
        pairs = pairs.drop(columns=["_key"])

        return pairs
