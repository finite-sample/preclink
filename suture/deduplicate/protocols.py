"""Protocols for deduplication operations."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from suture.deduplicate.cluster import DeduplicationReport


@runtime_checkable
class Deduplicator(Protocol):
    """Protocol for within-table deduplication."""

    def deduplicate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, "DeduplicationReport"]:
        """Remove duplicate records from a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated DataFrame, deduplication report).
        """
        ...
