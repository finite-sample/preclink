"""Clustering-based deduplication."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd

from preclink.block.blocker import FieldBlocker, FullBlocker
from preclink.score.scorer import PairwiseScorer

if TYPE_CHECKING:
    from preclink.score.protocols import Comparison


@dataclass
class DeduplicationReport:
    """Report on deduplication results."""

    original_count: int
    kept_count: int
    dropped_as_duplicate: int
    dropped_as_indistinguishable: int
    groups_found: int
    largest_group_size: int


class _UnionFind:
    """Union-Find data structure for connected components."""

    def __init__(self) -> None:
        self.parent: dict[object, object] = {}
        self.rank: dict[object, int] = {}

    def find(self, x: object) -> object:
        """Find root of element with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: object, y: object) -> None:
        """Union two elements by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_components(self) -> dict[object, list[object]]:
        """Return all connected components as root -> members mapping."""
        components: dict[object, list[object]] = {}
        for x in self.parent:
            root = self.find(x)
            if root not in components:
                components[root] = []
            components[root].append(x)
        return components


class ClusterDeduplicator:
    """Remove within-table duplicates using connected components."""

    def __init__(
        self,
        comparisons: list["Comparison"],
        threshold: float = 0.9,
        margin: float = 0.1,
        timestamp_column: str | None = None,
        block_on: str | list[str] | None = None,
    ) -> None:
        """Initialize cluster deduplicator.

        Args:
            comparisons: List of comparisons for scoring.
            threshold: Minimum score for duplicate detection.
            margin: Minimum quality difference to declare a winner.
            timestamp_column: Column name for recency bonus (newer is better).
            block_on: Field(s) to block on to avoid n² pairs.
        """
        self.comparisons = comparisons
        self.threshold = threshold
        self.margin = margin
        self.timestamp_column = timestamp_column
        if block_on is None:
            self.block_on: list[str] | None = None
        elif isinstance(block_on, str):
            self.block_on = [block_on]
        else:
            self.block_on = block_on

    def _compute_quality(self, df: pd.DataFrame) -> pd.Series:
        """Compute quality score for each record based on completeness."""
        comparison_columns = [c.column for c in self.comparisons]
        relevant_cols = [c for c in comparison_columns if c in df.columns]

        if not relevant_cols:
            return pd.Series(1.0, index=df.index)

        completeness = df[relevant_cols].notna().sum(axis=1) / len(relevant_cols)

        if self.timestamp_column and self.timestamp_column in df.columns:
            ts = pd.to_datetime(df[self.timestamp_column], errors="coerce")
            ts_min = ts.min()
            ts_max = ts.max()
            if ts_min != ts_max:
                recency = (ts - ts_min) / (ts_max - ts_min)
                recency = recency.fillna(0)
            else:
                recency = pd.Series(0.0, index=df.index)
            return completeness * (1 + 0.1 * recency)

        return completeness

    def deduplicate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DeduplicationReport]:
        """Remove duplicate records using connected components.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated DataFrame, deduplication report).
        """
        original_count = len(df)

        if len(df) <= 1:
            report = DeduplicationReport(
                original_count=original_count,
                kept_count=len(df),
                dropped_as_duplicate=0,
                dropped_as_indistinguishable=0,
                groups_found=len(df),
                largest_group_size=1 if len(df) == 1 else 0,
            )
            return df.copy(), report

        blocker = FieldBlocker(on=self.block_on) if self.block_on else FullBlocker()

        pairs = blocker.block(df, df)
        pairs = pairs[pairs["left_index"] < pairs["right_index"]]

        if pairs.empty:
            report = DeduplicationReport(
                original_count=original_count,
                kept_count=len(df),
                dropped_as_duplicate=0,
                dropped_as_indistinguishable=0,
                groups_found=len(df),
                largest_group_size=1,
            )
            return df.copy(), report

        scorer = PairwiseScorer(self.comparisons)
        scored = scorer.score(pairs)
        duplicates = scored[scored["score"] >= self.threshold]

        uf = _UnionFind()
        for idx in df.index:
            uf.find(idx)

        for _, row in duplicates.iterrows():
            uf.union(row["left_index"], row["right_index"])

        components = uf.get_components()

        quality = self._compute_quality(df)

        keep_indices: list[object] = []
        dropped_as_duplicate = 0
        dropped_as_indistinguishable = 0
        groups_found = 0
        largest_group_size = 0

        for members in components.values():
            size = len(members)
            largest_group_size = max(largest_group_size, size)

            if size == 1:
                keep_indices.append(members[0])
            else:
                groups_found += 1
                member_quality = quality[quality.index.isin(members)]
                scores = list(zip(member_quality.index, member_quality.values, strict=True))
                scores.sort(key=lambda x: x[1], reverse=True)

                best_idx, best_score = scores[0]
                second_score = scores[1][1]

                if best_score - second_score >= self.margin:
                    keep_indices.append(best_idx)
                    dropped_as_duplicate += size - 1
                else:
                    dropped_as_indistinguishable += size

        result_df = df.loc[keep_indices].copy()

        report = DeduplicationReport(
            original_count=original_count,
            kept_count=len(result_df),
            dropped_as_duplicate=dropped_as_duplicate,
            dropped_as_indistinguishable=dropped_as_indistinguishable,
            groups_found=groups_found,
            largest_group_size=largest_group_size,
        )

        return result_df, report


class ExactDeduplicator:
    """Remove exact duplicates based on specified columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        keep: Literal["first", "last"] = "first",
    ) -> None:
        """Initialize exact deduplicator.

        Args:
            columns: Columns to match on.
            keep: Which duplicate to keep.
        """
        self.columns = columns
        self.keep: Literal["first", "last"] = keep

    def deduplicate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DeduplicationReport]:
        """Remove exact duplicates.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (deduplicated DataFrame, deduplication report).
        """
        original_count = len(df)
        result = df.drop_duplicates(subset=self.columns, keep=self.keep)
        dropped = original_count - len(result)

        report = DeduplicationReport(
            original_count=original_count,
            kept_count=len(result),
            dropped_as_duplicate=dropped,
            dropped_as_indistinguishable=0,
            groups_found=0,
            largest_group_size=0,
        )
        return result, report
