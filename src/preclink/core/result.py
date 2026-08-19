"""Result containers for linkage operations."""

from dataclasses import dataclass

import pandas as pd

from preclink._typing import select
from preclink.inspect.diagnostics import LinkageDiagnostics
from preclink.inspect.report import InspectionReport


@dataclass
class LinkageResult:
    """Container for linkage results.

    Args:
        matches: DataFrame with matched pairs.
        diagnostics: Linkage diagnostics.
        left: Original left DataFrame.
        right: Original right DataFrame.
        candidate_pairs: Candidate pairs after blocking.
        filtered_pairs: Pairs after filtering.
    """

    matches: pd.DataFrame
    diagnostics: LinkageDiagnostics
    left: pd.DataFrame
    right: pd.DataFrame
    candidate_pairs: pd.DataFrame
    filtered_pairs: pd.DataFrame

    def inspect(self, margin_threshold: float = 0.1) -> InspectionReport:
        """Generate an inspection report for this result.

        Args:
            margin_threshold: Threshold for identifying ambiguous pairs.

        Returns:
            InspectionReport with detailed analysis.
        """
        from preclink.inspect.report import generate_report

        return generate_report(
            left=self.left,
            right=self.right,
            matches=self.matches,
            diagnostics=self.diagnostics,
            filtered_pairs=self.filtered_pairs,
            margin_threshold=margin_threshold,
        )

    def merge_left(self, suffixes: tuple[str, str] = ("", "_matched")) -> pd.DataFrame:
        """Merge matches back to left DataFrame.

        Args:
            suffixes: Suffixes for overlapping columns.

        Returns:
            Left DataFrame with matched right columns.
        """
        if self.matches.empty:
            return self.left.copy()

        right_cols = [c for c in self.matches.columns if c.endswith("_right")]
        match_data = select(self.matches, ["left_index", *right_cols]).copy()
        match_data = match_data.rename(
            columns={c: c.replace("_right", suffixes[1]) for c in right_cols}
        )
        match_data = match_data.set_index("left_index")

        return self.left.join(match_data, how="left")

    def merge_right(self, suffixes: tuple[str, str] = ("_matched", "")) -> pd.DataFrame:
        """Merge matches back to right DataFrame.

        Args:
            suffixes: Suffixes for overlapping columns.

        Returns:
            Right DataFrame with matched left columns.
        """
        if self.matches.empty:
            return self.right.copy()

        left_cols = [c for c in self.matches.columns if c.endswith("_left")]
        match_data = select(self.matches, ["right_index", *left_cols]).copy()
        match_data = match_data.rename(
            columns={c: c.replace("_left", suffixes[0]) for c in left_cols}
        )
        match_data = match_data.set_index("right_index")

        return self.right.join(match_data, how="left")
