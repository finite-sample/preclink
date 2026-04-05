"""Inspection report generation."""

from dataclasses import dataclass, field

import pandas as pd

from tether.inspect.diagnostics import LinkageDiagnostics


@dataclass
class InspectionReport:
    """Detailed inspection report for linkage results.

    Args:
        diagnostics: Linkage diagnostics.
        ambiguous_pairs: Pairs with close scores that may be ambiguous.
        unmatched_left: Left records that were not matched.
        unmatched_right: Right records that were not matched.
    """

    diagnostics: LinkageDiagnostics
    ambiguous_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_left: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched_right: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Generate a text summary of the report.

        Returns:
            Human-readable summary string.
        """
        lines = [
            "Linkage Report",
            "=" * 40,
            f"Left records:      {self.diagnostics.n_left:,}",
            f"Right records:     {self.diagnostics.n_right:,}",
            f"Candidate pairs:   {self.diagnostics.n_candidate_pairs:,}",
            f"Filtered pairs:    {self.diagnostics.n_filtered_pairs:,}",
            f"Final matches:     {self.diagnostics.n_matches:,}",
            f"Left match rate:   {self.diagnostics.match_rate_left:.1%}",
            f"Right match rate:  {self.diagnostics.match_rate_right:.1%}",
        ]

        if self.diagnostics.score_stats:
            lines.extend(
                [
                    "",
                    "Score Distribution",
                    "-" * 20,
                    f"  Min:    {self.diagnostics.score_stats['min']:.3f}",
                    f"  Max:    {self.diagnostics.score_stats['max']:.3f}",
                    f"  Mean:   {self.diagnostics.score_stats['mean']:.3f}",
                    f"  Median: {self.diagnostics.score_stats['median']:.3f}",
                ]
            )

        if not self.ambiguous_pairs.empty:
            lines.extend(
                [
                    "",
                    f"Ambiguous pairs: {len(self.ambiguous_pairs):,}",
                ]
            )

        return "\n".join(lines)


def generate_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    matches: pd.DataFrame,
    diagnostics: LinkageDiagnostics,
    filtered_pairs: pd.DataFrame,
    margin_threshold: float = 0.1,
) -> InspectionReport:
    """Generate an inspection report for linkage results.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        matches: Final matches.
        diagnostics: Linkage diagnostics.
        filtered_pairs: Pairs after filtering.
        margin_threshold: Threshold for identifying ambiguous pairs.

    Returns:
        InspectionReport with detailed analysis.
    """
    matched_left_indices = set(matches["left_index"]) if not matches.empty else set()
    matched_right_indices = set(matches["right_index"]) if not matches.empty else set()

    unmatched_left = left.loc[~left.index.isin(matched_left_indices)].copy()
    unmatched_right = right.loc[~right.index.isin(matched_right_indices)].copy()

    ambiguous = pd.DataFrame()
    if not filtered_pairs.empty and "score" in filtered_pairs.columns:
        ambiguous_indices = []
        for left_idx in filtered_pairs["left_index"].unique():
            left_pairs = filtered_pairs[filtered_pairs["left_index"] == left_idx]
            if len(left_pairs) >= 2:
                scores = left_pairs["score"].sort_values(ascending=False)
                if scores.iloc[0] - scores.iloc[1] < margin_threshold:
                    ambiguous_indices.extend(left_pairs.index.tolist())
        if ambiguous_indices:
            ambiguous = filtered_pairs.loc[ambiguous_indices].copy()

    return InspectionReport(
        diagnostics=diagnostics,
        ambiguous_pairs=ambiguous,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
    )
