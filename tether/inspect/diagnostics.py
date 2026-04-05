"""Diagnostic utilities for linkage inspection."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class LinkageDiagnostics:
    """Diagnostic statistics for linkage results.

    Args:
        n_left: Number of records in left DataFrame.
        n_right: Number of records in right DataFrame.
        n_candidate_pairs: Number of candidate pairs after blocking.
        n_filtered_pairs: Number of pairs after filtering.
        n_matches: Number of final matches.
        match_rate_left: Proportion of left records matched.
        match_rate_right: Proportion of right records matched.
        score_stats: Score distribution statistics.
    """

    n_left: int
    n_right: int
    n_candidate_pairs: int
    n_filtered_pairs: int
    n_matches: int
    match_rate_left: float
    match_rate_right: float
    score_stats: dict[str, float]


def compute_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    filtered_pairs: pd.DataFrame,
    matches: pd.DataFrame,
) -> LinkageDiagnostics:
    """Compute diagnostic statistics for linkage results.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        candidate_pairs: Candidate pairs after blocking.
        filtered_pairs: Pairs after filtering.
        matches: Final matches.

    Returns:
        LinkageDiagnostics with computed statistics.
    """
    n_left = len(left)
    n_right = len(right)

    n_matched_left = matches["left_index"].nunique() if not matches.empty else 0
    n_matched_right = matches["right_index"].nunique() if not matches.empty else 0

    score_stats: dict[str, float] = {}
    if not matches.empty and "score" in matches.columns:
        score_stats = {
            "min": float(matches["score"].min()),
            "max": float(matches["score"].max()),
            "mean": float(matches["score"].mean()),
            "median": float(matches["score"].median()),
            "std": float(matches["score"].std()),
        }

    return LinkageDiagnostics(
        n_left=n_left,
        n_right=n_right,
        n_candidate_pairs=len(candidate_pairs),
        n_filtered_pairs=len(filtered_pairs),
        n_matches=len(matches),
        match_rate_left=n_matched_left / n_left if n_left > 0 else 0.0,
        match_rate_right=n_matched_right / n_right if n_right > 0 else 0.0,
        score_stats=score_stats,
    )
