"""Scoring module for pairwise comparisons."""

from preclink.score.comparisons import (
    DateComparison,
    ExactComparison,
    NumericComparison,
    StringComparison,
)
from preclink.score.protocols import Comparison
from preclink.score.scorer import PairwiseScorer

__all__ = [
    "Comparison",
    "DateComparison",
    "ExactComparison",
    "NumericComparison",
    "PairwiseScorer",
    "StringComparison",
]
