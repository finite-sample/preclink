"""Scoring module for pairwise comparisons."""

from suture.score.comparisons import (
    DateComparison,
    ExactComparison,
    NumericComparison,
    StringComparison,
)
from suture.score.protocols import Comparison
from suture.score.scorer import PairwiseScorer

__all__ = [
    "Comparison",
    "DateComparison",
    "ExactComparison",
    "NumericComparison",
    "PairwiseScorer",
    "StringComparison",
]
