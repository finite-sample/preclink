"""Scoring module for pairwise comparisons."""

from tether.score.comparisons import (
    DateComparison,
    ExactComparison,
    NumericComparison,
    StringComparison,
)
from tether.score.protocols import Comparison
from tether.score.scorer import PairwiseScorer

__all__ = [
    "Comparison",
    "DateComparison",
    "ExactComparison",
    "NumericComparison",
    "PairwiseScorer",
    "StringComparison",
]
