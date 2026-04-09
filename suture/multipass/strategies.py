"""Multi-pass matching strategies."""

from dataclasses import dataclass

from suture._typing import DecisionMethod


@dataclass(frozen=True)
class PassConfig:
    """Configuration for a single pass in multi-pass matching.

    Args:
        min_score: Minimum score threshold for this pass.
        method: Decision method for this pass.
        margin: Optional margin filter for this pass.
    """

    min_score: float
    method: DecisionMethod = "hungarian"
    margin: float | None = None


def strict_then_relaxed(
    strict_threshold: float = 0.95,
    medium_threshold: float = 0.85,
    relaxed_threshold: float = 0.70,
) -> list[PassConfig]:
    """Create a strict-then-relaxed multi-pass strategy.

    Args:
        strict_threshold: Threshold for first strict pass.
        medium_threshold: Threshold for medium pass.
        relaxed_threshold: Threshold for final relaxed pass.

    Returns:
        List of PassConfig for multi-pass matching.
    """
    return [
        PassConfig(min_score=strict_threshold, method="hungarian"),
        PassConfig(min_score=medium_threshold, method="hungarian"),
        PassConfig(min_score=relaxed_threshold, method="greedy"),
    ]


def precision_first(threshold: float = 0.90) -> list[PassConfig]:
    """Create a precision-first single-pass strategy.

    Args:
        threshold: High threshold for precision.

    Returns:
        Single PassConfig list for high-precision matching.
    """
    return [PassConfig(min_score=threshold, method="hungarian", margin=0.1)]
