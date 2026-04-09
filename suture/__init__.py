"""suture: High-precision record linkage library."""

from suture.block import Blocker, Crosswalk, FieldBlocker, FullBlocker
from suture.core import LinkageResult, PipelineBuilder
from suture.core import Pipeline as PipelineExecutor
from suture.decide import (
    DecisionRule,
    GreedyDecision,
    HungarianDecision,
    RowSequentialDecision,
)
from suture.deduplicate import (
    ClusterDeduplicator,
    DeduplicationReport,
    Deduplicator,
    ExactDeduplicator,
)
from suture.filter import Filter, MarginFilter, ThresholdFilter
from suture.inspect import InspectionReport, LinkageDiagnostics
from suture.multipass import MultiPassOrchestrator, PassConfig
from suture.preprocess import MissingHandler, Preprocessor, TextNormalizer
from suture.score import (
    Comparison,
    DateComparison,
    ExactComparison,
    NumericComparison,
    PairwiseScorer,
    StringComparison,
)

try:
    from suture._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Blocker",
    "ClusterDeduplicator",
    "Comparison",
    "Crosswalk",
    "DateComparison",
    "DecisionRule",
    "DeduplicationReport",
    "Deduplicator",
    "ExactComparison",
    "ExactDeduplicator",
    "FieldBlocker",
    "Filter",
    "FullBlocker",
    "GreedyDecision",
    "HungarianDecision",
    "InspectionReport",
    "LinkageDiagnostics",
    "LinkageResult",
    "MarginFilter",
    "MissingHandler",
    "MultiPassOrchestrator",
    "NumericComparison",
    "PairwiseScorer",
    "PassConfig",
    "Pipeline",
    "PipelineBuilder",
    "PipelineExecutor",
    "Preprocessor",
    "RowSequentialDecision",
    "StringComparison",
    "TextNormalizer",
    "ThresholdFilter",
]


def Pipeline() -> PipelineBuilder:  # noqa: N802
    """Create a new pipeline builder.

    Returns:
        PipelineBuilder for fluent pipeline construction.
    """
    return PipelineBuilder()
