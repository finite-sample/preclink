"""preclink: High-precision record linkage library."""

from preclink.block import Blocker, Crosswalk, FieldBlocker, FullBlocker
from preclink.core import LinkageResult, PipelineBuilder
from preclink.core import Pipeline as PipelineExecutor
from preclink.decide import (
    DecisionRule,
    GreedyDecision,
    HungarianDecision,
    RowSequentialDecision,
)
from preclink.deduplicate import (
    ClusterDeduplicator,
    DeduplicationReport,
    Deduplicator,
    ExactDeduplicator,
)
from preclink.filter import Filter, MarginFilter, ThresholdFilter
from preclink.inspect import InspectionReport, LinkageDiagnostics
from preclink.multipass import MultiPassOrchestrator, PassConfig
from preclink.preprocess import MissingHandler, Preprocessor, TextNormalizer
from preclink.score import (
    Comparison,
    DateComparison,
    ExactComparison,
    NumericComparison,
    PairwiseScorer,
    StringComparison,
)

try:
    from preclink._version import __version__
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
