"""tether: High-precision record linkage library."""

from tether.block import Blocker, Crosswalk, FieldBlocker, FullBlocker
from tether.core import LinkageResult, PipelineBuilder
from tether.core import Pipeline as PipelineExecutor
from tether.decide import (
    DecisionRule,
    GreedyDecision,
    HungarianDecision,
    RowSequentialDecision,
)
from tether.deduplicate import (
    ClusterDeduplicator,
    DeduplicationReport,
    Deduplicator,
    ExactDeduplicator,
)
from tether.filter import Filter, MarginFilter, ThresholdFilter
from tether.inspect import InspectionReport, LinkageDiagnostics
from tether.multipass import MultiPassOrchestrator, PassConfig
from tether.preprocess import MissingHandler, Preprocessor, TextNormalizer
from tether.score import (
    Comparison,
    DateComparison,
    ExactComparison,
    NumericComparison,
    PairwiseScorer,
    StringComparison,
)

try:
    from tether._version import __version__
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
