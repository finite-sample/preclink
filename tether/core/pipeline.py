"""Pipeline builder and executor."""

from typing import TYPE_CHECKING, Self

import pandas as pd

from tether._typing import DecisionMethod
from tether.block.blocker import FieldBlocker, FullBlocker
from tether.core.config import (
    BlockConfig,
    DecideConfig,
    FilterConfig,
    PreprocessConfig,
    ScoreConfig,
)
from tether.core.result import LinkageResult
from tether.decide.greedy import GreedyDecision
from tether.decide.hungarian import HungarianDecision
from tether.decide.row_sequential import RowSequentialDecision
from tether.filter.margin import MarginFilter
from tether.filter.threshold import ThresholdFilter
from tether.inspect.diagnostics import compute_diagnostics
from tether.preprocess.normalizer import TextNormalizer
from tether.score.scorer import PairwiseScorer

if TYPE_CHECKING:
    from tether.block.crosswalk import Crosswalk
    from tether.score.protocols import Comparison


class PipelineBuilder:
    """Fluent builder for constructing linkage pipelines."""

    def __init__(self) -> None:
        """Initialize an empty pipeline builder."""
        self._preprocess_config: PreprocessConfig | None = None
        self._block_config: BlockConfig | None = None
        self._score_config: ScoreConfig | None = None
        self._filter_config: FilterConfig | None = None
        self._decide_config: DecideConfig | None = None

    def preprocess(
        self,
        normalize_unicode: bool = True,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        collapse_whitespace: bool = True,
        columns: list[str] | None = None,
    ) -> Self:
        """Configure preprocessing stage.

        Args:
            normalize_unicode: Normalize unicode characters.
            lowercase: Convert to lowercase.
            strip_whitespace: Strip whitespace.
            collapse_whitespace: Collapse multiple whitespace.
            columns: Columns to preprocess.

        Returns:
            Self for method chaining.
        """
        self._preprocess_config = PreprocessConfig(
            normalize_unicode=normalize_unicode,
            lowercase=lowercase,
            strip_whitespace=strip_whitespace,
            collapse_whitespace=collapse_whitespace,
            columns=columns,
        )
        return self

    def block(
        self,
        on: str | list[str],
        crosswalk: "Crosswalk | dict[str, str] | None" = None,
    ) -> Self:
        """Configure blocking stage.

        Args:
            on: Field(s) to block on.
            crosswalk: Optional crosswalk mapping.

        Returns:
            Self for method chaining.
        """
        self._block_config = BlockConfig(on=on, crosswalk=crosswalk)
        return self

    def score(self, comparisons: list["Comparison"]) -> Self:
        """Configure scoring stage.

        Args:
            comparisons: List of comparison operations.

        Returns:
            Self for method chaining.
        """
        self._score_config = ScoreConfig(comparisons=comparisons)
        return self

    def filter(self, min_score: float = 0.0, margin: float | None = None) -> Self:
        """Configure filtering stage.

        Args:
            min_score: Minimum score threshold.
            margin: Minimum margin for ambiguity removal.

        Returns:
            Self for method chaining.
        """
        self._filter_config = FilterConfig(min_score=min_score, margin=margin)
        return self

    def decide(self, method: DecisionMethod = "hungarian") -> Self:
        """Configure decision stage.

        Args:
            method: Decision algorithm to use.

        Returns:
            Self for method chaining.
        """
        self._decide_config = DecideConfig(method=method)
        return self

    def build(self) -> "Pipeline":
        """Build the configured pipeline.

        Returns:
            Configured Pipeline instance.

        Raises:
            ValueError: If score configuration is missing.
        """
        if self._score_config is None:
            msg = "Score configuration with comparisons is required"
            raise ValueError(msg)

        return Pipeline(
            preprocess_config=self._preprocess_config,
            block_config=self._block_config,
            score_config=self._score_config,
            filter_config=self._filter_config or FilterConfig(),
            decide_config=self._decide_config or DecideConfig(),
        )


class Pipeline:
    """Executable linkage pipeline."""

    def __init__(
        self,
        preprocess_config: PreprocessConfig | None,
        block_config: BlockConfig | None,
        score_config: ScoreConfig,
        filter_config: FilterConfig,
        decide_config: DecideConfig,
    ) -> None:
        """Initialize pipeline with configurations.

        Args:
            preprocess_config: Preprocessing configuration.
            block_config: Blocking configuration.
            score_config: Scoring configuration.
            filter_config: Filtering configuration.
            decide_config: Decision configuration.
        """
        self.preprocess_config = preprocess_config
        self.block_config = block_config
        self.score_config = score_config
        self.filter_config = filter_config
        self.decide_config = decide_config

    def link(self, left: pd.DataFrame, right: pd.DataFrame) -> LinkageResult:
        """Execute the linkage pipeline.

        Args:
            left: Left DataFrame to link.
            right: Right DataFrame to link.

        Returns:
            LinkageResult with matches and diagnostics.
        """
        left_processed = left.copy()
        right_processed = right.copy()

        if self.preprocess_config is not None:
            normalizer = TextNormalizer(
                normalize_unicode=self.preprocess_config.normalize_unicode,
                lowercase=self.preprocess_config.lowercase,
                strip_whitespace=self.preprocess_config.strip_whitespace,
                collapse_whitespace=self.preprocess_config.collapse_whitespace,
                columns=self.preprocess_config.columns,
            )
            left_processed = normalizer.preprocess(left_processed)
            right_processed = normalizer.preprocess(right_processed)

        blocker: FieldBlocker | FullBlocker
        if self.block_config is not None:
            blocker = FieldBlocker(
                on=self.block_config.on,
                crosswalk=self.block_config.crosswalk,
            )
        else:
            blocker = FullBlocker()

        candidate_pairs = blocker.block(left_processed, right_processed)

        scorer = PairwiseScorer(self.score_config.comparisons)
        scored_pairs = scorer.score(candidate_pairs)

        filtered_pairs = scored_pairs.copy()

        if self.filter_config.min_score > 0:
            threshold_filter = ThresholdFilter(self.filter_config.min_score)
            filtered_pairs = threshold_filter.filter(filtered_pairs)

        if self.filter_config.margin is not None:
            margin_filter = MarginFilter(self.filter_config.margin)
            filtered_pairs = margin_filter.filter(filtered_pairs)

        decision_rule: HungarianDecision | GreedyDecision | RowSequentialDecision
        if self.decide_config.method == "hungarian":
            decision_rule = HungarianDecision()
        elif self.decide_config.method == "greedy":
            decision_rule = GreedyDecision()
        else:
            decision_rule = RowSequentialDecision()

        matches = decision_rule.decide(filtered_pairs)

        diagnostics = compute_diagnostics(
            left=left,
            right=right,
            candidate_pairs=candidate_pairs,
            filtered_pairs=filtered_pairs,
            matches=matches,
        )

        return LinkageResult(
            matches=matches,
            diagnostics=diagnostics,
            left=left,
            right=right,
            candidate_pairs=candidate_pairs,
            filtered_pairs=filtered_pairs,
        )
