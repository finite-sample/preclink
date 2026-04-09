"""Multi-pass matching orchestrator."""

from typing import TYPE_CHECKING

import pandas as pd

from preclink._typing import DecisionMethod
from preclink.block.blocker import FieldBlocker, FullBlocker
from preclink.core.result import LinkageResult
from preclink.decide.greedy import GreedyDecision
from preclink.decide.hungarian import HungarianDecision
from preclink.decide.row_sequential import RowSequentialDecision
from preclink.filter.margin import MarginFilter
from preclink.filter.threshold import ThresholdFilter
from preclink.inspect.diagnostics import compute_diagnostics
from preclink.multipass.strategies import PassConfig
from preclink.preprocess.normalizer import TextNormalizer
from preclink.score.scorer import PairwiseScorer

if TYPE_CHECKING:
    from preclink.block.crosswalk import Crosswalk
    from preclink.score.protocols import Comparison


class MultiPassOrchestrator:
    """Orchestrate multi-pass record linkage.

    Runs multiple passes with progressively relaxed thresholds,
    removing matched records between passes for higher precision.
    """

    def run(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        passes: list[PassConfig] | list[dict[str, float | str]],
        comparisons: list["Comparison"],
        block_on: str | list[str] | None = None,
        crosswalk: "Crosswalk | dict[str, str] | None" = None,
        preprocess: bool = True,
    ) -> LinkageResult:
        """Execute multi-pass linkage.

        Args:
            left: Left DataFrame to link.
            right: Right DataFrame to link.
            passes: List of pass configurations.
            comparisons: Comparison operations for scoring.
            block_on: Optional field(s) for blocking.
            crosswalk: Optional crosswalk mapping.
            preprocess: Whether to preprocess text columns.

        Returns:
            Combined LinkageResult from all passes.
        """
        pass_configs = self._normalize_passes(passes)

        left_remaining = left.copy()
        right_remaining = right.copy()

        if preprocess:
            normalizer = TextNormalizer()
            left_remaining = normalizer.preprocess(left_remaining)
            right_remaining = normalizer.preprocess(right_remaining)

        all_matches: list[pd.DataFrame] = []
        all_candidate_pairs = pd.DataFrame()
        all_filtered_pairs = pd.DataFrame()

        for pass_config in pass_configs:
            if left_remaining.empty or right_remaining.empty:
                break

            blocker: FieldBlocker | FullBlocker
            if block_on is not None:
                blocker = FieldBlocker(on=block_on, crosswalk=crosswalk)
            else:
                blocker = FullBlocker()

            candidate_pairs = blocker.block(left_remaining, right_remaining)

            if candidate_pairs.empty:
                continue

            if all_candidate_pairs.empty:
                all_candidate_pairs = candidate_pairs.copy()
            else:
                all_candidate_pairs = pd.concat(
                    [all_candidate_pairs, candidate_pairs], ignore_index=True
                )

            scorer = PairwiseScorer(comparisons)
            scored_pairs = scorer.score(candidate_pairs)

            filtered_pairs = scored_pairs.copy()

            if pass_config.min_score > 0:
                threshold_filter = ThresholdFilter(pass_config.min_score)
                filtered_pairs = threshold_filter.filter(filtered_pairs)

            if pass_config.margin is not None:
                margin_filter = MarginFilter(pass_config.margin)
                filtered_pairs = margin_filter.filter(filtered_pairs)

            if all_filtered_pairs.empty:
                all_filtered_pairs = filtered_pairs.copy()
            else:
                all_filtered_pairs = pd.concat(
                    [all_filtered_pairs, filtered_pairs], ignore_index=True
                )

            if filtered_pairs.empty:
                continue

            decision_rule = self._get_decision_rule(pass_config.method)
            matches = decision_rule.decide(filtered_pairs)

            if not matches.empty:
                all_matches.append(matches)

                matched_left = set(matches["left_index"])
                matched_right = set(matches["right_index"])

                left_remaining = left_remaining[~left_remaining.index.isin(matched_left)].copy()
                right_remaining = right_remaining[~right_remaining.index.isin(matched_right)].copy()

        final_matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

        diagnostics = compute_diagnostics(
            left=left,
            right=right,
            candidate_pairs=all_candidate_pairs,
            filtered_pairs=all_filtered_pairs,
            matches=final_matches,
        )

        return LinkageResult(
            matches=final_matches,
            diagnostics=diagnostics,
            left=left,
            right=right,
            candidate_pairs=all_candidate_pairs,
            filtered_pairs=all_filtered_pairs,
        )

    def _normalize_passes(
        self, passes: list[PassConfig] | list[dict[str, float | str]]
    ) -> list[PassConfig]:
        """Convert dict-based pass configs to PassConfig objects.

        Args:
            passes: List of PassConfig or dict configurations.

        Returns:
            List of PassConfig objects.
        """
        result = []
        for p in passes:
            if isinstance(p, PassConfig):
                result.append(p)
            else:
                result.append(
                    PassConfig(
                        min_score=float(p.get("min_score", 0.0)),
                        method=p.get("method", "hungarian"),  # type: ignore[arg-type]
                        margin=float(p["margin"]) if "margin" in p else None,
                    )
                )
        return result

    def _get_decision_rule(
        self, method: DecisionMethod
    ) -> HungarianDecision | GreedyDecision | RowSequentialDecision:
        """Get decision rule instance for method.

        Args:
            method: Decision method name.

        Returns:
            Decision rule instance.
        """
        if method == "hungarian":
            return HungarianDecision()
        elif method == "greedy":
            return GreedyDecision()
        else:
            return RowSequentialDecision()
