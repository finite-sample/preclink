"""Tests for multi-pass linkage."""

import pandas as pd
from suture import StringComparison
from suture.multipass.orchestrator import MultiPassOrchestrator
from suture.multipass.strategies import PassConfig, precision_first, strict_then_relaxed


class TestMultiPassOrchestrator:
    def test_basic_multipass(self, sample_left_df, sample_right_df):
        orchestrator = MultiPassOrchestrator()
        result = orchestrator.run(
            sample_left_df,
            sample_right_df,
            passes=[
                PassConfig(min_score=0.95, method="hungarian"),
                PassConfig(min_score=0.7, method="greedy"),
            ],
            comparisons=[
                StringComparison("first_name"),
                StringComparison("last_name"),
            ],
            block_on="state",
        )

        assert result.matches is not None
        assert result.diagnostics is not None

    def test_with_dict_config(self, sample_left_df, sample_right_df):
        orchestrator = MultiPassOrchestrator()
        result = orchestrator.run(
            sample_left_df,
            sample_right_df,
            passes=[
                {"min_score": 0.95, "method": "hungarian"},
                {"min_score": 0.7, "method": "greedy"},
            ],
            comparisons=[StringComparison("first_name")],
            block_on="state",
        )

        assert result is not None

    def test_removes_matched_between_passes(self):
        left = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "state": ["CA", "CA"],
            }
        )
        right = pd.DataFrame(
            {
                "name": ["Alice", "Bobby"],
                "state": ["CA", "CA"],
            }
        )

        orchestrator = MultiPassOrchestrator()
        result = orchestrator.run(
            left,
            right,
            passes=[
                PassConfig(min_score=0.99, method="hungarian"),
                PassConfig(min_score=0.7, method="greedy"),
            ],
            comparisons=[StringComparison("name")],
            block_on="state",
        )

        left_matched = set(result.matches["left_index"])
        assert len(left_matched) == len(result.matches)

    def test_no_blocking(self, sample_left_df, sample_right_df):
        orchestrator = MultiPassOrchestrator()
        result = orchestrator.run(
            sample_left_df,
            sample_right_df,
            passes=[PassConfig(min_score=0.7)],
            comparisons=[StringComparison("first_name")],
        )

        assert result is not None


class TestPassConfig:
    def test_defaults(self):
        config = PassConfig(min_score=0.8)
        assert config.method == "hungarian"
        assert config.margin is None

    def test_custom_values(self):
        config = PassConfig(min_score=0.7, method="greedy", margin=0.1)
        assert config.min_score == 0.7
        assert config.method == "greedy"
        assert config.margin == 0.1


class TestStrategies:
    def test_strict_then_relaxed(self):
        passes = strict_then_relaxed()
        assert len(passes) == 3
        assert passes[0].min_score > passes[1].min_score > passes[2].min_score

    def test_strict_then_relaxed_custom(self):
        passes = strict_then_relaxed(
            strict_threshold=0.99,
            medium_threshold=0.9,
            relaxed_threshold=0.8,
        )
        assert passes[0].min_score == 0.99
        assert passes[1].min_score == 0.9
        assert passes[2].min_score == 0.8

    def test_precision_first(self):
        passes = precision_first()
        assert len(passes) == 1
        assert passes[0].min_score == 0.9
        assert passes[0].margin == 0.1

    def test_precision_first_custom(self):
        passes = precision_first(threshold=0.95)
        assert passes[0].min_score == 0.95
