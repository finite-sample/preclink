"""Tests for the pipeline builder and executor."""

import pandas as pd
import pytest
from tether import ExactComparison, Pipeline, StringComparison
from tether.core.pipeline import PipelineBuilder
from tether.core.result import LinkageResult


class TestPipelineBuilder:
    def test_fluent_interface(self):
        builder = PipelineBuilder()
        result = (
            builder.preprocess(lowercase=True)
            .block(on="state")
            .score(comparisons=[StringComparison("name")])
            .filter(min_score=0.5)
            .decide(method="hungarian")
        )
        assert result is builder

    def test_build_requires_score(self):
        builder = PipelineBuilder()
        with pytest.raises(ValueError, match="Score configuration"):
            builder.build()

    def test_build_with_minimum_config(self):
        pipeline = PipelineBuilder().score(comparisons=[StringComparison("name")]).build()
        assert pipeline is not None


class TestPipeline:
    def test_basic_linkage(self, sample_left_df, sample_right_df):
        result = (
            Pipeline()
            .preprocess(lowercase=True)
            .block(on="state")
            .score(
                comparisons=[
                    StringComparison("first_name"),
                    StringComparison("last_name"),
                ]
            )
            .filter(min_score=0.7)
            .decide(method="hungarian")
            .build()
            .link(sample_left_df, sample_right_df)
        )

        assert isinstance(result, LinkageResult)
        assert isinstance(result.matches, pd.DataFrame)
        assert result.diagnostics is not None

    def test_exact_matching(self, exact_match_left, exact_match_right):
        result = (
            Pipeline()
            .score(
                comparisons=[
                    ExactComparison("name"),
                    ExactComparison("city"),
                ]
            )
            .filter(min_score=1.0)
            .decide(method="hungarian")
            .build()
            .link(exact_match_left, exact_match_right)
        )

        assert len(result.matches) == 2

    def test_no_matches(self, no_match_left, no_match_right):
        result = (
            Pipeline()
            .block(on="state")
            .score(comparisons=[StringComparison("name")])
            .filter(min_score=0.9)
            .decide(method="hungarian")
            .build()
            .link(no_match_left, no_match_right)
        )

        assert len(result.matches) == 0

    def test_with_crosswalk(self):
        left = pd.DataFrame(
            {
                "name": ["Alice"],
                "state": ["California"],
            }
        )
        right = pd.DataFrame(
            {
                "name": ["Alice"],
                "state": ["CA"],
            }
        )

        result = (
            Pipeline()
            .block(on="state", crosswalk={"California": "CA"})
            .score(comparisons=[StringComparison("name")])
            .filter(min_score=0.9)
            .decide(method="hungarian")
            .build()
            .link(left, right)
        )

        assert len(result.matches) == 1

    def test_different_decision_methods(self, sample_left_df, sample_right_df):
        for method in ["hungarian", "greedy", "row_sequential"]:
            result = (
                Pipeline()
                .block(on="state")
                .score(comparisons=[StringComparison("first_name")])
                .filter(min_score=0.5)
                .decide(method=method)
                .build()
                .link(sample_left_df, sample_right_df)
            )
            assert isinstance(result, LinkageResult)

    def test_margin_filter(self, sample_left_df, sample_right_df):
        result = (
            Pipeline()
            .block(on="state")
            .score(comparisons=[StringComparison("first_name")])
            .filter(min_score=0.5, margin=0.1)
            .decide(method="hungarian")
            .build()
            .link(sample_left_df, sample_right_df)
        )

        assert isinstance(result, LinkageResult)


class TestLinkageResult:
    def test_inspect(self, sample_left_df, sample_right_df):
        result = (
            Pipeline()
            .block(on="state")
            .score(comparisons=[StringComparison("first_name")])
            .filter(min_score=0.5)
            .decide(method="hungarian")
            .build()
            .link(sample_left_df, sample_right_df)
        )

        report = result.inspect()
        assert report.diagnostics is not None
        summary = report.summary()
        assert "Linkage Report" in summary

    def test_merge_left(self, exact_match_left, exact_match_right):
        result = (
            Pipeline()
            .score(comparisons=[ExactComparison("name")])
            .filter(min_score=1.0)
            .decide(method="hungarian")
            .build()
            .link(exact_match_left, exact_match_right)
        )

        merged = result.merge_left()
        assert len(merged) == len(exact_match_left)

    def test_merge_right(self, exact_match_left, exact_match_right):
        result = (
            Pipeline()
            .score(comparisons=[ExactComparison("name")])
            .filter(min_score=1.0)
            .decide(method="hungarian")
            .build()
            .link(exact_match_left, exact_match_right)
        )

        merged = result.merge_right()
        assert len(merged) == len(exact_match_right)
