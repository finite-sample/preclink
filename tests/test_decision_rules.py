"""Tests for decision rule algorithms."""

import pandas as pd
import pytest
from tether.decide.greedy import GreedyDecision
from tether.decide.hungarian import HungarianDecision
from tether.decide.row_sequential import RowSequentialDecision


@pytest.fixture
def scored_pairs():
    return pd.DataFrame(
        {
            "left_index": [0, 0, 1, 1, 2],
            "right_index": [0, 1, 0, 1, 2],
            "score": [0.9, 0.7, 0.6, 0.95, 0.8],
        }
    )


@pytest.fixture
def ambiguous_pairs():
    return pd.DataFrame(
        {
            "left_index": [0, 0, 1, 1],
            "right_index": [0, 1, 0, 1],
            "score": [0.85, 0.84, 0.9, 0.5],
        }
    )


class TestHungarianDecision:
    def test_optimal_assignment(self, scored_pairs):
        decision = HungarianDecision()
        matches = decision.decide(scored_pairs)

        assert len(matches) == 3

        left_matched = set(matches["left_index"])
        right_matched = set(matches["right_index"])
        assert len(left_matched) == len(matches)
        assert len(right_matched) == len(matches)

    def test_empty_input(self):
        decision = HungarianDecision()
        empty_df = pd.DataFrame(columns=["left_index", "right_index", "score"])
        matches = decision.decide(empty_df)
        assert len(matches) == 0

    def test_single_pair(self):
        decision = HungarianDecision()
        single = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [0],
                "score": [0.9],
            }
        )
        matches = decision.decide(single)
        assert len(matches) == 1


class TestGreedyDecision:
    def test_best_first(self, scored_pairs):
        decision = GreedyDecision()
        matches = decision.decide(scored_pairs)

        assert len(matches) == 3

        left_matched = set(matches["left_index"])
        right_matched = set(matches["right_index"])
        assert len(left_matched) == len(matches)
        assert len(right_matched) == len(matches)

    def test_greedy_selection_order(self):
        pairs = pd.DataFrame(
            {
                "left_index": [0, 0, 1],
                "right_index": [0, 1, 0],
                "score": [0.8, 0.95, 0.7],
            }
        )

        decision = GreedyDecision()
        matches = decision.decide(pairs)

        best_match = matches.iloc[0]
        assert best_match["score"] == 0.95

    def test_empty_input(self):
        decision = GreedyDecision()
        empty_df = pd.DataFrame(columns=["left_index", "right_index", "score"])
        matches = decision.decide(empty_df)
        assert len(matches) == 0


class TestRowSequentialDecision:
    def test_sequential_matching(self, scored_pairs):
        decision = RowSequentialDecision()
        matches = decision.decide(scored_pairs)

        assert len(matches) <= 3

        left_matched = set(matches["left_index"])
        right_matched = set(matches["right_index"])
        assert len(left_matched) == len(matches)
        assert len(right_matched) == len(matches)

    def test_respects_order(self):
        pairs = pd.DataFrame(
            {
                "left_index": [0, 0, 1, 1],
                "right_index": [0, 1, 0, 1],
                "score": [0.7, 0.9, 0.95, 0.8],
            }
        )

        decision = RowSequentialDecision()
        matches = decision.decide(pairs)

        first_match = matches[matches["left_index"] == 0].iloc[0]
        assert first_match["right_index"] == 1

    def test_empty_input(self):
        decision = RowSequentialDecision()
        empty_df = pd.DataFrame(columns=["left_index", "right_index", "score"])
        matches = decision.decide(empty_df)
        assert len(matches) == 0
