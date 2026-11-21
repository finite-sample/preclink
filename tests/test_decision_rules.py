"""Tests for decision rule algorithms."""

import pandas as pd
import pytest

from preclink.decide.greedy import GreedyDecision
from preclink.decide.hungarian import HungarianDecision
from preclink.decide.row_sequential import RowSequentialDecision


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

    def test_categorical_source_order_and_ties(self):
        pairs = pd.DataFrame(
            {
                "left_index": pd.Categorical(
                    ["z", "a", "z", "a"], categories=["z", "unused", "a"], ordered=True
                ),
                "right_index": ["shared", "shared", "third", "other"],
                "score": [9, 9, 8, 9],
                "payload": ["first", "second", "third", "fourth"],
            },
            index=pd.Index([30, 8, 13, 2], name="left_index"),
        )
        expected = pd.DataFrame([pairs.iloc[1], pairs.iloc[2]])
        pd.testing.assert_frame_equal(RowSequentialDecision().decide(pairs), expected)

    def test_missing_left_id_retains_source_priority(self):
        pairs = pd.DataFrame(
            {
                "left_index": [2, float("nan"), 1, 2, 1],
                "right_index": ["shared", "unused", "shared", "second", "third"],
                "score": [8, 9, 8, 7, 7],
            },
            index=[5, 30, 5, 8, 7],
        )
        # Python's sorted unique IDs here are [2, nan, 1]. Equal row labels
        # in different source groups remain eligible for separate decisions.
        expected = pd.DataFrame([pairs.iloc[0], pairs.iloc[4]])
        pd.testing.assert_frame_equal(RowSequentialDecision().decide(pairs), expected)

    def test_missing_right_id_uses_pandas_membership(self):
        pairs = pd.DataFrame(
            {"left_index": [0, 1], "right_index": [float("nan")] * 2, "score": [1, 1]}
        )
        first = pairs.iloc[0]
        expected = [first]
        if not pairs["right_index"].isin([first["right_index"]]).iloc[1]:
            expected.append(pairs.iloc[1])
        pd.testing.assert_frame_equal(
            RowSequentialDecision().decide(pairs), pd.DataFrame(expected)
        )

    def test_empty_input(self):
        decision = RowSequentialDecision()
        empty_df = pd.DataFrame(columns=["left_index", "right_index", "score"])
        matches = decision.decide(empty_df)
        assert len(matches) == 0
