"""Tests for blocking operations."""

import pandas as pd
from suture.block.blocker import FieldBlocker, FullBlocker
from suture.block.crosswalk import Crosswalk


class TestFieldBlocker:
    def test_single_field_blocking(self, sample_left_df, sample_right_df):
        blocker = FieldBlocker(on="state")
        pairs = blocker.block(sample_left_df, sample_right_df)

        assert "left_index" in pairs.columns
        assert "right_index" in pairs.columns

        ca_left = sample_left_df[sample_left_df["state"] == "CA"]
        ca_right = sample_right_df[sample_right_df["state"] == "CA"]
        expected_ca_pairs = len(ca_left) * len(ca_right)

        ny_left = sample_left_df[sample_left_df["state"] == "NY"]
        ny_right = sample_right_df[sample_right_df["state"] == "NY"]
        expected_ny_pairs = len(ny_left) * len(ny_right)

        tx_left = sample_left_df[sample_left_df["state"] == "TX"]
        tx_right = sample_right_df[sample_right_df["state"] == "TX"]
        expected_tx_pairs = len(tx_left) * len(tx_right)

        expected_total = expected_ca_pairs + expected_ny_pairs + expected_tx_pairs
        assert len(pairs) == expected_total

    def test_multiple_field_blocking(self):
        left = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Alice"],
                "city": ["NYC", "LA", "LA"],
            }
        )
        right = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Alice"],
                "city": ["NYC", "NYC", "LA"],
            }
        )

        blocker = FieldBlocker(on=["name", "city"])
        pairs = blocker.block(left, right)

        assert len(pairs) == 2

    def test_with_crosswalk(self):
        left = pd.DataFrame(
            {
                "state": ["CA", "California"],
            }
        )
        right = pd.DataFrame(
            {
                "state": ["CA", "CA"],
            }
        )

        crosswalk = {"California": "CA"}
        blocker = FieldBlocker(on="state", crosswalk=crosswalk)
        pairs = blocker.block(left, right)

        assert len(pairs) == 4

    def test_missing_block_values(self):
        left = pd.DataFrame(
            {
                "state": ["CA", None, "NY"],
            }
        )
        right = pd.DataFrame(
            {
                "state": ["CA", "NY", None],
            }
        )

        blocker = FieldBlocker(on="state")
        pairs = blocker.block(left, right)

        assert len(pairs) == 2


class TestFullBlocker:
    def test_full_blocking(self):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"id": [3, 4, 5]})

        blocker = FullBlocker()
        pairs = blocker.block(left, right)

        assert len(pairs) == 6


class TestCrosswalk:
    def test_apply(self):
        crosswalk = Crosswalk({"CA": "California", "NY": "New York"})
        series = pd.Series(["CA", "NY", "TX"])
        result = crosswalk.apply(series)

        assert result.iloc[0] == "California"
        assert result.iloc[1] == "New York"
        assert result.iloc[2] == "TX"

    def test_apply_with_missing(self):
        crosswalk = Crosswalk({"CA": "California"})
        series = pd.Series(["CA", None, "NY"])
        result = crosswalk.apply(series)

        assert result.iloc[0] == "California"
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == "NY"

    def test_validate(self):
        crosswalk = Crosswalk({"CA": "California"})
        errors = crosswalk.validate()
        assert len(errors) == 0
