"""Tests for comparison operations."""

import pandas as pd
import pytest
from suture.score.comparisons import (
    DateComparison,
    ExactComparison,
    NumericComparison,
    StringComparison,
)


class TestStringComparison:
    def test_jaro_winkler_exact_match(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["Alice", "Bob"])
        right = pd.Series(["Alice", "Bob"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == pytest.approx(1.0)

    def test_jaro_winkler_similar(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["John"])
        right = pd.Series(["Jon"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0.9

    def test_jaro_winkler_different(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["Alice"])
        right = pd.Series(["Bob"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] < 0.5

    def test_levenshtein(self):
        comp = StringComparison("name", algorithm="levenshtein")
        left = pd.Series(["John", "Alice"])
        right = pd.Series(["Jon", "Alice"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0.7
        assert scores.iloc[1] == pytest.approx(1.0)

    def test_damerau_levenshtein(self):
        comp = StringComparison("name", algorithm="damerau_levenshtein")
        left = pd.Series(["John"])
        right = pd.Series(["Jonh"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0.7

    def test_missing_values(self):
        comp = StringComparison("name")
        left = pd.Series(["Alice", None, "Bob"])
        right = pd.Series(["Alice", "Jane", None])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == 0.0
        assert scores.iloc[2] == 0.0

    def test_weight(self):
        comp = StringComparison("name", weight=2.0)
        assert comp.weight == 2.0


class TestExactComparison:
    def test_exact_match(self):
        comp = ExactComparison("id")
        left = pd.Series([1, 2, 3])
        right = pd.Series([1, 2, 4])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 1.0
        assert scores.iloc[2] == 0.0

    def test_string_exact_match(self):
        comp = ExactComparison("name")
        left = pd.Series(["Alice", "Bob"])
        right = pd.Series(["Alice", "Bob"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 1.0

    def test_missing_values(self):
        comp = ExactComparison("id")
        left = pd.Series([1, None, 3])
        right = pd.Series([1, 2, None])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 0.0
        assert scores.iloc[2] == 0.0


class TestNumericComparison:
    def test_exact_match_no_tolerance(self):
        comp = NumericComparison("value", tolerance=0)
        left = pd.Series([100, 200, 300])
        right = pd.Series([100, 200, 301])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 1.0
        assert scores.iloc[2] == 0.0

    def test_with_tolerance(self):
        comp = NumericComparison("value", tolerance=10)
        left = pd.Series([100, 100, 100])
        right = pd.Series([100, 105, 115])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == pytest.approx(0.5)
        assert scores.iloc[2] == pytest.approx(0.0)

    def test_gaussian_scale(self):
        comp = NumericComparison("value", tolerance=10, scale="gaussian")
        left = pd.Series([100])
        right = pd.Series([105])
        scores = comp.compare(left, right)
        assert 0 < scores.iloc[0] < 1

    def test_missing_values(self):
        comp = NumericComparison("value", tolerance=10)
        left = pd.Series([100, None])
        right = pd.Series([100, 100])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == 0.0


class TestDateComparison:
    def test_exact_match(self):
        comp = DateComparison("date", tolerance_days=0)
        left = pd.Series(["2024-01-01", "2024-01-02"])
        right = pd.Series(["2024-01-01", "2024-01-02"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 1.0

    def test_with_tolerance(self):
        comp = DateComparison("date", tolerance_days=10)
        left = pd.Series(["2024-01-01", "2024-01-01", "2024-01-01"])
        right = pd.Series(["2024-01-01", "2024-01-06", "2024-01-15"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == pytest.approx(0.5)
        assert scores.iloc[2] == pytest.approx(0.0)

    def test_missing_values(self):
        comp = DateComparison("date")
        left = pd.Series(["2024-01-01", None])
        right = pd.Series(["2024-01-01", "2024-01-01"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == 1.0
        assert scores.iloc[1] == 0.0
