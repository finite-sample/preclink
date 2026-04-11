"""Tests for comparison operations."""

import pandas as pd
import pytest
from preclink.score.comparisons import (
    DateComparison,
    ExactComparison,
    NumericComparison,
    StringComparison,
    TFIDFStringComparison,
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


class TestTFIDFStringComparison:
    def test_rare_value_higher_score(self):
        comp = TFIDFStringComparison("name", algorithm="jaro_winkler")
        left_values = pd.Series(["John Smith", "John Smith", "John Smith", "Jagmohan Trivikramji"])
        right_values = pd.Series(["John Smith", "John Smith", "John Smith", "Jagmohan Trivikramji"])
        comp.set_idf_weights(left_values, right_values)

        common_left = pd.Series(["John Smith"])
        common_right = pd.Series(["John Smith"])
        rare_left = pd.Series(["Jagmohan Trivikramji"])
        rare_right = pd.Series(["Jagmohan Trivikramji"])

        common_scores = comp.compare(common_left, common_right)
        rare_scores = comp.compare(rare_left, rare_right)

        assert rare_scores.iloc[0] > common_scores.iloc[0]

    def test_exact_match_without_idf(self):
        comp = TFIDFStringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["Alice", "Bob"])
        right = pd.Series(["Alice", "Bob"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] == pytest.approx(1.0)
        assert scores.iloc[1] == pytest.approx(1.0)

    def test_with_idf_setup(self):
        comp = TFIDFStringComparison("name", algorithm="jaro_winkler")
        all_left = pd.Series(["Alice", "Bob", "Alice", "Alice"])
        all_right = pd.Series(["Alice", "Charlie", "Alice", "Bob"])
        comp.set_idf_weights(all_left, all_right)

        left = pd.Series(["Alice", "Bob"])
        right = pd.Series(["Alice", "Bob"])
        scores = comp.compare(left, right)

        assert scores.iloc[1] > scores.iloc[0]

    def test_missing_values(self):
        comp = TFIDFStringComparison("name")
        left = pd.Series(["Alice", None, "Bob"])
        right = pd.Series(["Alice", "Jane", None])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0
        assert scores.iloc[1] == 0.0
        assert scores.iloc[2] == 0.0

    def test_levenshtein_algorithm(self):
        comp = TFIDFStringComparison("name", algorithm="levenshtein")
        left = pd.Series(["John"])
        right = pd.Series(["Jon"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0.5

    def test_damerau_levenshtein_algorithm(self):
        comp = TFIDFStringComparison("name", algorithm="damerau_levenshtein")
        left = pd.Series(["John"])
        right = pd.Series(["Jonh"])
        scores = comp.compare(left, right)
        assert scores.iloc[0] > 0.5

    def test_weight(self):
        comp = TFIDFStringComparison("name", weight=2.0)
        assert comp.weight == 2.0

    def test_idf_computation(self):
        comp = TFIDFStringComparison("name")
        left = pd.Series(["A", "A", "A", "B"])
        right = pd.Series(["A", "A", "C", "C"])
        comp.set_idf_weights(left, right)

        idf_a = comp._get_idf_weight("A")
        idf_b = comp._get_idf_weight("B")
        idf_c = comp._get_idf_weight("C")

        assert idf_b > idf_a
        assert idf_c > idf_a
        assert idf_b > idf_c

    def test_empty_values(self):
        comp = TFIDFStringComparison("name")
        empty_left = pd.Series([], dtype=str)
        empty_right = pd.Series([], dtype=str)
        comp.set_idf_weights(empty_left, empty_right)

        assert comp._max_idf == 1.0
        assert comp._get_idf_weight("anything") == 1.0
