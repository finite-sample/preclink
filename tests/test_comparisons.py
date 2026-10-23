"""Tests for comparison operations."""

import math

import pandas as pd
import pytest
from rapidfuzz import distance as rf_distance

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
        left_values = pd.Series(
            ["John Smith", "John Smith", "John Smith", "Jagmohan Trivikramji"]
        )
        right_values = pd.Series(
            ["John Smith", "John Smith", "John Smith", "Jagmohan Trivikramji"]
        )
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


RAPIDFUZZ_SIMILARITY = {
    "jaro_winkler": rf_distance.JaroWinkler.normalized_similarity,
    "levenshtein": rf_distance.Levenshtein.normalized_similarity,
    "damerau_levenshtein": rf_distance.DamerauLevenshtein.normalized_similarity,
}
PAIR_LEFT = pd.Series(
    ["Jonathan", None, "Mary", 12, "Smith"], index=[10, 20, 30, 40, 50]
)
PAIR_RIGHT = pd.Series(["Jonathon", "Ann", None, "12", "Smyth"], index=[1, 2, 3, 4, 5])


@pytest.mark.parametrize("algorithm", list(RAPIDFUZZ_SIMILARITY))
class TestStringScoresMatchRapidfuzz:
    def test_string_comparison(self, algorithm):
        sim = RAPIDFUZZ_SIMILARITY[algorithm]
        scores = StringComparison("name", algorithm=algorithm).compare(
            PAIR_LEFT, PAIR_RIGHT
        )
        assert list(scores.index) == [10, 20, 30, 40, 50]
        assert list(scores) == [
            sim("Jonathan", "Jonathon"),
            0.0,
            0.0,
            1.0,
            sim("Smith", "Smyth"),
        ]

    def test_tfidf_comparison(self, algorithm):
        sim = RAPIDFUZZ_SIMILARITY[algorithm]
        comp = TFIDFStringComparison("name", algorithm=algorithm)
        corpus = pd.Series(["Jonathan", "Jonathan", "Jonathon", "Smith", "Smyth"])
        comp.set_idf_weights(corpus, pd.Series(["12", "Smith"]))

        n = 7
        counts = {"Jonathan": 2, "Jonathon": 1, "Smith": 2, "Smyth": 1, "12": 1}
        max_idf = math.log(n / 1)

        def weight(value):
            return math.log(n / counts[value]) / max_idf

        def expected(l_str, r_str):
            return sim(l_str, r_str) * ((weight(l_str) + weight(r_str)) / 2)

        scores = comp.compare(PAIR_LEFT, PAIR_RIGHT)
        assert list(scores.index) == [10, 20, 30, 40, 50]
        assert list(scores) == pytest.approx(
            [
                expected("Jonathan", "Jonathon"),
                0.0,
                0.0,
                expected("12", "12"),
                expected("Smith", "Smyth"),
            ],
            rel=1e-12,
        )


@pytest.mark.parametrize("comparison_cls", [StringComparison, TFIDFStringComparison])
def test_unknown_algorithm_raises(comparison_cls):
    comp = comparison_cls("name", algorithm="soundex")
    with pytest.raises(ValueError, match="Unknown algorithm: soundex"):
        comp.compare(pd.Series(["a"]), pd.Series(["a"]))
