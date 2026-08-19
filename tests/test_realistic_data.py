"""Comprehensive tests with realistic data for validation.

This module contains:
1. Ground-truth benchmark tests (Febrl-style synthetic data)
2. Synthetic data with controlled corruption
3. Property-based tests for TFIDFStringComparison
4. Integration tests with real-world patterns
5. Regression tests against sklearn TF-IDF
"""

import random
import string

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.feature_extraction.text import TfidfVectorizer

from preclink import ExactComparison, Pipeline, StringComparison
from preclink.preprocess.normalizer import CompletenessFilter
from preclink.score.comparisons import TFIDFStringComparison

# =============================================================================
# 1. Ground-Truth Benchmark Tests (Febrl-style)
# =============================================================================


def generate_febrl_style_data(n_records: int, n_duplicates: int, seed: int = 42):
    """Generate synthetic benchmark data with known ground truth.

    Creates records with first_name, last_name, dob, address, and introduces
    controlled duplicates with variations.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    first_names = [
        "John",
        "Jane",
        "Robert",
        "Mary",
        "Michael",
        "Patricia",
        "William",
        "Jennifer",
        "David",
        "Linda",
        "James",
        "Elizabeth",
        "Richard",
        "Barbara",
        "Joseph",
        "Susan",
        "Thomas",
        "Jessica",
        "Charles",
        "Sarah",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
    ]
    streets = ["Main St", "Oak Ave", "Elm St", "Park Rd", "Cedar Ln", "Maple Dr"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]

    records = [
        {
            "rec_id": f"rec-{i:04d}",
            "first_name": rng.choice(first_names),
            "last_name": rng.choice(last_names),
            "dob": (
                f"19{rng.randint(50, 99)}-"
                f"{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ),
            "address": f"{rng.randint(1, 999)} {rng.choice(streets)}",
            "city": rng.choice(cities),
        }
        for i in range(n_records)
    ]

    df_a = pd.DataFrame(records)

    duplicates = []
    ground_truth = []
    dup_indices = np_rng.choice(
        n_records, size=min(n_duplicates, n_records), replace=False
    )

    for idx in dup_indices:
        orig = records[idx].copy()
        dup = orig.copy()
        dup["rec_id"] = f"dup-{idx:04d}"

        mod_type = rng.choice(["typo", "swap", "abbrev", "missing"])
        if mod_type == "typo" and len(dup["first_name"]) > 1:
            pos = rng.randint(0, len(dup["first_name"]) - 1)
            chars = list(dup["first_name"])
            chars[pos] = rng.choice(string.ascii_lowercase)
            dup["first_name"] = "".join(chars)
        elif mod_type == "swap" and len(dup["last_name"]) > 2:
            chars = list(dup["last_name"])
            p1 = rng.randint(0, len(chars) - 2)
            chars[p1], chars[p1 + 1] = chars[p1 + 1], chars[p1]
            dup["last_name"] = "".join(chars)
        elif mod_type == "abbrev":
            dup["address"] = (
                dup["address"].replace("Street", "St").replace("Avenue", "Ave")
            )
        elif mod_type == "missing":
            dup["city"] = None

        duplicates.append(dup)
        ground_truth.append((orig["rec_id"], dup["rec_id"]))

    df_b = pd.DataFrame(duplicates)

    return df_a, df_b, ground_truth


class TestGroundTruthBenchmark:
    """Tests using synthetic benchmark data with known matches."""

    def test_febrl_style_basic_linkage(self):
        df_a, df_b, ground_truth = generate_febrl_style_data(100, 20, seed=42)

        pipeline = (
            Pipeline()
            .score(
                [
                    StringComparison(
                        "first_name", algorithm="jaro_winkler", weight=2.0
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=2.0),
                    ExactComparison("dob", weight=1.5),
                ]
            )
            .filter(min_score=0.6)
            .build()
        )
        result = pipeline.link(df_a, df_b)

        gt_pairs = {(a, b) for a, b in ground_truth}
        found_pairs = set(
            zip(
                result.matches["rec_id_left"],
                result.matches["rec_id_right"],
                strict=True,
            )
        )

        true_positives = len(gt_pairs & found_pairs)
        recall = true_positives / len(gt_pairs) if gt_pairs else 0

        assert recall > 0.5, f"Recall too low: {recall:.2%}"

    def test_febrl_style_high_precision(self):
        df_a, df_b, ground_truth = generate_febrl_style_data(50, 10, seed=123)

        pipeline = (
            Pipeline()
            .score(
                [
                    StringComparison(
                        "first_name", algorithm="jaro_winkler", weight=1.0
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=1.0),
                    ExactComparison("dob", weight=2.0),
                ]
            )
            .filter(min_score=0.8)
            .build()
        )
        result = pipeline.link(df_a, df_b)

        gt_pairs = {(a, b) for a, b in ground_truth}
        found_pairs = set(
            zip(
                result.matches["rec_id_left"],
                result.matches["rec_id_right"],
                strict=True,
            )
        )

        true_positives = len(gt_pairs & found_pairs)
        precision = true_positives / len(found_pairs) if found_pairs else 1.0

        assert precision > 0.3, f"Precision too low: {precision:.2%}"


# =============================================================================
# 2. Synthetic Data with Controlled Corruption
# =============================================================================


def apply_typo(s: str, rng: random.Random) -> str:
    """Apply a random typo to a string."""
    if len(s) < 2:
        return s
    typo_type = rng.choice(["insert", "delete", "replace", "transpose"])
    chars = list(s)
    pos = rng.randint(0, len(chars) - 1)

    if typo_type == "insert":
        chars.insert(pos, rng.choice(string.ascii_lowercase))
    elif typo_type == "delete" and len(chars) > 1:
        chars.pop(pos)
    elif typo_type == "replace":
        chars[pos] = rng.choice(string.ascii_lowercase)
    elif typo_type == "transpose" and pos < len(chars) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    return "".join(chars)


class TestControlledCorruption:
    """Tests with synthetically corrupted data."""

    def test_single_typo_recovery(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        original = "Elizabeth"
        typo = "Elizabth"

        left = pd.Series([original])
        right = pd.Series([typo])
        score = comp.compare(left, right).iloc[0]

        assert score > 0.9, f"Single typo should score > 0.9, got {score:.3f}"

    def test_transposition_recovery(self):
        comp = StringComparison("name", algorithm="damerau_levenshtein")
        original = "Robert"
        transposed = "Robret"

        left = pd.Series([original])
        right = pd.Series([transposed])
        score = comp.compare(left, right).iloc[0]

        assert score > 0.8, f"Transposition should score > 0.8, got {score:.3f}"

    def test_multiple_typos_degradation(self):
        comp = StringComparison("name", algorithm="levenshtein")
        original = "Christopher"

        rng = random.Random(42)
        corrupted = original
        scores = []

        for _ in range(5):
            left = pd.Series([original])
            right = pd.Series([corrupted])
            score = comp.compare(left, right).iloc[0]
            scores.append(score)
            corrupted = apply_typo(corrupted, rng)

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "More typos should reduce score"

    def test_abbreviation_patterns(self):
        comp = StringComparison("address", algorithm="jaro_winkler")

        test_cases = [
            ("123 Main Street", "123 Main St"),
            ("456 Oak Avenue", "456 Oak Ave"),
            ("Robert", "Bob"),
            ("William", "Bill"),
        ]

        for full, abbrev in test_cases:
            left = pd.Series([full])
            right = pd.Series([abbrev])
            score = comp.compare(left, right).iloc[0]
            assert score > 0.5, f"'{full}' vs '{abbrev}' should match, got {score:.3f}"

    def test_missing_values_dont_match(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["Alice", None, "Bob"])
        right = pd.Series([None, "Jane", "Bob"])
        scores = comp.compare(left, right)

        assert scores.iloc[0] == 0.0
        assert scores.iloc[1] == 0.0
        assert scores.iloc[2] == pytest.approx(1.0)


# =============================================================================
# 3. Property-Based Tests for TFIDFStringComparison
# =============================================================================


class TestTFIDFProperties:
    """Property-based tests for TF-IDF string comparison."""

    @given(st.text(min_size=1, max_size=50, alphabet=string.ascii_letters + " "))
    @settings(max_examples=50)
    def test_identical_strings_high_score(self, s: str):
        s = s.strip()
        if not s:
            return

        comp = TFIDFStringComparison("name", algorithm="jaro_winkler")
        left = pd.Series([s])
        right = pd.Series([s])
        score = comp.compare(left, right).iloc[0]

        assert 0.9 <= score <= 1.0, f"Identical '{s}' should score >= 0.9, got {score}"

    @given(st.text(min_size=3, max_size=20, alphabet=string.ascii_lowercase))
    @settings(max_examples=50)
    def test_scores_bounded_zero_one(self, s: str):
        comp = TFIDFStringComparison("name", algorithm="levenshtein")
        left = pd.Series([s, s + "xyz", "completely_different"])
        right = pd.Series([s, s, s])
        scores = comp.compare(left, right)

        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_rare_scores_higher_than_common(self):
        comp = TFIDFStringComparison("name", algorithm="jaro_winkler")

        left_corpus = pd.Series(["John Smith"] * 100 + ["Xerxes Quigley"] * 1)
        right_corpus = pd.Series(["John Smith"] * 100 + ["Xerxes Quigley"] * 1)
        comp.set_idf_weights(left_corpus, right_corpus)

        common = comp.compare(
            pd.Series(["John Smith"]), pd.Series(["John Smith"])
        ).iloc[0]
        rare = comp.compare(
            pd.Series(["Xerxes Quigley"]), pd.Series(["Xerxes Quigley"])
        ).iloc[0]

        assert rare > common, f"Rare ({rare:.3f}) should > common ({common:.3f})"

    def test_idf_formula_correctness(self):
        comp = TFIDFStringComparison("name")

        left = pd.Series(["A", "A", "A", "A", "B"])
        right = pd.Series(["A", "A", "A", "C", "C"])
        comp.set_idf_weights(left, right)

        n = 10
        expected_idf_a = np.log(n / 7)
        expected_idf_b = np.log(n / 1)
        expected_idf_c = np.log(n / 2)

        actual_idf_a = comp._idf_weights.get("A", 0)
        actual_idf_b = comp._idf_weights.get("B", 0)
        actual_idf_c = comp._idf_weights.get("C", 0)

        assert pytest.approx(actual_idf_a, rel=0.01) == expected_idf_a
        assert pytest.approx(actual_idf_b, rel=0.01) == expected_idf_b
        assert pytest.approx(actual_idf_c, rel=0.01) == expected_idf_c

    @given(
        st.lists(
            st.text(min_size=1, max_size=10, alphabet=string.ascii_letters),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=30)
    def test_idf_weights_all_positive(self, values):
        values = [v.strip() for v in values if v.strip()]
        if len(values) < 2:
            return

        comp = TFIDFStringComparison("name")
        series = pd.Series(values)
        comp.set_idf_weights(series, series)

        for weight in comp._idf_weights.values():
            assert weight >= 0, f"IDF weight should be non-negative: {weight}"


# =============================================================================
# 4. Integration Tests with Real-World Patterns
# =============================================================================


class TestRealWorldPatterns:
    """Tests with realistic name/address variations."""

    @pytest.mark.parametrize(
        ("full", "nick", "min_score"),
        [
            ("Robert", "Bob", 0.3),
            ("William", "Bill", 0.3),
            ("Richard", "Dick", 0.3),
            ("Elizabeth", "Liz", 0.3),
            ("Michael", "Mike", 0.5),
            ("Jennifer", "Jenny", 0.5),
            ("Katherine", "Kate", 0.3),
            ("Katherine", "Kathy", 0.5),
            ("Christopher", "Chris", 0.5),
        ],
    )
    def test_nickname_matching(self, full, nick, min_score):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series([full])
        right = pd.Series([nick])
        score = comp.compare(left, right).iloc[0]
        assert score > min_score, f"'{full}' vs '{nick}' should score > {min_score}"

    @pytest.mark.parametrize(
        ("addr1", "addr2", "min_score"),
        [
            ("123 Main Street", "123 Main St", 0.8),
            ("456 Oak Avenue", "456 Oak Ave", 0.8),
            ("789 Elm Boulevard", "789 Elm Blvd", 0.8),
            ("101 Park Road", "101 Park Rd", 0.8),
            ("202 Cedar Lane", "202 Cedar Ln", 0.8),
            ("303 Maple Drive", "303 Maple Dr", 0.8),
            ("Apartment 4", "Apt 4", 0.5),
            ("Suite 100", "Ste 100", 0.7),
        ],
    )
    def test_address_abbreviations(self, addr1, addr2, min_score):
        comp = StringComparison("address", algorithm="jaro_winkler")
        left = pd.Series([addr1])
        right = pd.Series([addr2])
        score = comp.compare(left, right).iloc[0]
        assert score > min_score, f"'{addr1}' vs '{addr2}' should score > {min_score}"

    @pytest.mark.parametrize(
        ("name1", "name2"),
        [
            ("IBM", "International Business Machines"),
            ("AT&T", "American Telephone Telegraph"),
            ("HP", "Hewlett Packard"),
        ],
    )
    def test_business_abbreviations_low_match(self, name1, name2):
        comp = StringComparison("company", algorithm="jaro_winkler")
        left = pd.Series([name1])
        right = pd.Series([name2])
        score = comp.compare(left, right).iloc[0]
        assert score < 0.6, f"Acronym '{name1}' should not strongly match '{name2}'"

    def test_unicode_normalization(self):
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["José García", "Müller", "naïve"])
        right = pd.Series(["Jose Garcia", "Muller", "naive"])
        scores = comp.compare(left, right)

        assert all(score > 0.8 for score in scores)

    def test_case_sensitivity_jaro_winkler(self):
        """Jaro-Winkler is case-sensitive: matching needs same case or preprocessing."""
        comp = StringComparison("name", algorithm="jaro_winkler")
        left = pd.Series(["JOHN", "alice"])
        right = pd.Series(["john", "ALICE"])
        scores = comp.compare(left, right)

        assert scores.iloc[0] < 1.0
        assert scores.iloc[1] < 1.0


# =============================================================================
# 5. Regression Tests Against sklearn TF-IDF
# =============================================================================


class TestSklearnRegression:
    """Verify TF-IDF behavior against sklearn implementation."""

    def test_rare_terms_have_higher_idf_than_common(self):
        """Core property: rare terms should have higher IDF than common terms."""
        corpus = ["common"] * 50 + ["rare"] * 2

        comp = TFIDFStringComparison("name")
        series = pd.Series(corpus)
        comp.set_idf_weights(series, pd.Series(dtype=str))

        idf_common = comp._idf_weights.get("common", 0)
        idf_rare = comp._idf_weights.get("rare", 0)

        assert idf_rare > idf_common, (
            f"Rare IDF ({idf_rare}) should > common ({idf_common})"
        )

    def test_idf_increases_with_rarity(self):
        """IDF should increase as term frequency decreases."""
        corpus = (
            ["very_common"] * 100 + ["common"] * 50 + ["uncommon"] * 10 + ["rare"] * 2
        )

        comp = TFIDFStringComparison("name")
        series = pd.Series(corpus)
        comp.set_idf_weights(series, pd.Series(dtype=str))

        idf_very_common = comp._idf_weights.get("very_common", 0)
        idf_common = comp._idf_weights.get("common", 0)
        idf_uncommon = comp._idf_weights.get("uncommon", 0)
        idf_rare = comp._idf_weights.get("rare", 0)

        assert idf_rare > idf_uncommon > idf_common > idf_very_common

    def test_sklearn_and_our_idf_same_ordering(self):
        """Both implementations should order terms by rarity the same way."""
        corpus = ["apple"] * 10 + ["banana"] * 5 + ["cherry"] * 2 + ["date"] * 1

        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)
        vectorizer.fit(corpus)
        sklearn_idf = dict(
            zip(vectorizer.get_feature_names_out(), vectorizer.idf_, strict=True)
        )
        sklearn_order = sorted(
            sklearn_idf.keys(), key=lambda x: sklearn_idf[x], reverse=True
        )

        comp = TFIDFStringComparison("name")
        series = pd.Series(corpus)
        comp.set_idf_weights(series, pd.Series(dtype=str))
        our_order = sorted(
            comp._idf_weights.keys(), key=lambda x: comp._idf_weights[x], reverse=True
        )

        assert sklearn_order == our_order


# =============================================================================
# 6. CompletenessFilter Tests
# =============================================================================


class TestCompletenessFilterRealistic:
    """Realistic tests for CompletenessFilter."""

    def test_filter_incomplete_records(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David", "Eve"],
                "email": ["a@test.com", None, "c@test.com", None, "e@test.com"],
                "phone": ["111", "222", "333", None, None],
            }
        )

        filt = CompletenessFilter(min_completeness=1.0)
        filtered, report = filt.filter(df)

        assert len(filtered) == 1
        assert report.dropped_count == 4

    def test_partial_completeness_threshold(self):
        df = pd.DataFrame(
            {
                "a": ["x", "x", None, "x"],
                "b": ["x", None, "x", None],
                "c": ["x", "x", "x", None],
            }
        )

        filt = CompletenessFilter(min_completeness=0.5)
        filtered, report = filt.filter(df)

        assert len(filtered) == 3
        assert report.dropped_count == 1

    def test_required_columns_only(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", None, "Charlie"],
                "optional": [None, None, None],
            }
        )

        filt = CompletenessFilter(min_completeness=1.0, required_columns=["id", "name"])
        filtered, report = filt.filter(df)

        assert len(filtered) == 2
        assert 1 in report.dropped_indices

    def test_empty_string_as_missing(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie"],
                "city": ["NYC", "LA", ""],
            }
        )

        filt = CompletenessFilter(min_completeness=1.0)
        filtered, _ = filt.filter(df)

        assert len(filtered) == 1
        assert filtered.iloc[0]["name"] == "Alice"


# =============================================================================
# 7. End-to-End Pipeline Tests
# =============================================================================


class TestEndToEndPipeline:
    """Full pipeline tests with realistic data."""

    def test_pipeline_with_preprocessing(self):
        left = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Bob"],
                "last_name": ["Smith", "Doe", "Johnson"],
                "dob": ["1990-01-15", "1985-06-20", "1992-03-10"],
            }
        )

        right = pd.DataFrame(
            {
                "id": [101, 102, 103],
                "first_name": ["Jon", "Jane", "Robert"],
                "last_name": ["Smith", "Doe", "Johnson"],
                "dob": ["1990-01-15", "1985-06-20", "1992-03-10"],
            }
        )

        pipeline = (
            Pipeline()
            .preprocess(lowercase=True)
            .score(
                [
                    StringComparison(
                        "first_name", algorithm="jaro_winkler", weight=1.5
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=1.5),
                    ExactComparison("dob", weight=1.0),
                ]
            )
            .filter(min_score=0.7)
            .build()
        )

        result = pipeline.link(left, right)

        assert len(result.matches) >= 2

    def test_tfidf_improves_rare_matches(self):
        left = pd.DataFrame(
            {
                "id": range(1, 6),
                "name": [
                    "John Smith",
                    "John Smith",
                    "John Smith",
                    "Xerxes Quigley",
                    "Jane Doe",
                ],
            }
        )
        right = pd.DataFrame(
            {
                "id": range(101, 106),
                "name": [
                    "John Smith",
                    "John Smith",
                    "John Smith",
                    "Xerxes Quigley",
                    "Jane Doe",
                ],
            }
        )

        tfidf_comp = TFIDFStringComparison("name", algorithm="jaro_winkler", weight=1.0)
        tfidf_comp.set_idf_weights(left["name"], right["name"])

        common_score = tfidf_comp.compare(
            pd.Series(["John Smith"]), pd.Series(["John Smith"])
        ).iloc[0]
        rare_score = tfidf_comp.compare(
            pd.Series(["Xerxes Quigley"]), pd.Series(["Xerxes Quigley"])
        ).iloc[0]

        assert rare_score > common_score
