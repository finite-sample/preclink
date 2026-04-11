"""Tests for preprocessing operations."""

import pandas as pd
import pytest
from preclink.preprocess.normalizer import (
    CompletenessFilter,
    CompletenessReport,
    TextNormalizer,
)


class TestCompletenessFilter:
    def test_no_filtering_when_threshold_zero(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie"],
                "city": ["NYC", "LA", None],
            }
        )
        filt = CompletenessFilter(min_completeness=0.0)
        filtered, report = filt.filter(df)
        assert len(filtered) == 3
        assert report.dropped_count == 0

    def test_filter_incomplete_records(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie", None],
                "city": ["NYC", None, "SF", "LA"],
                "state": ["NY", None, "CA", None],
            }
        )
        filt = CompletenessFilter(min_completeness=0.5)
        filtered, report = filt.filter(df)
        assert len(filtered) == 2
        assert report.dropped_count == 2
        assert 1 in report.dropped_indices
        assert 3 in report.dropped_indices

    def test_filter_with_required_columns(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": [None, "LA", "SF"],
                "optional": [None, None, None],
            }
        )
        filt = CompletenessFilter(min_completeness=1.0, required_columns=["name", "city"])
        filtered, report = filt.filter(df)
        assert len(filtered) == 2
        assert "Alice" not in filtered["name"].values

    def test_completeness_report(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie"],
                "city": ["NYC", None, "SF"],
            }
        )
        filt = CompletenessFilter(min_completeness=0.75)
        _, report = filt.filter(df)
        assert isinstance(report, CompletenessReport)
        assert report.total_count == 3
        assert report.retained_count + report.dropped_count == report.total_count
        assert 0.0 <= report.drop_rate <= 1.0

    def test_completeness_scores(self):
        df = pd.DataFrame(
            {
                "a": ["x", None, "z", None],
                "b": ["x", "y", None, None],
            }
        )
        filt = CompletenessFilter(min_completeness=0.0)
        _, report = filt.filter(df)
        assert report.completeness_scores.iloc[0] == pytest.approx(1.0)
        assert report.completeness_scores.iloc[1] == pytest.approx(0.5)
        assert report.completeness_scores.iloc[2] == pytest.approx(0.5)
        assert report.completeness_scores.iloc[3] == pytest.approx(0.0)

    def test_empty_string_treated_as_missing(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie"],
                "city": ["NYC", "LA", ""],
            }
        )
        filt = CompletenessFilter(min_completeness=1.0)
        filtered, report = filt.filter(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]["name"] == "Alice"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="must be between"):
            CompletenessFilter(min_completeness=1.5)
        with pytest.raises(ValueError, match="must be between"):
            CompletenessFilter(min_completeness=-0.1)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"name": [], "city": []})
        filt = CompletenessFilter(min_completeness=0.5)
        filtered, report = filt.filter(df)
        assert len(filtered) == 0
        assert report.dropped_count == 0
        assert report.drop_rate == 0.0


class TestTextNormalizer:
    def test_normalize_unicode(self):
        df = pd.DataFrame({"name": ["caf\u00e9", "na\u00efve"]})
        normalizer = TextNormalizer(normalize_unicode=True)
        result = normalizer.preprocess(df)
        assert result["name"].iloc[0] == "café"

    def test_lowercase(self):
        df = pd.DataFrame({"name": ["ALICE", "Bob"]})
        normalizer = TextNormalizer(lowercase=True)
        result = normalizer.preprocess(df)
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[1] == "bob"

    def test_strip_whitespace(self):
        df = pd.DataFrame({"name": ["  Alice  ", "Bob   "]})
        normalizer = TextNormalizer(strip_whitespace=True)
        result = normalizer.preprocess(df)
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[1] == "bob"

    def test_collapse_whitespace(self):
        df = pd.DataFrame({"name": ["Alice   Smith", "Bob    Johnson"]})
        normalizer = TextNormalizer(collapse_whitespace=True)
        result = normalizer.preprocess(df)
        assert result["name"].iloc[0] == "alice smith"
        assert result["name"].iloc[1] == "bob johnson"

    def test_specific_columns(self):
        df = pd.DataFrame(
            {
                "name": ["ALICE"],
                "city": ["NYC"],
            }
        )
        normalizer = TextNormalizer(columns=["name"])
        result = normalizer.preprocess(df)
        assert result["name"].iloc[0] == "alice"
        assert result["city"].iloc[0] == "NYC"

    def test_preserve_nulls(self):
        df = pd.DataFrame({"name": ["Alice", None, "Bob"]})
        normalizer = TextNormalizer()
        result = normalizer.preprocess(df)
        assert pd.isna(result["name"].iloc[1])
