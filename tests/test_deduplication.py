"""Tests for deduplication operations."""

import pandas as pd
from preclink.deduplicate import ClusterDeduplicator, DeduplicationReport
from preclink.score.comparisons import ExactComparison, StringComparison


class TestClusterDeduplicator:
    def test_no_duplicates(self):
        """Records that don't match stay separate."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": ["NYC", "LA", "Chicago"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")], threshold=0.9
        )
        result, report = dedup.deduplicate(df)

        assert len(result) == 3
        assert report.original_count == 3
        assert report.kept_count == 3
        assert report.dropped_as_duplicate == 0
        assert report.dropped_as_indistinguishable == 0

    def test_pairwise_duplicates_separate_groups(self):
        """A~B but not A~C or B~C → two groups: {A,B}, {C}."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice", "Charlie"],
                "city": ["NYC", "NYC", "LA"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")], threshold=0.9, margin=0.0
        )
        result, report = dedup.deduplicate(df)

        assert report.groups_found == 1
        assert "Charlie" in result["name"].values

    def test_chain_duplicates_single_group(self):
        """A~B~C chain → one group {A,B,C}."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alyce", "Alise"],
                "city": ["NYC", "NYC", "NYC"],
            }
        )
        dedup = ClusterDeduplicator(
            [StringComparison("name"), ExactComparison("city")], threshold=0.5, margin=0.0
        )
        _, report = dedup.deduplicate(df)

        assert report.groups_found == 1

    def test_clear_winner_kept(self):
        """Record with more complete data wins."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice", "Bob"],
                "city": ["NYC", None, "LA"],
                "state": ["NY", None, "CA"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city"), ExactComparison("state")],
            threshold=0.3,
            margin=0.1,
        )
        result, report = dedup.deduplicate(df)

        assert len(result) == 2
        assert report.dropped_as_duplicate == 1
        assert report.dropped_as_indistinguishable == 0
        alice_row = result[result["name"] == "Alice"]
        assert len(alice_row) == 1
        assert alice_row["city"].iloc[0] == "NYC"

    def test_indistinguishable_dropped(self):
        """Records with same quality → all dropped."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice"],
                "city": ["NYC", "NYC"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")], threshold=0.9, margin=0.1
        )
        result, report = dedup.deduplicate(df)

        assert len(result) == 0
        assert report.dropped_as_indistinguishable == 2
        assert report.dropped_as_duplicate == 0

    def test_report_counts(self):
        """Report counts match expectations."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice", "Bob", "Bob", "Charlie"],
                "city": ["NYC", None, "LA", "LA", "Chicago"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")], threshold=0.9, margin=0.1
        )
        result, report = dedup.deduplicate(df)

        assert report.original_count == 5
        assert report.kept_count == len(result)
        total_dropped = report.dropped_as_duplicate + report.dropped_as_indistinguishable
        assert report.original_count - report.kept_count == total_dropped

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty with correct report."""
        df = pd.DataFrame({"name": [], "city": []})
        dedup = ClusterDeduplicator([ExactComparison("name")], threshold=0.9)
        result, report = dedup.deduplicate(df)

        assert len(result) == 0
        assert report.original_count == 0
        assert report.kept_count == 0

    def test_single_record(self):
        """Single record returned unchanged."""
        df = pd.DataFrame({"name": ["Alice"], "city": ["NYC"]})
        dedup = ClusterDeduplicator([ExactComparison("name")], threshold=0.9)
        result, report = dedup.deduplicate(df)

        assert len(result) == 1
        assert report.original_count == 1
        assert report.kept_count == 1
        assert report.largest_group_size == 1

    def test_timestamp_recency_bonus(self):
        """Newer records get quality bonus when timestamp column specified."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice"],
                "city": [None, None],
                "updated": ["2020-01-01", "2024-01-01"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")],
            threshold=0.5,
            margin=0.05,
            timestamp_column="updated",
        )
        result, report = dedup.deduplicate(df)

        assert len(result) == 1
        assert result["updated"].iloc[0] == "2024-01-01"

    def test_blocking_reduces_pairs(self):
        """block_on parameter limits pair generation."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Alice", "Bob", "Bob"],
                "state": ["CA", "CA", "NY", "NY"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name")], threshold=0.9, margin=0.0, block_on="state"
        )
        _, report = dedup.deduplicate(df)

        assert report.groups_found == 2

    def test_largest_group_size_tracked(self):
        """Largest group size is correctly tracked."""
        df = pd.DataFrame(
            {
                "name": ["A", "A", "A", "B", "B"],
                "city": ["X", "X", "X", "Y", "Y"],
            }
        )
        dedup = ClusterDeduplicator(
            [ExactComparison("name"), ExactComparison("city")], threshold=0.9, margin=0.0
        )
        _, report = dedup.deduplicate(df)

        assert report.largest_group_size == 3


class TestDeduplicationReport:
    def test_dataclass_fields(self):
        """DeduplicationReport has expected fields."""
        report = DeduplicationReport(
            original_count=100,
            kept_count=80,
            dropped_as_duplicate=15,
            dropped_as_indistinguishable=5,
            groups_found=10,
            largest_group_size=4,
        )

        assert report.original_count == 100
        assert report.kept_count == 80
        assert report.dropped_as_duplicate == 15
        assert report.dropped_as_indistinguishable == 5
        assert report.groups_found == 10
        assert report.largest_group_size == 4
