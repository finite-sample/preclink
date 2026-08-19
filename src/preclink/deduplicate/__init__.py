"""Deduplication module for within-table duplicate removal."""

from preclink.deduplicate.cluster import (
    ClusterDeduplicator,
    DeduplicationReport,
    ExactDeduplicator,
)
from preclink.deduplicate.protocols import Deduplicator

__all__ = [
    "ClusterDeduplicator",
    "DeduplicationReport",
    "Deduplicator",
    "ExactDeduplicator",
]
