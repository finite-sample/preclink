"""Deduplication module for within-table duplicate removal."""

from tether.deduplicate.cluster import (
    ClusterDeduplicator,
    DeduplicationReport,
    ExactDeduplicator,
)
from tether.deduplicate.protocols import Deduplicator

__all__ = [
    "ClusterDeduplicator",
    "DeduplicationReport",
    "Deduplicator",
    "ExactDeduplicator",
]
