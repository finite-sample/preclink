"""Deduplication module for within-table duplicate removal."""

from suture.deduplicate.cluster import (
    ClusterDeduplicator,
    DeduplicationReport,
    ExactDeduplicator,
)
from suture.deduplicate.protocols import Deduplicator

__all__ = [
    "ClusterDeduplicator",
    "DeduplicationReport",
    "Deduplicator",
    "ExactDeduplicator",
]
