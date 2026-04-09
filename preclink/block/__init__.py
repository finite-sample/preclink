"""Blocking module for reducing comparison space."""

from preclink.block.blocker import FieldBlocker, FullBlocker
from preclink.block.crosswalk import Crosswalk
from preclink.block.protocols import Blocker

__all__ = [
    "Blocker",
    "Crosswalk",
    "FieldBlocker",
    "FullBlocker",
]
