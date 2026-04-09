"""Configuration dataclasses for pipeline stages."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from preclink._typing import DecisionMethod, MissingPolicy

if TYPE_CHECKING:
    from preclink.block.crosswalk import Crosswalk
    from preclink.score.protocols import Comparison


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing stage.

    Args:
        normalize_unicode: Whether to normalize unicode characters.
        lowercase: Whether to convert to lowercase.
        strip_whitespace: Whether to strip whitespace.
        collapse_whitespace: Whether to collapse multiple whitespace.
        missing_policy: How to handle missing values.
        columns: Specific columns to preprocess.
    """

    normalize_unicode: bool = True
    lowercase: bool = True
    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    missing_policy: MissingPolicy = "skip"
    columns: list[str] | None = None


@dataclass(frozen=True)
class BlockConfig:
    """Configuration for blocking stage.

    Args:
        on: Field(s) to block on.
        crosswalk: Optional crosswalk mapping.
    """

    on: str | list[str]
    crosswalk: "Crosswalk | dict[str, str] | None" = None


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for scoring stage.

    Args:
        comparisons: List of comparison operations.
    """

    comparisons: list["Comparison"] = field(default_factory=list)


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for filtering stage.

    Args:
        min_score: Minimum score threshold.
        margin: Minimum margin for ambiguity removal.
    """

    min_score: float = 0.0
    margin: float | None = None


@dataclass(frozen=True)
class DecideConfig:
    """Configuration for decision stage.

    Args:
        method: Decision algorithm to use.
    """

    method: DecisionMethod = "hungarian"
