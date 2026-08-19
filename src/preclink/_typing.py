"""Type definitions and narrowing helpers for preclink."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    import pandas as pd

DecisionMethod = Literal["hungarian", "greedy", "row_sequential"]
StringAlgorithm = Literal["jaro_winkler", "levenshtein", "damerau_levenshtein"]
MissingPolicy = Literal["skip", "zero", "penalize"]


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read one column of ``frame`` as a Series.

    ``frame[key]`` is typed as ``Series | DataFrame`` because ``key`` may be a
    list of labels. Every call site here passes a single label, so the result
    is always a Series; narrowing it in one documented place beats a cast at
    each of the twenty-odd uses.

    Args:
        frame: DataFrame to read from.
        name: A single column label.

    Returns:
        The column as a Series.
    """
    return cast("pd.Series", frame[name])


def rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Select the rows of ``frame`` where ``mask`` is true.

    Boolean-mask indexing carries the same ``Series | DataFrame`` ambiguity as
    label indexing; a boolean mask always yields a DataFrame.

    Args:
        frame: DataFrame to filter.
        mask: Boolean Series aligned to ``frame``'s index.

    Returns:
        The selected rows.
    """
    return cast("pd.DataFrame", frame[mask])


def select(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Select several columns of ``frame`` as a DataFrame.

    The list-of-labels form of ``frame[key]`` always yields a DataFrame, but
    shares the ``Series | DataFrame`` return type with the single-label form.

    Args:
        frame: DataFrame to read from.
        names: Column labels to keep, in order.

    Returns:
        The selected columns.
    """
    return cast("pd.DataFrame", frame[names])


def as_float(value: object) -> float:
    """Coerce a Series reduction to a float.

    Reductions such as ``Series.std()`` are typed as ``Series | float``
    because the same method on a DataFrame reduces to a Series; on a Series
    the result is always a scalar.

    Args:
        value: The result of a Series reduction.

    Returns:
        The value as a float.
    """
    return float(cast("float", value))


def as_series(value: object) -> pd.Series:
    """Narrow a Series-valued conversion result to a Series.

    Converters like ``pd.to_numeric`` accept scalars, arrays and Series, so
    their return type is a wide union; called on a Series they return a Series.

    Args:
        value: The result of a pandas conversion.

    Returns:
        The value as a Series.
    """
    return cast("pd.Series", value)
