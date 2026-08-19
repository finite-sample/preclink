"""Text normalization preprocessing."""

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


@dataclass
class CompletenessReport:
    """Report of records dropped due to incomplete data."""

    dropped_indices: list[int]
    dropped_count: int
    total_count: int
    completeness_scores: pd.Series

    @property
    def retained_count(self) -> int:
        """Number of records retained."""
        return self.total_count - self.dropped_count

    @property
    def drop_rate(self) -> float:
        """Fraction of records dropped."""
        return self.dropped_count / self.total_count if self.total_count > 0 else 0.0


class CompletenessFilter:
    """Filter records based on data completeness.

    Records with completeness below the threshold are dropped.
    Completeness is measured as the fraction of required columns
    that have non-null, non-empty values.
    """

    def __init__(
        self,
        min_completeness: float = 0.0,
        required_columns: list[str] | None = None,
    ) -> None:
        """Initialize completeness filter.

        Args:
            min_completeness: Minimum completeness threshold (0.0 to 1.0).
                Records with completeness below this threshold are dropped.
            required_columns: Columns to check for completeness.
                If None, all columns are checked.
        """
        if not 0.0 <= min_completeness <= 1.0:
            msg = "min_completeness must be between 0.0 and 1.0"
            raise ValueError(msg)
        self.min_completeness = min_completeness
        self.required_columns = required_columns

    def _compute_completeness(self, df: pd.DataFrame) -> pd.Series:
        """Compute completeness score for each record.

        Args:
            df: Input DataFrame.

        Returns:
            Series of completeness scores between 0 and 1.
        """
        if self.required_columns is not None:
            cols = [c for c in self.required_columns if c in df.columns]
        else:
            cols = list(df.columns)

        if not cols:
            return pd.Series(1.0, index=df.index)

        subset = df[cols]
        non_null = subset.notna()
        non_empty = subset.apply(
            lambda col: col.apply(lambda x: x != "" if isinstance(x, str) else True)
        )
        valid = non_null & non_empty
        completeness: pd.Series = valid.sum(axis=1) / len(cols)
        return completeness

    def filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CompletenessReport]:
        """Filter records based on completeness.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (filtered DataFrame, completeness report).
        """
        completeness_scores = self._compute_completeness(df)
        mask = completeness_scores >= self.min_completeness
        dropped_indices = list(df.index[~mask])

        report = CompletenessReport(
            dropped_indices=dropped_indices,
            dropped_count=len(dropped_indices),
            total_count=len(df),
            completeness_scores=completeness_scores,
        )

        filtered_df: pd.DataFrame = df.loc[mask].copy()
        return filtered_df, report


class TextNormalizer:
    """Normalize text columns in a DataFrame."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        collapse_whitespace: bool = True,
        columns: list[str] | None = None,
    ) -> None:
        """Initialize text normalizer.

        Args:
            normalize_unicode: Normalize unicode characters.
            lowercase: Convert to lowercase.
            strip_whitespace: Strip whitespace.
            collapse_whitespace: Collapse multiple whitespace.
            columns: Columns to process.
        """
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.collapse_whitespace = collapse_whitespace
        self.columns = columns

    def _normalize_string(self, value: str) -> str:
        """Normalize a single string value.

        Args:
            value: String to normalize.

        Returns:
            Normalized string.
        """
        if self.normalize_unicode:
            value = unicodedata.normalize("NFKC", value)

        if self.lowercase:
            value = value.lower()

        if self.strip_whitespace:
            value = value.strip()

        if self.collapse_whitespace:
            value = re.sub(r"\s+", " ", value)

        return value

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns in the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with normalized text columns.
        """
        result = df.copy()

        if self.columns is not None:
            cols_to_process = self.columns
        else:
            cols_to_process = result.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

        for col in cols_to_process:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: self._normalize_string(str(x)) if pd.notna(x) else x
                )

        return result
