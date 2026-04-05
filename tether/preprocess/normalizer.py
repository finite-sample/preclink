"""Text normalization preprocessing."""

import re
import unicodedata

import pandas as pd


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
            cols_to_process = result.select_dtypes(include=["object", "string"]).columns.tolist()

        for col in cols_to_process:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: self._normalize_string(str(x)) if pd.notna(x) else x
                )

        return result
