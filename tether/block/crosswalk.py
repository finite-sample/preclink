"""Crosswalk validation and application."""

import pandas as pd


class Crosswalk:
    """Mapping between blocking key values."""

    def __init__(self, mapping: dict[str, str]) -> None:
        """Initialize crosswalk with value mapping.

        Args:
            mapping: Dictionary of value mappings.
        """
        self.mapping = mapping

    def apply(self, series: pd.Series) -> pd.Series:
        """Apply crosswalk mapping to a series.

        Args:
            series: Series of values to normalize.

        Returns:
            Series with mapped values.
        """
        return series.map(lambda x: self.mapping.get(str(x), x) if pd.notna(x) else x)

    def validate(self) -> list[str]:
        """Validate the crosswalk mapping.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []
        for key, value in self.mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                errors.append(f"Non-string key or value: {key} -> {value}")
        return errors
