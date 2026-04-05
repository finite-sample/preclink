"""Pytest fixtures for tether tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_left_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "first_name": ["John", "Jane", "Bob", "Alice"],
            "last_name": ["Smith", "Doe", "Johnson", "Williams"],
            "state": ["CA", "NY", "CA", "TX"],
            "dob": ["1990-01-15", "1985-06-20", "1992-03-10", "1988-12-05"],
        }
    )


@pytest.fixture
def sample_right_df():
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "first_name": ["Jon", "Jane", "Robert", "Alice", "Charlie"],
            "last_name": ["Smith", "Doe", "Johnson", "Williams", "Brown"],
            "state": ["CA", "NY", "CA", "TX", "FL"],
            "dob": ["1990-01-15", "1985-06-20", "1992-03-10", "1988-12-05", "1995-07-22"],
        }
    )


@pytest.fixture
def exact_match_left():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "city": ["NYC", "LA"],
        }
    )


@pytest.fixture
def exact_match_right():
    return pd.DataFrame(
        {
            "id": [101, 102],
            "name": ["Alice", "Bob"],
            "city": ["NYC", "LA"],
        }
    )


@pytest.fixture
def no_match_left():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "state": ["CA", "NY"],
        }
    )


@pytest.fixture
def no_match_right():
    return pd.DataFrame(
        {
            "id": [101, 102],
            "name": ["Charlie", "David"],
            "state": ["TX", "FL"],
        }
    )
