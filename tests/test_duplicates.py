"""
Tests for SmartClean duplicates module.
Run with: pytest tests/test_duplicates.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from smartclean.modules.duplicates import (
    detect_duplicates,
    remove_duplicates,
    count_duplicates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """DataFrame with no duplicates."""
    return pd.DataFrame({
        "name":   ["Alice", "Bob", "Charlie", "Diana"],
        "age":    [30, 25, 35, 28],
        "city":   ["Lagos", "Abuja", "London", "Paris"],
    })


@pytest.fixture
def duped_df() -> pd.DataFrame:
    """DataFrame with 2 duplicate rows appended."""
    base = pd.DataFrame({
        "name":   ["Alice", "Bob", "Charlie", "Diana"],
        "age":    [30, 25, 35, 28],
        "city":   ["Lagos", "Abuja", "London", "Paris"],
    })
    dupes = base.iloc[[0, 2]].copy()  # duplicate Alice and Charlie
    return pd.concat([base, dupes], ignore_index=True)


@pytest.fixture
def subset_duped_df() -> pd.DataFrame:
    """DataFrame where rows share the same name+age but differ in city."""
    return pd.DataFrame({
        "name": ["Alice", "Alice", "Bob", "Bob"],
        "age":  [30, 30, 25, 25],
        "city": ["Lagos", "Abuja", "London", "Paris"],  # different cities
    })


@pytest.fixture
def all_duped_df() -> pd.DataFrame:
    """DataFrame where every row is duplicated exactly once."""
    return pd.DataFrame({
        "name": ["Alice", "Alice", "Bob", "Bob"],
        "age":  [30, 30, 25, 25],
        "city": ["Lagos", "Lagos", "London", "London"],
    })


@pytest.fixture
def titanic() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


@pytest.fixture
def titanic_with_dupes(titanic) -> pd.DataFrame:
    """Titanic with 10 known duplicate rows appended."""
    dupes = titanic.iloc[:10].copy()
    return pd.concat([titanic, dupes], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. detect_duplicates()
# ---------------------------------------------------------------------------

class TestDetectDuplicates:

    def test_returns_boolean_series(self, duped_df):
        result = detect_duplicates(duped_df)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_length_matches_dataframe(self, duped_df):
        result = detect_duplicates(duped_df)
        assert len(result) == len(duped_df)

    def test_no_duplicates_all_false(self, clean_df):
        result = detect_duplicates(clean_df)
        assert result.sum() == 0

    def test_detects_correct_count(self, duped_df):
        result = detect_duplicates(duped_df)
        assert result.sum() == 2

    def test_keep_first_marks_later_occurrence(self, duped_df):
        """With keep='first', the original rows should not be marked."""
        result = detect_duplicates(duped_df, keep="first")
        # First 4 rows are originals — none should be marked
        assert result.iloc[:4].sum() == 0
        # Last 2 rows are duplicates — both should be marked
        assert result.iloc[4:].sum() == 2

    def test_keep_last_marks_earlier_occurrence(self, duped_df):
        """With keep='last', the duplicate rows should not be marked."""
        result = detect_duplicates(duped_df, keep="last")
        # Last 2 rows are the kept ones — not marked
        assert result.iloc[4:].sum() == 0

    def test_keep_false_marks_all_occurrences(self, all_duped_df):
        """With keep=False, all occurrences of duplicates are marked."""
        result = detect_duplicates(all_duped_df, keep=False)
        assert result.sum() == 4  # all 4 rows are duplicates

    def test_subset_detection(self, subset_duped_df):
        """Duplicates on name+age subset, even though city differs."""
        result = detect_duplicates(subset_duped_df, subset=["name", "age"])
        assert result.sum() == 2

    def test_subset_no_duplicates_without_subset(self, subset_duped_df):
        """Full-row comparison — no duplicates since city differs."""
        result = detect_duplicates(subset_duped_df)
        assert result.sum() == 0

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            detect_duplicates([1, 2, 3])

    def test_raises_on_invalid_subset_column(self, clean_df):
        with pytest.raises(ValueError, match="not found"):
            detect_duplicates(clean_df, subset=["nonexistent"])

    def test_does_not_modify_original(self, duped_df):
        original_len = len(duped_df)
        detect_duplicates(duped_df)
        assert len(duped_df) == original_len


# ---------------------------------------------------------------------------
# 2. remove_duplicates()
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:

    def test_returns_dataframe(self, duped_df):
        result = remove_duplicates(duped_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, duped_df):
        original_len = len(duped_df)
        remove_duplicates(duped_df)
        assert len(duped_df) == original_len

    def test_removes_correct_count(self, duped_df):
        result = remove_duplicates(duped_df)
        assert len(result) == len(duped_df) - 2

    def test_no_duplicates_in_result(self, duped_df):
        result = remove_duplicates(duped_df)
        assert result.duplicated().sum() == 0

    def test_clean_df_unchanged(self, clean_df):
        result = remove_duplicates(clean_df)
        assert len(result) == len(clean_df)

    def test_index_is_reset(self, duped_df):
        result = remove_duplicates(duped_df)
        assert list(result.index) == list(range(len(result)))

    def test_keep_first(self, duped_df):
        """Keep='first' — original rows are retained."""
        result = remove_duplicates(duped_df, keep="first")
        assert len(result) == 4
        assert result.iloc[0]["name"] == "Alice"

    def test_keep_last(self, duped_df):
        """Keep='last' — last occurrence is retained."""
        result = remove_duplicates(duped_df, keep="last")
        assert len(result) == 4

    def test_keep_false_removes_all_occurrences(self, all_duped_df):
        """Keep=False — all occurrences of duplicated rows are removed."""
        result = remove_duplicates(all_duped_df, keep=False)
        assert len(result) == 0

    def test_subset_based_removal(self, subset_duped_df):
        """Remove based on name+age subset — keeps first city per person."""
        result = remove_duplicates(subset_duped_df, subset=["name", "age"])
        assert len(result) == 2
        assert result.duplicated(subset=["name", "age"]).sum() == 0

    def test_column_count_preserved(self, duped_df):
        result = remove_duplicates(duped_df)
        assert len(result.columns) == len(duped_df.columns)

    def test_data_values_intact(self, duped_df):
        result = remove_duplicates(duped_df)
        assert "Alice" in result["name"].values
        assert "Bob" in result["name"].values

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            remove_duplicates("not a dataframe")

    def test_raises_on_invalid_subset_column(self, clean_df):
        with pytest.raises(ValueError, match="not found"):
            remove_duplicates(clean_df, subset=["does_not_exist"])

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = remove_duplicates(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result = remove_duplicates(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 3. count_duplicates()
# ---------------------------------------------------------------------------

class TestCountDuplicates:

    def test_returns_integer(self, duped_df):
        result = count_duplicates(duped_df)
        assert isinstance(result, int)

    def test_correct_count(self, duped_df):
        assert count_duplicates(duped_df) == 2

    def test_zero_for_clean_df(self, clean_df):
        assert count_duplicates(clean_df) == 0

    def test_subset_count(self, subset_duped_df):
        assert count_duplicates(subset_duped_df, subset=["name", "age"]) == 2

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            count_duplicates({"a": [1, 2]})


# ---------------------------------------------------------------------------
# 4. Titanic integration
# ---------------------------------------------------------------------------

class TestTitanicDuplicates:

    def test_no_duplicates_in_clean_titanic(self, titanic):
        assert count_duplicates(titanic) == 0

    def test_detects_added_duplicates(self, titanic_with_dupes):
        assert count_duplicates(titanic_with_dupes) == 10

    def test_removes_added_duplicates(self, titanic_with_dupes, titanic):
        result = remove_duplicates(titanic_with_dupes)
        assert len(result) == len(titanic)

    def test_no_duplicates_after_removal(self, titanic_with_dupes):
        result = remove_duplicates(titanic_with_dupes)
        assert result.duplicated().sum() == 0

    def test_row_data_intact_after_removal(self, titanic_with_dupes, titanic):
        result = remove_duplicates(titanic_with_dupes)
        assert result["PassengerId"].tolist() == titanic["PassengerId"].tolist()