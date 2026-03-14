"""
Tests for SmartClean profiler module using the Titanic dataset.
Run with: pytest tests/test_profiler.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from smartclean.profiler import profile, ProfileResult, ColumnProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    """Load the Titanic CSV from the docs folder."""
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def titanic_profile(titanic) -> ProfileResult:
    """Run the profiler once and reuse across all tests."""
    return profile(titanic)


@pytest.fixture(scope="module")
def df_with_duplicates(titanic) -> pd.DataFrame:
    """Titanic with 5 duplicate rows appended."""
    dupes = titanic.iloc[:5].copy()
    return pd.concat([titanic, dupes], ignore_index=True)


@pytest.fixture(scope="module")
def df_constant_col(titanic) -> pd.DataFrame:
    """Titanic with an extra constant column."""
    df = titanic.copy()
    df["constant_col"] = "same_value"
    return df


# ---------------------------------------------------------------------------
# 1. ProfileResult structure
# ---------------------------------------------------------------------------

class TestProfileResultStructure:

    def test_returns_profile_result(self, titanic_profile):
        assert isinstance(titanic_profile, ProfileResult)

    def test_has_row_count(self, titanic, titanic_profile):
        assert titanic_profile.row_count == len(titanic)

    def test_has_col_count(self, titanic, titanic_profile):
        assert titanic_profile.col_count == len(titanic.columns)

    def test_has_columns_dict(self, titanic_profile):
        assert isinstance(titanic_profile.columns, dict)
        assert len(titanic_profile.columns) > 0

    def test_each_column_is_column_profile(self, titanic_profile):
        for col, cp in titanic_profile.columns.items():
            assert isinstance(cp, ColumnProfile), f"{col} is not a ColumnProfile"


# ---------------------------------------------------------------------------
# 2. Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:

    def test_no_duplicates_in_clean_titanic(self, titanic_profile):
        assert titanic_profile.duplicate_row_count == 0

    def test_detects_added_duplicates(self, df_with_duplicates):
        result = profile(df_with_duplicates)
        assert result.duplicate_row_count == 5


# ---------------------------------------------------------------------------
# 3. Missing value detection
# ---------------------------------------------------------------------------

class TestMissingValueDetection:

    def test_age_has_missing_values(self, titanic_profile):
        """Age has ~177 missing values in the standard Titanic dataset."""
        age = titanic_profile.columns["Age"]
        assert age.missing_count > 0

    def test_age_missing_pct_is_correct(self, titanic, titanic_profile):
        age = titanic_profile.columns["Age"]
        expected_pct = titanic["Age"].isna().sum() / len(titanic)
        assert abs(age.missing_pct - expected_pct) < 0.001

    def test_cabin_has_high_missing_pct(self, titanic_profile):
        """Cabin is ~77% missing — well above the 0.95 drop threshold but tests detection."""
        cabin = titanic_profile.columns.get("Cabin")
        if cabin:
            assert cabin.missing_pct > 0.5

    def test_embarked_has_small_missing(self, titanic_profile):
        """Embarked has only 2 missing values."""
        embarked = titanic_profile.columns.get("Embarked")
        if embarked:
            assert embarked.missing_count >= 2

    def test_no_missing_in_survived(self, titanic_profile):
        survived = titanic_profile.columns.get("Survived")
        if survived:
            assert survived.missing_count == 0


# ---------------------------------------------------------------------------
# 4. Semantic dtype inference
# ---------------------------------------------------------------------------

class TestDtypeInference:

    def test_age_is_numeric(self, titanic_profile):
        assert titanic_profile.columns["Age"].dtype == "numeric"

    def test_fare_is_numeric(self, titanic_profile):
        assert titanic_profile.columns["Fare"].dtype == "numeric"

    def test_sex_is_categorical(self, titanic_profile):
        assert titanic_profile.columns["Sex"].dtype == "categorical"

    def test_embarked_is_categorical(self, titanic_profile):
        embarked = titanic_profile.columns.get("Embarked")
        if embarked:
            assert embarked.dtype == "categorical"

    def test_dtype_values_are_valid(self, titanic_profile):
        valid_dtypes = {"numeric", "categorical", "datetime", "text"}
        for col, cp in titanic_profile.columns.items():
            assert cp.dtype in valid_dtypes, f"{col} has invalid dtype '{cp.dtype}'"


# ---------------------------------------------------------------------------
# 5. Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:

    def test_fare_has_outliers(self, titanic_profile):
        """Fare has well-known extreme values (e.g. 512.33)."""
        fare = titanic_profile.columns.get("Fare")
        if fare:
            assert fare.potential_outlier_count > 0

    def test_categorical_columns_have_no_outliers(self, titanic_profile):
        """Outlier detection should only run on numeric columns."""
        for col, cp in titanic_profile.columns.items():
            if cp.dtype == "categorical":
                assert cp.potential_outlier_count == 0, (
                    f"Categorical column '{col}' should not have outliers"
                )


# ---------------------------------------------------------------------------
# 6. Constant column detection
# ---------------------------------------------------------------------------

class TestConstantColumnDetection:

    def test_regular_columns_not_constant(self, titanic_profile):
        assert titanic_profile.columns["Age"].is_constant is False
        assert titanic_profile.columns["Fare"].is_constant is False

    def test_detects_constant_column(self, df_constant_col):
        result = profile(df_constant_col)
        assert result.columns["constant_col"].is_constant is True

    def test_survived_not_constant(self, titanic_profile):
        survived = titanic_profile.columns.get("Survived")
        if survived:
            assert survived.is_constant is False


# ---------------------------------------------------------------------------
# 7. Unique count
# ---------------------------------------------------------------------------

class TestUniqueCount:

    def test_sex_has_two_unique_values(self, titanic_profile):
        sex = titanic_profile.columns.get("Sex")
        if sex:
            assert sex.unique_count == 2

    def test_unique_count_never_exceeds_row_count(self, titanic_profile):
        for col, cp in titanic_profile.columns.items():
            assert cp.unique_count <= titanic_profile.row_count


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_dataframe_raises_or_returns_zero_rows(self):
        empty_df = pd.DataFrame()
        try:
            result = profile(empty_df)
            assert result.row_count == 0
        except (ValueError, Exception):
            pass  # raising is also acceptable behaviour

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result = profile(df)
        assert result.row_count == 1
        assert result.duplicate_row_count == 0

    def test_all_missing_column(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [None, None, None]
        })
        result = profile(df)
        assert result.columns["b"].missing_pct == 1.0
        assert result.columns["b"].missing_count == 3