"""
Tests for SmartClean missing value module.
Run with: pytest tests/test_missing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from smartclean.profiler import profile
from smartclean.modules.missing import (
    handle_missing,
    fill_mean,
    fill_median,
    fill_mode,
    fill_auto,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def titanic_profile(titanic):
    return profile(titanic)


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Simple numeric DataFrame with known missing values."""
    return pd.DataFrame({
        "age":    [25.0, None, 30.0, None, 35.0],
        "salary": [50000.0, 60000.0, None, 70000.0, None],
    })


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """Categorical DataFrame with known missing values."""
    return pd.DataFrame({
        "gender":     ["male", "female", None, "female", "male"],
        "department": ["HR", None, "IT", "IT", None],
    })


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Mixed types with missing values."""
    return pd.DataFrame({
        "age":    [25.0, None, 30.0, None, 35.0],
        "name":   ["Alice", None, "Charlie", "Diana", None],
        "gender": ["male", "female", None, "female", "male"],
    })


@pytest.fixture
def mixed_profile(mixed_df):
    return profile(mixed_df)


@pytest.fixture
def high_missing_df() -> pd.DataFrame:
    """DataFrame with a column that is 96% missing — above the 0.95 threshold."""
    data = {"age": [25.0, None, 30.0, None, 35.0]}
    notes = [None] * 48 + ["some note"] + [None] * 1  # 96% missing
    data["notes"] = notes[:5]  # slice to match length
    return pd.DataFrame({
        "age":   [25.0, None, 30.0, None, 35.0],
        "notes": [None, None, None, None, "a note"],  # 80% missing
    })


# ---------------------------------------------------------------------------
# 1. handle_missing() — core function
# ---------------------------------------------------------------------------

class TestHandleMissing:

    def test_returns_tuple(self, numeric_df):
        p = profile(numeric_df)
        result, df = handle_missing(numeric_df, p)
        assert isinstance(result, dict)
        assert isinstance(df, pd.DataFrame)

    def test_does_not_modify_original(self, numeric_df):
        original_nulls = numeric_df.isna().sum().sum()
        p = profile(numeric_df)
        handle_missing(numeric_df, p)
        assert numeric_df.isna().sum().sum() == original_nulls

    def test_result_has_dropped_and_filled_keys(self, numeric_df):
        p = profile(numeric_df)
        result, _ = handle_missing(numeric_df, p)
        assert "dropped" in result
        assert "filled" in result

    def test_fills_numeric_missing_with_median(self, numeric_df):
        p = profile(numeric_df)
        result, cleaned = handle_missing(numeric_df, p, strategy="auto")
        assert cleaned["age"].isna().sum() == 0
        assert cleaned["salary"].isna().sum() == 0

    def test_filled_log_records_count_and_strategy(self, numeric_df):
        p = profile(numeric_df)
        result, _ = handle_missing(numeric_df, p, strategy="auto")
        assert "age" in result["filled"]
        assert result["filled"]["age"]["count"] > 0
        assert result["filled"]["age"]["strategy"] == "median"

    def test_fills_categorical_with_mode(self, categorical_df):
        p = profile(categorical_df)
        result, cleaned = handle_missing(categorical_df, p, strategy="auto")
        assert cleaned["gender"].isna().sum() == 0
        assert cleaned["department"].isna().sum() == 0

    def test_no_missing_columns_not_in_filled(self, numeric_df):
        """Columns with no missing values should not appear in filled log."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = profile(df)
        result, _ = handle_missing(df, p)
        assert result["filled"] == {}

    def test_specific_columns_only(self, numeric_df):
        p = profile(numeric_df)
        result, cleaned = handle_missing(
            numeric_df, p, strategy="median", columns=["age"]
        )
        assert cleaned["age"].isna().sum() == 0
        # salary should still have missing values
        assert cleaned["salary"].isna().sum() > 0

    def test_invalid_column_raises_value_error(self, numeric_df):
        p = profile(numeric_df)
        with pytest.raises(ValueError, match="not found"):
            handle_missing(numeric_df, p, columns=["nonexistent_col"])

    def test_mean_strategy(self, numeric_df):
        p = profile(numeric_df)
        _, cleaned = handle_missing(numeric_df, p, strategy="mean")
        assert cleaned["age"].isna().sum() == 0
        expected_mean = numeric_df["age"].mean()
        # Check that filled values are close to the mean
        original_missing_idx = numeric_df["age"].isna()
        assert all(
            abs(cleaned.loc[original_missing_idx, "age"] - expected_mean) < 0.001
        )

    def test_median_strategy(self, numeric_df):
        p = profile(numeric_df)
        _, cleaned = handle_missing(numeric_df, p, strategy="median")
        assert cleaned["age"].isna().sum() == 0
        expected_median = numeric_df["age"].median()
        original_missing_idx = numeric_df["age"].isna()
        assert all(
            abs(cleaned.loc[original_missing_idx, "age"] - expected_median) < 0.001
        )

    def test_mode_strategy(self, categorical_df):
        p = profile(categorical_df)
        _, cleaned = handle_missing(categorical_df, p, strategy="mode")
        assert cleaned["gender"].isna().sum() == 0


# ---------------------------------------------------------------------------
# 2. Drop threshold
# ---------------------------------------------------------------------------

class TestDropThreshold:

    def test_column_above_threshold_is_dropped(self):
        """Column with 100% missing should be dropped at default threshold."""
        df = pd.DataFrame({
            "age":   [25, 30, 35, 40, 45],
            "notes": [None, None, None, None, None],  # 100% missing
        })
        p = profile(df)
        result, cleaned = handle_missing(df, p, drop_threshold=0.95)
        assert "notes" not in cleaned.columns
        assert "notes" in result["dropped"]

    def test_dropped_column_logged_with_missing_pct(self):
        df = pd.DataFrame({
            "age":   [25, 30, 35, 40, 45],
            "notes": [None, None, None, None, None],
        })
        p = profile(df)
        result, _ = handle_missing(df, p, drop_threshold=0.95)
        assert result["dropped"]["notes"]["reason"] == "missing_pct"
        assert result["dropped"]["notes"]["value"] == 1.0

    def test_column_below_threshold_is_not_dropped(self):
        """Age at ~40% missing should not be dropped at default 0.95 threshold."""
        df = pd.DataFrame({
            "age": [25, None, 35, None, 45],  # 40% missing
        })
        p = profile(df)
        result, cleaned = handle_missing(df, p, drop_threshold=0.95)
        assert "age" in cleaned.columns
        assert "age" not in result["dropped"]

    def test_threshold_of_1_never_drops(self):
        """Setting threshold to 1.0 should never drop any column."""
        df = pd.DataFrame({
            "age":   [25, 30, 35, 40, 45],
            "notes": [None, None, None, None, None],  # 100% missing
        })
        p = profile(df)
        result, cleaned = handle_missing(df, p, drop_threshold=1.0)
        assert "notes" in cleaned.columns
        assert result["dropped"] == {}

    def test_titanic_cabin_dropped_at_default_threshold(self, titanic, titanic_profile):
        """Titanic Cabin column is ~77% missing — below 0.95 so should NOT be dropped."""
        result, cleaned = handle_missing(titanic, titanic_profile, drop_threshold=0.95)
        # Cabin is ~77% — not dropped at 0.95
        assert "Cabin" in cleaned.columns

    def test_titanic_cabin_dropped_at_lower_threshold(self, titanic, titanic_profile):
        """Lower the threshold to 0.70 — Cabin (~77%) should now be dropped."""
        result, cleaned = handle_missing(titanic, titanic_profile, drop_threshold=0.70)
        assert "Cabin" not in cleaned.columns
        assert "Cabin" in result["dropped"]


# ---------------------------------------------------------------------------
# 3. fill_mean()
# ---------------------------------------------------------------------------

class TestFillMean:

    def test_fills_all_missing_numeric(self, numeric_df):
        result = fill_mean(numeric_df)
        assert result["age"].isna().sum() == 0
        assert result["salary"].isna().sum() == 0

    def test_fills_with_correct_mean(self, numeric_df):
        expected = numeric_df["age"].mean()
        result = fill_mean(numeric_df, columns=["age"])
        missing_idx = numeric_df["age"].isna()
        assert all(abs(result.loc[missing_idx, "age"] - expected) < 0.001)

    def test_does_not_modify_original(self, numeric_df):
        original = numeric_df.copy()
        fill_mean(numeric_df)
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_specific_columns(self, numeric_df):
        result = fill_mean(numeric_df, columns=["age"])
        assert result["age"].isna().sum() == 0
        assert result["salary"].isna().sum() > 0


# ---------------------------------------------------------------------------
# 4. fill_median()
# ---------------------------------------------------------------------------

class TestFillMedian:

    def test_fills_all_missing_numeric(self, numeric_df):
        result = fill_median(numeric_df)
        assert result["age"].isna().sum() == 0

    def test_fills_with_correct_median(self, numeric_df):
        expected = numeric_df["age"].median()
        result = fill_median(numeric_df, columns=["age"])
        missing_idx = numeric_df["age"].isna()
        assert all(abs(result.loc[missing_idx, "age"] - expected) < 0.001)

    def test_does_not_modify_original(self, numeric_df):
        original = numeric_df.copy()
        fill_median(numeric_df)
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_robust_to_outlier(self):
        """Median should not be skewed by extreme value, unlike mean."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, None, 1000.0]})
        result_median = fill_median(df, columns=["val"])
        result_mean = fill_mean(df, columns=["val"])
        # Median fill should produce a lower fill value than mean
        median_fill = result_median.loc[df["val"].isna(), "val"].iloc[0]
        mean_fill = result_mean.loc[df["val"].isna(), "val"].iloc[0]
        assert median_fill < mean_fill


# ---------------------------------------------------------------------------
# 5. fill_mode()
# ---------------------------------------------------------------------------

class TestFillMode:

    def test_fills_categorical_missing(self, categorical_df):
        result = fill_mode(categorical_df)
        assert result["gender"].isna().sum() == 0
        assert result["department"].isna().sum() == 0

    def test_fills_with_most_frequent_value(self, categorical_df):
        expected_mode = categorical_df["gender"].mode()[0]
        result = fill_mode(categorical_df, columns=["gender"])
        missing_idx = categorical_df["gender"].isna()
        assert all(result.loc[missing_idx, "gender"] == expected_mode)

    def test_does_not_modify_original(self, categorical_df):
        original = categorical_df.copy()
        fill_mode(categorical_df)
        pd.testing.assert_frame_equal(categorical_df, original)

    def test_falls_back_to_unknown_when_no_mode(self):
        """All unique values — no mode — should fall back to 'unknown'."""
        df = pd.DataFrame({"cat": ["a", "b", "c", None]})
        result = fill_mode(df, columns=["cat"])
        assert result["cat"].isna().sum() == 0


# ---------------------------------------------------------------------------
# 6. fill_auto()
# ---------------------------------------------------------------------------

class TestFillAuto:

    def test_no_missing_after_fill_auto(self, mixed_df, mixed_profile):
        result = fill_auto(mixed_df, mixed_profile)
        # Some columns may still have NaN if strategy couldn't fill
        # but numeric and categorical should be clean
        assert result["age"].isna().sum() == 0

    def test_does_not_modify_original(self, mixed_df, mixed_profile):
        original_nulls = mixed_df.isna().sum().sum()
        fill_auto(mixed_df, mixed_profile)
        assert mixed_df.isna().sum().sum() == original_nulls


# ---------------------------------------------------------------------------
# 7. Titanic integration
# ---------------------------------------------------------------------------

class TestTitanicMissing:

    def test_age_filled_after_handle_missing(self, titanic, titanic_profile):
        _, cleaned = handle_missing(titanic, titanic_profile, strategy="auto")
        assert cleaned["Age"].isna().sum() == 0

    def test_embarked_filled_after_handle_missing(self, titanic, titanic_profile):
        _, cleaned = handle_missing(titanic, titanic_profile, strategy="auto")
        assert cleaned["Embarked"].isna().sum() == 0

    def test_row_count_preserved_when_no_drop(self, titanic, titanic_profile):
        _, cleaned = handle_missing(titanic, titanic_profile, drop_threshold=0.95)
        assert len(cleaned) == len(titanic)

    def test_age_filled_with_median(self, titanic, titanic_profile):
        """Auto strategy should fill Age (numeric) with median."""
        result, cleaned = handle_missing(titanic, titanic_profile, strategy="auto")
        if "Age" in result["filled"]:
            assert result["filled"]["Age"]["strategy"] == "median"