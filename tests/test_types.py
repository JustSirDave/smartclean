"""
Tests for SmartClean types module.
Run with: pytest tests/test_types.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from smartclean.profiler import profile
from smartclean.modules.types import (
    fix_types,
    convert_numeric,
    convert_datetime,
    convert_boolean,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def string_numeric_df() -> pd.DataFrame:
    """DataFrame where numeric values are stored as strings."""
    return pd.DataFrame({
        "age":    ["25", "30", "35", "28", "32"],
        "salary": ["50000", "60000", "70000", "55000", "65000"],
        "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    })


@pytest.fixture
def string_bool_df() -> pd.DataFrame:
    """DataFrame where boolean values are stored as strings."""
    return pd.DataFrame({
        "active":    ["true", "false", "true", "false", "true"],
        "verified":  ["yes", "no", "yes", "yes", "no"],
        "name":      ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    })


@pytest.fixture
def string_date_df() -> pd.DataFrame:
    """DataFrame where dates are stored as strings."""
    return pd.DataFrame({
        "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"],
        "name":        ["Alice", "Bob", "Charlie", "Diana"],
    })


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Mixed types needing correction."""
    return pd.DataFrame({
        "age":     ["25", "30", "35"],
        "active":  ["true", "false", "true"],
        "name":    ["Alice", "Bob", "Charlie"],
    })


# ---------------------------------------------------------------------------
# 1. convert_numeric()
# ---------------------------------------------------------------------------

class TestConvertNumeric:

    def test_converts_string_to_numeric(self, string_numeric_df):
        result = convert_numeric(string_numeric_df, columns=["age", "salary"])
        assert pd.api.types.is_numeric_dtype(result["age"])
        assert pd.api.types.is_numeric_dtype(result["salary"])

    def test_correct_values_after_conversion(self, string_numeric_df):
        result = convert_numeric(string_numeric_df, columns=["age"])
        assert result["age"].tolist() == [25, 30, 35, 28, 32]

    def test_does_not_modify_original(self, string_numeric_df):
        original_dtype = string_numeric_df["age"].dtype
        convert_numeric(string_numeric_df, columns=["age"])
        assert string_numeric_df["age"].dtype == original_dtype

    def test_non_numeric_strings_become_nan(self):
        df = pd.DataFrame({"val": ["25", "abc", "30"]})
        result = convert_numeric(df, columns=["val"])
        assert result["val"].isna().sum() == 1

    def test_text_column_not_converted(self, string_numeric_df):
        result = convert_numeric(string_numeric_df, columns=["name"])
        assert result["name"].dtype == object

    def test_applies_to_all_object_columns_if_none_specified(self, string_numeric_df):
        result = convert_numeric(string_numeric_df)
        assert pd.api.types.is_numeric_dtype(result["age"])
        assert pd.api.types.is_numeric_dtype(result["salary"])


# ---------------------------------------------------------------------------
# 2. convert_datetime()
# ---------------------------------------------------------------------------

class TestConvertDatetime:

    def test_converts_string_to_datetime(self, string_date_df):
        result = convert_datetime(string_date_df, columns=["signup_date"])
        assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])

    def test_correct_values_after_conversion(self, string_date_df):
        result = convert_datetime(string_date_df, columns=["signup_date"])
        assert result["signup_date"].iloc[0] == pd.Timestamp("2023-01-15")

    def test_does_not_modify_original(self, string_date_df):
        original_dtype = string_date_df["signup_date"].dtype
        convert_datetime(string_date_df, columns=["signup_date"])
        assert string_date_df["signup_date"].dtype == original_dtype

    def test_non_date_strings_not_converted(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        result = convert_datetime(df, columns=["name"])
        assert result["name"].dtype == object


# ---------------------------------------------------------------------------
# 3. convert_boolean()
# ---------------------------------------------------------------------------

class TestConvertBoolean:

    def test_converts_true_false_strings(self, string_bool_df):
        result = convert_boolean(string_bool_df, columns=["active"])
        assert pd.api.types.is_bool_dtype(result["active"]) or \
               str(result["active"].dtype) == "boolean"

    def test_converts_yes_no_strings(self, string_bool_df):
        result = convert_boolean(string_bool_df, columns=["verified"])
        assert str(result["verified"].dtype) == "boolean"

    def test_true_maps_to_true(self, string_bool_df):
        result = convert_boolean(string_bool_df, columns=["active"])
        assert result["active"].iloc[0] == True

    def test_false_maps_to_false(self, string_bool_df):
        result = convert_boolean(string_bool_df, columns=["active"])
        assert result["active"].iloc[1] == False

    def test_text_column_not_converted(self, string_bool_df):
        result = convert_boolean(string_bool_df, columns=["name"])
        assert result["name"].dtype == object

    def test_does_not_modify_original(self, string_bool_df):
        original_dtype = string_bool_df["active"].dtype
        convert_boolean(string_bool_df, columns=["active"])
        assert string_bool_df["active"].dtype == original_dtype

    def test_mixed_non_bool_not_converted(self):
        """Column with non-boolean values should not be converted."""
        df = pd.DataFrame({"status": ["active", "inactive", "pending"]})
        result = convert_boolean(df, columns=["status"])
        assert result["status"].dtype == object


# ---------------------------------------------------------------------------
# 4. fix_types() — integration
# ---------------------------------------------------------------------------

class TestFixTypes:

    def test_returns_tuple(self, string_numeric_df):
        p = profile(string_numeric_df)
        result = fix_types(string_numeric_df, p)
        assert isinstance(result, tuple)
        conversions, df = result
        assert isinstance(conversions, dict)
        assert isinstance(df, pd.DataFrame)

    def test_does_not_modify_original(self, string_numeric_df):
        original_dtype = string_numeric_df["age"].dtype
        p = profile(string_numeric_df)
        fix_types(string_numeric_df, p)
        assert string_numeric_df["age"].dtype == original_dtype

    def test_logs_converted_columns(self, string_numeric_df):
        p = profile(string_numeric_df)
        conversions, _ = fix_types(string_numeric_df, p)
        assert isinstance(conversions, dict)

    def test_no_conversions_on_clean_df(self):
        df = pd.DataFrame({
            "age":  [25, 30, 35],
            "name": ["Alice", "Bob", "Charlie"],
        })
        p = profile(df)
        conversions, cleaned = fix_types(df, p)
        assert pd.api.types.is_numeric_dtype(cleaned["age"])

    def test_shape_preserved(self, string_numeric_df):
        p = profile(string_numeric_df)
        _, cleaned = fix_types(string_numeric_df, p)
        assert cleaned.shape == string_numeric_df.shape