"""
Tests for SmartClean text module.
Run with: pytest tests/test_text.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

from smartclean.profiler import profile
from smartclean.modules.text import (
    clean_text,
    strip_whitespace,
    normalize_case,
    remove_special_chars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def messy_text_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name":    ["  alice  ", "BOB", "  Charlie  "],
        "city":    [" lagos ", "ABUJA", " london "],
        "gender":  ["female", "MALE", "Female"],
        "age":     [30, 25, 35],
    })


@pytest.fixture
def special_chars_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name":   ["Alice!", "Bob@work", "Charlie#3"],
        "notes":  ["good.", "bad!", "ok?"],
    })


@pytest.fixture
def titanic() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. strip_whitespace()
# ---------------------------------------------------------------------------

class TestStripWhitespace:

    def test_strips_leading_trailing_spaces(self, messy_text_df):
        result = strip_whitespace(messy_text_df, columns=["name"])
        assert result["name"].iloc[0] == "alice"
        assert result["name"].iloc[2] == "Charlie"

    def test_does_not_modify_original(self, messy_text_df):
        original = messy_text_df["name"].iloc[0]
        strip_whitespace(messy_text_df, columns=["name"])
        assert messy_text_df["name"].iloc[0] == original

    def test_applies_to_all_object_columns_if_none(self, messy_text_df):
        result = strip_whitespace(messy_text_df)
        assert result["name"].iloc[0] == "alice"
        assert result["city"].iloc[0] == "lagos"

    def test_numeric_columns_unaffected(self, messy_text_df):
        result = strip_whitespace(messy_text_df)
        assert result["age"].tolist() == [30, 25, 35]

    def test_none_values_unaffected(self):
        df = pd.DataFrame({"name": ["  Alice  ", None, "  Bob  "]})
        result = strip_whitespace(df, columns=["name"])
        assert result["name"].iloc[1] is None or pd.isna(result["name"].iloc[1])
        assert result["name"].iloc[0] == "Alice"


# ---------------------------------------------------------------------------
# 2. normalize_case()
# ---------------------------------------------------------------------------

class TestNormalizeCase:

    def test_title_case_default(self, messy_text_df):
        result = normalize_case(messy_text_df, columns=["name"])
        assert result["name"].iloc[1] == "Bob"
        assert result["gender"].iloc[0] == "female"  # unchanged (not in columns)

    def test_lower_case(self, messy_text_df):
        result = normalize_case(messy_text_df, columns=["gender"], case="lower")
        assert result["gender"].iloc[1] == "male"

    def test_upper_case(self, messy_text_df):
        result = normalize_case(messy_text_df, columns=["city"], case="upper")
        assert result["city"].iloc[1] == "ABUJA"

    def test_invalid_case_raises(self, messy_text_df):
        with pytest.raises(ValueError, match="Invalid case"):
            normalize_case(messy_text_df, columns=["name"], case="camel")

    def test_does_not_modify_original(self, messy_text_df):
        original = messy_text_df["name"].iloc[1]
        normalize_case(messy_text_df, columns=["name"])
        assert messy_text_df["name"].iloc[1] == original

    def test_numeric_columns_unaffected(self, messy_text_df):
        result = normalize_case(messy_text_df)
        assert result["age"].tolist() == [30, 25, 35]

    def test_none_values_unaffected(self):
        df = pd.DataFrame({"name": ["alice", None, "bob"]})
        result = normalize_case(df, columns=["name"])
        assert pd.isna(result["name"].iloc[1])


# ---------------------------------------------------------------------------
# 3. remove_special_chars()
# ---------------------------------------------------------------------------

class TestRemoveSpecialChars:

    def test_removes_special_characters(self, special_chars_df):
        result = remove_special_chars(special_chars_df, columns=["name"])
        assert "!" not in result["name"].iloc[0]
        assert "@" not in result["name"].iloc[1]
        assert "#" not in result["name"].iloc[2]

    def test_keeps_spaces_by_default(self):
        df = pd.DataFrame({"name": ["hello world!"]})
        result = remove_special_chars(df, columns=["name"])
        assert " " in result["name"].iloc[0]

    def test_removes_spaces_when_keep_spaces_false(self):
        df = pd.DataFrame({"name": ["hello world!"]})
        result = remove_special_chars(df, columns=["name"], keep_spaces=False)
        assert " " not in result["name"].iloc[0]

    def test_does_not_modify_original(self, special_chars_df):
        original = special_chars_df["name"].iloc[0]
        remove_special_chars(special_chars_df, columns=["name"])
        assert special_chars_df["name"].iloc[0] == original

    def test_alphanumeric_preserved(self, special_chars_df):
        result = remove_special_chars(special_chars_df, columns=["name"])
        assert "Alice" in result["name"].iloc[0]
        assert "Bob" in result["name"].iloc[1]


# ---------------------------------------------------------------------------
# 4. clean_text() — main function
# ---------------------------------------------------------------------------

class TestCleanText:

    def test_returns_tuple(self, messy_text_df):
        p = profile(messy_text_df)
        result = clean_text(messy_text_df, p)
        assert isinstance(result, tuple)
        cols, df = result
        assert isinstance(cols, list)
        assert isinstance(df, pd.DataFrame)

    def test_does_not_modify_original(self, messy_text_df):
        original = messy_text_df["name"].iloc[0]
        p = profile(messy_text_df)
        clean_text(messy_text_df, p)
        assert messy_text_df["name"].iloc[0] == original

    def test_strips_whitespace_by_default(self, messy_text_df):
        p = profile(messy_text_df)
        _, cleaned = clean_text(messy_text_df, p)
        assert cleaned["name"].iloc[0] == "Alice"
        assert cleaned["city"].iloc[0] == "Lagos"

    def test_normalizes_case_by_default(self, messy_text_df):
        p = profile(messy_text_df)
        _, cleaned = clean_text(messy_text_df, p)
        assert cleaned["gender"].iloc[1] == "Male"

    def test_logs_cleaned_columns(self, messy_text_df):
        p = profile(messy_text_df)
        cols, _ = clean_text(messy_text_df, p)
        assert len(cols) > 0

    def test_numeric_columns_not_in_cleaned_list(self, messy_text_df):
        p = profile(messy_text_df)
        cols, _ = clean_text(messy_text_df, p)
        assert "age" not in cols

    def test_specific_columns_only(self, messy_text_df):
        p = profile(messy_text_df)
        _, cleaned = clean_text(messy_text_df, p, columns=["name"])
        # city should be unchanged
        assert cleaned["city"].iloc[0] == " lagos "

    def test_invalid_column_raises(self, messy_text_df):
        p = profile(messy_text_df)
        with pytest.raises(ValueError, match="not found"):
            clean_text(messy_text_df, p, columns=["nonexistent"])

    def test_remove_special_chars_off_by_default(self, special_chars_df):
        p = profile(special_chars_df)
        _, cleaned = clean_text(special_chars_df, p, remove_special_chars=False)
        assert "!" in cleaned["name"].iloc[0]

    def test_remove_special_chars_opt_in(self, special_chars_df):
        p = profile(special_chars_df)
        _, cleaned = clean_text(
            special_chars_df, p, remove_special_chars=True
        )
        assert "!" not in cleaned["name"].iloc[0]


# ---------------------------------------------------------------------------
# 5. Titanic integration
# ---------------------------------------------------------------------------

class TestTitanicText:

    def test_sex_column_normalised(self, titanic):
        p = profile(titanic)
        _, cleaned = clean_text(titanic, p)
        unique_vals = cleaned["Sex"].dropna().unique()
        for val in unique_vals:
            assert val == val.strip()

    def test_shape_preserved(self, titanic):
        p = profile(titanic)
        _, cleaned = clean_text(titanic, p)
        assert cleaned.shape[0] == titanic.shape[0]