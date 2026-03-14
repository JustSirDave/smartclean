"""
Tests for SmartClean columns module.
Run with: pytest tests/test_columns.py -v
"""

import pytest
import pandas as pd
from smartclean.modules.columns import snake_case, clean_column_names, snake_case_columns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def messy_df():
    """DataFrame with a variety of messy column name formats."""
    return pd.DataFrame(columns=[
        "First Name",       # spaces
        "TOTAL SALES",      # uppercase + spaces
        "customer-id",      # hyphen
        "customer.id",      # dot
        "  Age  ",          # leading/trailing whitespace
        "100score",         # starts with digit
        "__weird__",        # leading/trailing underscores
        "already_clean",    # already snake_case — should be unchanged
    ])


@pytest.fixture
def titanic():
    """Load the Titanic CSV for real-world column name testing."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. snake_case() — single name conversion
# ---------------------------------------------------------------------------

class TestSnakeCase:

    def test_spaces_to_underscores(self):
        assert snake_case("First Name") == "first_name"

    def test_uppercase_to_lowercase(self):
        assert snake_case("TOTAL SALES") == "total_sales"

    def test_hyphen_to_underscore(self):
        assert snake_case("customer-id") == "customer_id"

    def test_dot_to_underscore(self):
        assert snake_case("customer.id") == "customer_id"

    def test_strips_whitespace(self):
        assert snake_case("  Age  ") == "age"

    def test_digit_prefix(self):
        assert snake_case("100score") == "col_100score"

    def test_strips_leading_trailing_underscores(self):
        assert snake_case("__weird__") == "weird"

    def test_already_clean_unchanged(self):
        assert snake_case("already_clean") == "already_clean"

    def test_collapses_multiple_underscores(self):
        assert snake_case("first__name") == "first_name"

    def test_single_character(self):
        assert snake_case("A") == "a"

    def test_slash_to_underscore(self):
        assert snake_case("revenue/cost") == "revenue_cost"

    def test_mixed_separators(self):
        assert snake_case("Total - Sales.Amount") == "total_sales_amount"

    def test_empty_string_returns_unnamed(self):
        assert snake_case("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        assert snake_case("!!!") == "unnamed"

    def test_numeric_string(self):
        result = snake_case("123")
        assert result.startswith("col_")

    def test_returns_string(self):
        assert isinstance(snake_case("Any Column"), str)

    def test_non_string_input_handled(self):
        # Column names can sometimes be integers in pandas
        assert isinstance(snake_case(42), str)


# ---------------------------------------------------------------------------
# 2. clean_column_names() — DataFrame transformation
# ---------------------------------------------------------------------------

class TestCleanColumnNames:

    def test_returns_dataframe(self, messy_df):
        result = clean_column_names(messy_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, messy_df):
        original_cols = list(messy_df.columns)
        clean_column_names(messy_df)
        assert list(messy_df.columns) == original_cols

    def test_column_count_preserved(self, messy_df):
        result = clean_column_names(messy_df)
        assert len(result.columns) == len(messy_df.columns)

    def test_first_name_normalised(self, messy_df):
        result = clean_column_names(messy_df)
        assert "first_name" in result.columns

    def test_total_sales_normalised(self, messy_df):
        result = clean_column_names(messy_df)
        assert "total_sales" in result.columns

    def test_customer_hyphen_id_normalised(self, messy_df):
        result = clean_column_names(messy_df)
        assert "customer_id" in result.columns

    def test_all_columns_lowercase(self, messy_df):
        result = clean_column_names(messy_df)
        for col in result.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_no_spaces_in_columns(self, messy_df):
        result = clean_column_names(messy_df)
        for col in result.columns:
            assert " " not in col, f"Column '{col}' contains a space"

    def test_no_hyphens_in_columns(self, messy_df):
        result = clean_column_names(messy_df)
        for col in result.columns:
            assert "-" not in col, f"Column '{col}' contains a hyphen"

    def test_no_dots_in_columns(self, messy_df):
        result = clean_column_names(messy_df)
        for col in result.columns:
            assert "." not in col, f"Column '{col}' contains a dot"

    def test_no_column_starts_with_digit(self, messy_df):
        result = clean_column_names(messy_df)
        for col in result.columns:
            assert not col[0].isdigit(), f"Column '{col}' starts with a digit"

    def test_already_clean_column_unchanged(self, messy_df):
        result = clean_column_names(messy_df)
        assert "already_clean" in result.columns

    def test_duplicate_columns_after_normalisation(self):
        """'Name' and 'name' both normalise to 'name' — should be deduplicated."""
        df = pd.DataFrame(columns=["Name", "name", "NAME"])
        result = clean_column_names(df)
        assert len(result.columns) == len(set(result.columns)), \
            "Duplicate column names after normalisation"

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            clean_column_names({"col": [1, 2, 3]})

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = clean_column_names(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0

    def test_data_values_preserved(self):
        """Column renaming must not affect the data values."""
        df = pd.DataFrame({"First Name": ["Alice", "Bob"], "Age": [30, 25]})
        result = clean_column_names(df)
        assert result["first_name"].tolist() == ["Alice", "Bob"]
        assert result["age"].tolist() == [30, 25]


# ---------------------------------------------------------------------------
# 3. Titanic real-world test
# ---------------------------------------------------------------------------

class TestTitanicColumns:

    def test_all_titanic_columns_are_clean(self, titanic):
        result = clean_column_names(titanic)
        for col in result.columns:
            assert " " not in col
            assert col == col.lower()
            assert "-" not in col

    def test_titanic_column_count_preserved(self, titanic):
        result = clean_column_names(titanic)
        assert len(result.columns) == len(titanic.columns)

    def test_titanic_data_shape_preserved(self, titanic):
        result = clean_column_names(titanic)
        assert result.shape == titanic.shape


# ---------------------------------------------------------------------------
# 4. snake_case_columns() alias
# ---------------------------------------------------------------------------

class TestSnakeCaseColumnsAlias:

    def test_alias_produces_same_result(self, messy_df):
        result_a = clean_column_names(messy_df)
        result_b = snake_case_columns(messy_df)
        assert list(result_a.columns) == list(result_b.columns)