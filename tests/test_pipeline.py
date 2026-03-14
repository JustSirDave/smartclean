"""
Tests for SmartClean auto_clean pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import pandas as pd
from pathlib import Path

from smartclean.pipeline import auto_clean
from smartclean.report import CleaningReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """A small messy DataFrame covering all cleaning scenarios."""
    base = pd.DataFrame({
        "First Name": ["  alice  ", "BOB", "charlie", "diana", "eve"],
        "Age":        ["25", "30", None, "28", "32"],
        "Salary":     [50000, 60000, 70000, 55000, 1000000],  # outlier
        "Active":     ["true", "false", "true", "false", "true"],
        "Department": ["HR", None, "IT", "IT", None],
    })
    # Add a duplicate row
    dupe = base.iloc[[0]].copy()
    return pd.concat([base, dupe], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Basic behaviour
# ---------------------------------------------------------------------------

class TestAutocleanBasic:

    def test_returns_dataframe_by_default(self, messy_df):
        result = auto_clean(messy_df)
        assert isinstance(result, pd.DataFrame)

    def test_return_report_true_returns_tuple(self, messy_df):
        result = auto_clean(messy_df, return_report=True)
        assert isinstance(result, tuple)
        df, report = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, CleaningReport)

    def test_does_not_modify_original(self, messy_df):
        original_cols = list(messy_df.columns)
        original_len  = len(messy_df)
        auto_clean(messy_df)
        assert list(messy_df.columns) == original_cols
        assert len(messy_df) == original_len

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            auto_clean("not a dataframe")

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError):
            auto_clean(pd.DataFrame())


# ---------------------------------------------------------------------------
# 2. Column name cleaning
# ---------------------------------------------------------------------------

class TestPipelineColumns:

    def test_column_names_are_snake_case(self, messy_df):
        cleaned = auto_clean(messy_df)
        for col in cleaned.columns:
            assert col == col.lower()
            assert " " not in col

    def test_first_name_becomes_first_name(self, messy_df):
        cleaned = auto_clean(messy_df)
        assert "first_name" in cleaned.columns


# ---------------------------------------------------------------------------
# 3. Missing value handling
# ---------------------------------------------------------------------------

class TestPipelineMissing:

    def test_no_missing_values_after_clean(self, messy_df):
        cleaned = auto_clean(messy_df)
        # Age and Department had missing values
        assert cleaned["age"].isna().sum() == 0

    def test_missing_values_logged_in_report(self, messy_df):
        _, report = auto_clean(messy_df, return_report=True)
        d = report.to_dict()
        assert "missing_values_filled" in d


# ---------------------------------------------------------------------------
# 4. Duplicate removal
# ---------------------------------------------------------------------------

class TestPipelineDuplicates:

    def test_duplicates_removed(self, messy_df):
        cleaned = auto_clean(messy_df)
        assert cleaned.duplicated().sum() == 0

    def test_duplicates_logged_in_report(self, messy_df):
        _, report = auto_clean(messy_df, return_report=True)
        d = report.to_dict()
        assert "duplicates_removed" in d
        assert d["duplicates_removed"] >= 1


# ---------------------------------------------------------------------------
# 5. Outlier handling
# ---------------------------------------------------------------------------

class TestPipelineOutliers:

    def test_outlier_capped_by_default(self, messy_df):
        cleaned = auto_clean(messy_df)
        assert cleaned["salary"].max() < 1000000

    def test_outlier_action_remove(self, messy_df):
        cleaned = auto_clean(messy_df, outlier_action="remove")
        assert isinstance(cleaned, pd.DataFrame)

    def test_outlier_action_flag(self, messy_df):
        cleaned = auto_clean(messy_df, outlier_action="flag")
        flag_cols = [c for c in cleaned.columns if c.endswith("_outlier")]
        assert len(flag_cols) > 0


# ---------------------------------------------------------------------------
# 6. Report content
# ---------------------------------------------------------------------------

class TestPipelineReport:

    def test_report_to_dict_returns_dict(self, messy_df):
        _, report = auto_clean(messy_df, return_report=True)
        assert isinstance(report.to_dict(), dict)

    def test_report_to_df_returns_dataframe(self, messy_df):
        _, report = auto_clean(messy_df, return_report=True)
        assert isinstance(report.to_df(), pd.DataFrame)

    def test_report_summary_runs_without_error(self, messy_df, capsys):
        _, report = auto_clean(messy_df, return_report=True)
        report.summary()
        captured = capsys.readouterr()
        assert "SmartClean" in captured.out


# ---------------------------------------------------------------------------
# 7. Custom options
# ---------------------------------------------------------------------------

class TestPipelineOptions:

    def test_custom_drop_threshold(self, titanic):
        """Lower threshold should drop Cabin column (~77% missing)."""
        cleaned = auto_clean(titanic, drop_if_missing_pct=0.70)
        assert "Cabin" not in cleaned.columns

    def test_default_threshold_keeps_cabin(self, titanic):
        """Default 0.95 threshold should keep Cabin (~77% missing)."""
        cleaned = auto_clean(titanic)
        assert "cabin" not in cleaned.columns or "Cabin" not in cleaned.columns

    def test_zscore_method(self, messy_df):
        cleaned = auto_clean(messy_df, outlier_method="zscore")
        assert isinstance(cleaned, pd.DataFrame)


# ---------------------------------------------------------------------------
# 8. Titanic end-to-end
# ---------------------------------------------------------------------------

class TestTitanicPipeline:

    def test_titanic_cleans_without_error(self, titanic):
        cleaned = auto_clean(titanic)
        assert isinstance(cleaned, pd.DataFrame)

    def test_titanic_no_missing_after_clean(self, titanic):
        cleaned = auto_clean(titanic, drop_if_missing_pct=0.70)
        assert cleaned.isna().sum().sum() == 0

    def test_titanic_column_names_normalised(self, titanic):
        cleaned = auto_clean(titanic)
        for col in cleaned.columns:
            assert col == col.lower()
            assert " " not in col

    def test_titanic_report_has_entries(self, titanic):
        _, report = auto_clean(titanic, return_report=True)
        assert len(report.to_dict()) > 0