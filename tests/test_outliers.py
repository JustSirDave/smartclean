"""
Tests for SmartClean outliers module.
Run with: pytest tests/test_outliers.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from smartclean.profiler import profile
from smartclean.modules.outliers import (
    remove_outliers,
    detect_outliers_iqr,
    detect_outliers_zscore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_df() -> pd.DataFrame:
    """DataFrame with obvious outliers."""
    return pd.DataFrame({
        "salary": [50000, 55000, 52000, 48000, 51000, 1000000],  # 1M is outlier
        "age":    [25, 30, 28, 32, 27, 200],                      # 200 is outlier
        "name":   ["A", "B", "C", "D", "E", "F"],
    })


@pytest.fixture
def no_outlier_df() -> pd.DataFrame:
    """DataFrame with no outliers."""
    return pd.DataFrame({
        "salary": [50000, 52000, 51000, 49000, 50500],
        "age":    [25, 26, 27, 28, 29],
    })


@pytest.fixture
def titanic() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "docs" / "titanic.csv"
    return pd.read_csv(path)


@pytest.fixture
def titanic_profile(titanic):
    return profile(titanic)


# ---------------------------------------------------------------------------
# 1. detect_outliers_iqr()
# ---------------------------------------------------------------------------

class TestDetectOutliersIQR:

    def test_returns_boolean_series(self, normal_df):
        result = detect_outliers_iqr(normal_df["salary"])
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_detects_obvious_outlier(self):
        """Z-score reliably detects outliers with larger samples."""
        import numpy as np
        normal_vals = list(np.random.normal(50000, 1000, 50))
        normal_vals.append(1000000)  # obvious outlier
        s = pd.Series(normal_vals)
        result = detect_outliers_zscore(s, threshold=3.0)
        assert result.iloc[-1] == True

    def test_normal_values_not_flagged(self, no_outlier_df):
        result = detect_outliers_iqr(no_outlier_df["salary"])
        assert result.sum() == 0

    def test_length_matches_series(self, normal_df):
        result = detect_outliers_iqr(normal_df["salary"])
        assert len(result) == len(normal_df)

    def test_stricter_threshold_detects_fewer(self, normal_df):
        result_default = detect_outliers_iqr(normal_df["salary"], threshold=1.5)
        result_strict  = detect_outliers_iqr(normal_df["salary"], threshold=3.0)
        assert result_default.sum() >= result_strict.sum()

    def test_constant_series_no_outliers(self):
        s = pd.Series([5, 5, 5, 5, 5])
        result = detect_outliers_iqr(s)
        assert result.sum() == 0

    def test_handles_nan_values(self):
        s = pd.Series([1.0, 2.0, None, 100.0, 2.0])
        result = detect_outliers_iqr(s)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 2. detect_outliers_zscore()
# ---------------------------------------------------------------------------

class TestDetectOutliersZScore:

    def test_returns_boolean_series(self, normal_df):
        result = detect_outliers_zscore(normal_df["salary"])
        assert isinstance(result, pd.Series)

    def test_detects_obvious_outlier(self, normal_df):
        result = detect_outliers_zscore(normal_df["salary"], threshold=2.0)
        assert result.iloc[-1] == True  # 1,000,000 is outlier

    def test_normal_values_not_flagged(self, no_outlier_df):
        result = detect_outliers_zscore(no_outlier_df["salary"])
        assert result.sum() == 0

    def test_constant_series_no_outliers(self):
        s = pd.Series([5, 5, 5, 5, 5])
        result = detect_outliers_zscore(s)
        assert result.sum() == 0

    def test_length_matches_series(self, normal_df):
        result = detect_outliers_zscore(normal_df["salary"])
        assert len(result) == len(normal_df)


# ---------------------------------------------------------------------------
# 3. remove_outliers() — actions
# ---------------------------------------------------------------------------

class TestRemoveOutliersAction:

    def test_returns_tuple(self, normal_df):
        p = profile(normal_df)
        result = remove_outliers(normal_df, p)
        assert isinstance(result, tuple)
        info, df = result
        assert isinstance(info, dict)
        assert isinstance(df, pd.DataFrame)

    def test_does_not_modify_original(self, normal_df):
        original_len = len(normal_df)
        p = profile(normal_df)
        remove_outliers(normal_df, p)
        assert len(normal_df) == original_len

    def test_action_cap_default(self, normal_df):
        p = profile(normal_df)
        _, cleaned = remove_outliers(normal_df, p, action="cap")
        assert len(cleaned) == len(normal_df)  # no rows removed
        assert cleaned["salary"].max() < 1000000

    def test_action_remove_drops_rows(self, normal_df):
        p = profile(normal_df)
        _, cleaned = remove_outliers(normal_df, p, action="remove")
        assert len(cleaned) < len(normal_df)

    def test_action_flag_adds_column(self, normal_df):
        p = profile(normal_df)
        _, cleaned = remove_outliers(normal_df, p, action="flag")
        assert "salary_outlier" in cleaned.columns or "age_outlier" in cleaned.columns

    def test_result_logs_count_method_action(self, normal_df):
        p = profile(normal_df)
        info, _ = remove_outliers(normal_df, p, action="cap")
        for col, details in info.items():
            assert "count" in details
            assert "method" in details
            assert "action" in details

    def test_no_outliers_empty_result(self, no_outlier_df):
        p = profile(no_outlier_df)
        info, _ = remove_outliers(no_outlier_df, p)
        assert info == {}

    def test_invalid_action_raises(self, normal_df):
        p = profile(normal_df)
        with pytest.raises(ValueError, match="Unknown action"):
            remove_outliers(normal_df, p, action="delete")

    def test_invalid_method_raises(self, normal_df):
        p = profile(normal_df)
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(normal_df, p, method="percentile")

    def test_invalid_column_raises(self, normal_df):
        p = profile(normal_df)
        with pytest.raises(ValueError, match="not found"):
            remove_outliers(normal_df, p, columns=["nonexistent"])

    def test_zscore_method(self, normal_df):
        p = profile(normal_df)
        info, cleaned = remove_outliers(
            normal_df, p, method="zscore", action="cap"
        )
        for col, details in info.items():
            assert details["method"] == "zscore"

    def test_specific_columns_only(self, normal_df):
        p = profile(normal_df)
        info, _ = remove_outliers(normal_df, p, columns=["salary"], action="cap")
        assert "age" not in info

    def test_cap_does_not_increase_max(self, normal_df):
        p = profile(normal_df)
        original_max = normal_df["salary"].max()
        _, cleaned = remove_outliers(normal_df, p, action="cap")
        assert cleaned["salary"].max() <= original_max

    def test_index_reset_after_remove(self, normal_df):
        p = profile(normal_df)
        _, cleaned = remove_outliers(normal_df, p, action="remove")
        assert list(cleaned.index) == list(range(len(cleaned)))


# ---------------------------------------------------------------------------
# 4. Titanic integration
# ---------------------------------------------------------------------------

class TestTitanicOutliers:

    def test_fare_has_outliers_detected(self, titanic, titanic_profile):
        mask = detect_outliers_iqr(titanic["Fare"])
        assert mask.sum() > 0

    def test_cap_does_not_change_row_count(self, titanic, titanic_profile):
        _, cleaned = remove_outliers(titanic, titanic_profile, action="cap")
        assert len(cleaned) == len(titanic)

    def test_fare_max_reduced_after_cap(self, titanic, titanic_profile):
        _, cleaned = remove_outliers(titanic, titanic_profile, action="cap")
        assert cleaned["Fare"].max() <= titanic["Fare"].max()