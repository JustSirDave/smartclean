"""
smartclean — Intelligent data cleaning for pandas DataFrames.

Quick start:
    import smartclean as sc

    # One-liner automatic cleaning
    clean_df = sc.auto_clean(df)

    # With cleaning report
    clean_df, report = sc.auto_clean(df, return_report=True)
    report.summary()

    # Manual control via chained API
    cleaner = sc.Cleaner(df)
    clean_df = (
        cleaner
        .clean_columns()
        .fix_types()
        .handle_missing()
        .remove_duplicates()
        .clean_text()
        .remove_outliers()
        .output()
    )

    # Load data
    df = sc.read("data.csv")
    df = sc.read("data.xlsx")
    df = sc.read("data.json")

    # Profile only
    profile = sc.profile(df)
    print(profile)
"""

from smartclean.pipeline import auto_clean
from smartclean.cleaner import Cleaner
from smartclean.profiler import profile
from smartclean.report import CleaningReport
from smartclean.io import read

from smartclean.modules.missing import (
    fill_mean,
    fill_median,
    fill_mode,
    fill_auto,
)
from smartclean.modules.duplicates import (
    detect_duplicates,
    remove_duplicates,
    count_duplicates,
)
from smartclean.modules.columns import (
    clean_column_names,
    snake_case_columns,
)
from smartclean.modules.types import (
    convert_numeric,
    convert_datetime,
    convert_boolean,
)
from smartclean.modules.text import (
    strip_whitespace,
    normalize_case,
    remove_special_chars,
)
from smartclean.modules.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
)

__version__ = "0.1.0"
__author__  = "David Ekundayo Lucas"
__all__ = [
    # Core
    "auto_clean",
    "Cleaner",
    "profile",
    "CleaningReport",
    "read",
    # Missing
    "fill_mean",
    "fill_median",
    "fill_mode",
    "fill_auto",
    # Duplicates
    "detect_duplicates",
    "remove_duplicates",
    "count_duplicates",
    # Columns
    "clean_column_names",
    "snake_case_columns",
    # Types
    "convert_numeric",
    "convert_datetime",
    "convert_boolean",
    # Text
    "strip_whitespace",
    "normalize_case",
    "remove_special_chars",
    # Outliers
    "detect_outliers_iqr",
    "detect_outliers_zscore",
]