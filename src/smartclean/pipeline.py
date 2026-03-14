"""
smartclean/pipeline.py

Auto-clean pipeline — the core one-command cleaning feature.

Pipeline steps (correct order):
    1. Profile dataset
    2. Drop columns exceeding missing threshold (drop only, no fill yet)
    3. Clean column names
    4. Fix data types  <- MUST run before handle_missing
    5. Handle missing values (now safe — correct dtypes)
    6. Remove duplicates
    7. Clean text fields
    8. Detect and handle outliers
    9. Generate CleaningReport
"""

from __future__ import annotations

import pandas as pd
from typing import Literal, Tuple, Union

from smartclean.profiler import profile
from smartclean.report import CleaningReport
from smartclean.modules.columns import clean_column_names
from smartclean.modules.missing import handle_missing
from smartclean.modules.duplicates import remove_duplicates
from smartclean.modules.types import fix_types
from smartclean.modules.text import clean_text
from smartclean.modules.outliers import remove_outliers


def auto_clean(
    df: pd.DataFrame,
    drop_if_missing_pct: float = 0.95,
    outlier_method: Literal["iqr", "zscore"] = "iqr",
    outlier_action: Literal["remove", "cap", "flag"] = "cap",
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, CleaningReport]]:
    """
    Automatically clean a DataFrame in one function call.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean. The original is never modified.
    drop_if_missing_pct : float, optional
        Columns with missing values above this threshold are dropped.
        Default is 0.95. Set to 1.0 to disable.
    outlier_method : {"iqr", "zscore"}, optional
        Outlier detection method. Default is "iqr".
    outlier_action : {"remove", "cap", "flag"}, optional
        How to handle detected outliers. Default is "cap".
    return_report : bool, optional
        If True, returns (cleaned_df, report). Default is False.

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, CleaningReport)

    Examples
    --------
    >>> clean_df = sc.auto_clean(df)
    >>> clean_df, report = sc.auto_clean(df, return_report=True)
    >>> report.summary()
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    df = df.copy()
    report = CleaningReport()

    # Step 1: Profile
    current_profile = profile(df)

    # Step 2: Drop columns exceeding missing threshold (drop only, no fill yet)
    cols_to_drop = [
        col for col, cp in current_profile.columns.items()
        if cp.missing_pct > drop_if_missing_pct and col in df.columns
    ]
    if cols_to_drop:
        dropped_log = {
            col: {
                "reason": "missing_pct",
                "value": round(current_profile.columns[col].missing_pct, 4),
            }
            for col in cols_to_drop
        }
        df = df.drop(columns=cols_to_drop)
        report.log("columns_dropped", dropped_log)
        current_profile = profile(df)

    # Step 3: Clean column names
    old_columns = list(df.columns)
    df = clean_column_names(df)
    new_columns = list(df.columns)
    renamed = {old: new for old, new in zip(old_columns, new_columns) if old != new}
    if renamed:
        report.log("columns_renamed", renamed)
    current_profile = profile(df)

    # Step 4: Fix data types — MUST run before handle_missing
    # so median/mean operate on proper numeric dtypes, not strings
    conversions, df = fix_types(df, current_profile)
    if conversions:
        report.log("types_converted", conversions)
    current_profile = profile(df)

    # Step 5: Handle missing values — safe now, types are corrected
    missing_result, df = handle_missing(
        df,
        profile=current_profile,
        strategy="auto",
        drop_threshold=1.0,  # dropping already handled in Step 2
    )
    if missing_result.get("filled"):
        report.log("missing_values_filled", missing_result["filled"])

    # Step 6: Remove duplicates
    before = len(df)
    df = remove_duplicates(df)
    removed = before - len(df)
    if removed > 0:
        report.log("duplicates_removed", removed)

    # Step 7: Clean text fields
    current_profile = profile(df)
    cleaned_cols, df = clean_text(df, profile=current_profile)
    if cleaned_cols:
        report.log("text_cleaned", cleaned_cols)

    # Step 8: Detect and handle outliers
    current_profile = profile(df)
    outlier_result, df = remove_outliers(
        df,
        profile=current_profile,
        method=outlier_method,
        action=outlier_action,
    )
    if outlier_result:
        report.log("outliers_handled", outlier_result)

    if return_report:
        return df, report

    return df