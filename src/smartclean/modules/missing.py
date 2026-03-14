"""
smartclean/modules/missing.py

Missing value handling module.

Responsible for detecting and filling missing values in a DataFrame.
Supports both automatic strategy selection (based on column semantic type)
and manual strategy specification.

Auto strategy defaults (from gap resolutions):
    numeric     → median
    categorical → mode  (fallback: "unknown" if no clear mode)
    datetime    → forward fill (fallback: backward fill)
    text        → constant "unknown"

Threshold-based column dropping:
    Columns with missing_pct > drop_threshold are dropped before
    any filling is attempted and logged in the cleaning report.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple, Any

from smartclean.profiler import ProfileResult


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def handle_missing(
    df: pd.DataFrame,
    profile: ProfileResult,
    strategy: Literal["auto", "mean", "median", "mode"] = "auto",
    columns: Optional[List[str]] = None,
    drop_threshold: float = 0.95,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Detect and fill missing values in a DataFrame.

    Columns exceeding `drop_threshold` are dropped first and recorded
    in the result. Remaining missing values are filled using the
    specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    profile : ProfileResult
        The profiler output for this DataFrame.
    strategy : str, optional
        One of "auto", "mean", "median", "mode". Default is "auto".
        "auto" selects the best strategy per column based on dtype.
    columns : list of str, optional
        Specific columns to process. If None, processes all columns
        that have at least one missing value.
    drop_threshold : float, optional
        Columns with missing_pct above this value are dropped.
        Default is 0.95. Set to 1.0 to disable dropping.

    Returns
    -------
    tuple of (result_dict, cleaned_df)
        result_dict contains:
            "dropped" : dict {col: {reason, value}} or empty dict
            "filled"  : dict {col: {count, strategy}} or empty dict
        cleaned_df is the processed DataFrame.
    """
    df = df.copy()
    result: Dict[str, Any] = {"dropped": {}, "filled": {}}

    # ── Step 1: Drop columns exceeding the missing threshold ────────────────
    cols_to_check = _resolve_columns(df, columns, profile)

    for col in list(cols_to_check):
        col_profile = profile.columns.get(col)
        if col_profile is None:
            continue
        if col_profile.missing_pct > drop_threshold:
            df = df.drop(columns=[col])
            result["dropped"][col] = {
                "reason": "missing_pct",
                "value": round(col_profile.missing_pct, 4),
            }
            cols_to_check = [c for c in cols_to_check if c != col]

    # ── Step 2: Fill remaining missing values ───────────────────────────────
    for col in cols_to_check:
        if col not in df.columns:
            continue

        col_profile = profile.columns.get(col)
        if col_profile is None:
            continue

        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        # Determine which strategy to apply
        if strategy == "auto":
            applied_strategy = _auto_strategy(col_profile.dtype)
        else:
            applied_strategy = strategy

        # Apply the strategy
        df[col] = _apply_strategy(df[col], applied_strategy, col_profile.dtype)

        # Record how many were filled and which strategy was used
        filled_count = missing_count - df[col].isna().sum()
        if filled_count > 0:
            result["filled"][col] = {
                "count": int(filled_count),
                "strategy": applied_strategy,
            }

    return result, df


# ---------------------------------------------------------------------------
# Individual fill functions (also callable directly per SRS spec)
# ---------------------------------------------------------------------------

def fill_mean(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column mean.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to fill. If None, applies to all numeric columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in target_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def fill_median(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with the column median.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to fill. If None, applies to all numeric columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="number").columns.tolist()
    for col in target_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def fill_mode(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values with the most frequent value (mode).

    If no clear mode exists, falls back to the constant "unknown"
    for object columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to fill. If None, applies to all object/categorical columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in target_cols:
        if col in df.columns and df[col].isna().any():
            mode_series = df[col].mode()
            if len(mode_series) > 0:
                df[col] = df[col].fillna(mode_series[0])
            else:
                df[col] = df[col].fillna("unknown")
    return df


def fill_auto(df: pd.DataFrame, profile: ProfileResult) -> pd.DataFrame:
    """
    Automatically fill missing values using the best strategy
    per column semantic type.

    Parameters
    ----------
    df : pd.DataFrame
    profile : ProfileResult
        Profiler output for this DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    _, cleaned_df = handle_missing(df, profile, strategy="auto")
    return cleaned_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    profile: ProfileResult,
) -> List[str]:
    """
    Return the list of columns to process.

    If columns is specified, validate they exist in the DataFrame.
    Otherwise return all columns that have at least one missing value.
    """
    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {missing_cols}"
            )
        return list(columns)

    # Default: all columns with at least one missing value
    return [
        col for col in df.columns
        if df[col].isna().any()
    ]


def _auto_strategy(dtype: str) -> str:
    """
    Return the recommended fill strategy for a given semantic dtype.

    Parameters
    ----------
    dtype : str
        One of "numeric", "categorical", "datetime", "text".

    Returns
    -------
    str
        The strategy name.
    """
    mapping = {
        "numeric":     "median",
        "categorical": "mode",
        "datetime":    "ffill",
        "text":        "constant",
    }
    return mapping.get(dtype, "mode")


def _apply_strategy(
    series: pd.Series,
    strategy: str,
    dtype: str,
) -> pd.Series:
    """
    Apply a fill strategy to a single pandas Series.

    Parameters
    ----------
    series : pd.Series
    strategy : str
        One of "median", "mean", "mode", "ffill", "bfill", "constant".
    dtype : str
        The semantic dtype of the column — used for fallback decisions.

    Returns
    -------
    pd.Series
    """
    if strategy == "median":
        fill_value = series.median()
        return series.fillna(fill_value)

    elif strategy == "mean":
        fill_value = series.mean()
        return series.fillna(fill_value)

    elif strategy == "mode":
        mode_series = series.mode()
        if len(mode_series) > 0:
            return series.fillna(mode_series[0])
        # Fallback if no mode (all values unique)
        return series.fillna("unknown")

    elif strategy == "ffill":
        filled = series.ffill()
        # If forward fill leaves NaNs at the start, use backward fill
        if filled.isna().any():
            filled = filled.bfill()
        return filled

    elif strategy == "bfill":
        filled = series.bfill()
        if filled.isna().any():
            filled = filled.ffill()
        return filled

    elif strategy == "constant":
        return series.fillna("unknown")

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Valid options: 'auto', 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'."
        )