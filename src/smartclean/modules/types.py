"""
smartclean/modules/types.py

Data type correction module.

Responsible for detecting and converting columns stored as incorrect
types — typically strings that should be numeric, datetime, or boolean.

Functions:
    fix_types()          — main function, uses profiler output
    convert_numeric()    — convert string columns to numeric
    convert_datetime()   — convert string columns to datetime
    convert_boolean()    — convert string columns to boolean
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from smartclean.profiler import ProfileResult


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fix_types(
    df: pd.DataFrame,
    profile: ProfileResult,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Detect and correct incorrect data types across all columns.

    Uses the profiler's semantic dtype to decide which conversions
    to attempt. Only converts columns where the current pandas dtype
    does not match the detected semantic type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    profile : ProfileResult
        The profiler output for this DataFrame.

    Returns
    -------
    tuple of (conversions_dict, cleaned_df)
        conversions_dict : {column_name: new_dtype_string}
        cleaned_df       : DataFrame with corrected types.
    """
    df = df.copy()
    conversions: Dict[str, str] = {}

    for col, col_profile in profile.columns.items():
        if col not in df.columns:
            continue

        current_dtype = df[col].dtype
        semantic_dtype = col_profile.dtype

        # Numeric: object column that should be numeric
        if semantic_dtype == "numeric" and current_dtype == object:
            converted = _try_convert_numeric(df[col])
            if converted is not None:
                df[col] = converted
                conversions[col] = str(df[col].dtype)

        # Datetime: object column that should be datetime
        elif semantic_dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(current_dtype):
            converted = _try_convert_datetime(df[col])
            if converted is not None:
                df[col] = converted
                conversions[col] = str(df[col].dtype)

        # Boolean: object column that should be boolean
        elif semantic_dtype == "categorical" and current_dtype == object:
            converted = _try_convert_boolean(df[col])
            if converted is not None:
                df[col] = converted
                conversions[col] = str(df[col].dtype)

    return conversions, df


def convert_numeric(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert specified columns (or all object columns) to numeric types.

    Non-convertible values are coerced to NaN rather than raising errors.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to convert. If None, attempts all object columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    for col in target_cols:
        if col not in df.columns:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only apply if at least some values converted successfully
        if converted.notna().sum() > 0:
            df[col] = converted

    return df


def convert_datetime(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert specified columns (or all object columns) to datetime types.

    Non-convertible values are coerced to NaT rather than raising errors.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to convert. If None, attempts all object columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    for col in target_cols:
        if col not in df.columns:
            continue
        converted = pd.to_datetime(df[col], errors="coerce")
        # Only apply if at least some values converted successfully
        if converted.notna().sum() > 0:
            non_null_original = df[col].dropna()
            success_rate = converted.notna().sum() / max(len(non_null_original), 1)
            if success_rate >= 0.8:
                df[col] = converted

    return df


def convert_boolean(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert columns containing boolean-like string values to boolean type.

    Recognises: "true"/"false", "yes"/"no", "1"/"0", "t"/"f", "y"/"n"
    (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to convert. If None, attempts all object columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    for col in target_cols:
        if col not in df.columns:
            continue
        converted = _try_convert_boolean(df[col])
        if converted is not None:
            df[col] = converted

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Known boolean-like string mappings
_BOOL_TRUE  = {"true", "yes", "1", "t", "y", "on"}
_BOOL_FALSE = {"false", "no", "0", "f", "n", "off"}
_BOOL_ALL   = _BOOL_TRUE | _BOOL_FALSE


def _try_convert_numeric(series: pd.Series) -> Optional[pd.Series]:
    """
    Attempt to convert a Series to numeric.

    Returns the converted Series if successful (>80% of non-null
    values convert), otherwise returns None.
    """
    converted = pd.to_numeric(series.str.strip() if hasattr(series, "str") else series, errors="coerce")
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    success_rate = converted.notna().sum() / len(non_null)
    if success_rate >= 0.8:
        return converted
    return None


def _try_convert_datetime(series: pd.Series) -> Optional[pd.Series]:
    """
    Attempt to convert a Series to datetime.

    Returns the converted Series if successful (>80% of non-null
    values convert), otherwise returns None.
    """
    try:
        converted = pd.to_datetime(series, errors="coerce")
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        success_rate = converted.notna().sum() / len(non_null)
        if success_rate >= 0.8:
            return converted
        return None
    except Exception:
        return None


def _try_convert_boolean(series: pd.Series) -> Optional[pd.Series]:
    """
    Attempt to convert a Series to boolean.

    Only converts if all non-null unique values are boolean-like strings.
    Returns the converted Series or None if not boolean-like.
    """
    if not hasattr(series, "str"):
        return None

    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    unique_vals = set(non_null.astype(str).str.strip().str.lower().unique())

    # Only convert if ALL unique values are boolean-like
    if not unique_vals.issubset(_BOOL_ALL):
        return None

    # Must have both true and false values present (not just all-true)
    has_true  = bool(unique_vals & _BOOL_TRUE)
    has_false = bool(unique_vals & _BOOL_FALSE)
    if not (has_true and has_false):
        return None

    bool_map = {v: True for v in _BOOL_TRUE}
    bool_map.update({v: False for v in _BOOL_FALSE})

    converted = series.str.strip().str.lower().map(bool_map)
    return converted.astype("boolean")  # pandas nullable boolean