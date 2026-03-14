"""
smartclean/modules/text.py

Text cleaning and normalisation module.

Responsible for cleaning string columns — stripping whitespace,
normalising case, removing special characters, and category normalisation.

Functions:
    clean_text()           — main function, uses profiler output
    strip_whitespace()     — strip leading/trailing whitespace
    normalize_case()       — title-case string values
    remove_special_chars() — remove non-alphanumeric characters
"""

from __future__ import annotations

import re
import pandas as pd
from typing import List, Optional, Tuple

from smartclean.profiler import ProfileResult


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def clean_text(
    df: pd.DataFrame,
    profile: ProfileResult,
    columns: Optional[List[str]] = None,
    normalize_case: bool = True,
    strip_whitespace: bool = True,
    remove_special_chars: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Normalise string columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    profile : ProfileResult
        The profiler output for this DataFrame.
    columns : list of str, optional
        Specific columns to clean. If None, applies to all columns
        with semantic dtype "text" or "categorical".
    normalize_case : bool, optional
        Title-case string values. Default is True.
    strip_whitespace : bool, optional
        Strip leading and trailing whitespace. Default is True.
    remove_special_chars : bool, optional
        Remove non-alphanumeric characters (excluding spaces).
        Default is False — opt-in only.

    Returns
    -------
    tuple of (cleaned_columns_list, cleaned_df)
        cleaned_columns_list : list of column names that were cleaned.
        cleaned_df           : DataFrame with cleaned text columns.
    """
    df = df.copy()
    cleaned_cols: List[str] = []

    # Resolve which columns to clean
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        target_cols = columns
    else:
        # Auto-select text and categorical columns
        target_cols = [
            col for col, cp in profile.columns.items()
            if cp.dtype in ("text", "categorical") and col in df.columns
        ]

    for col in target_cols:
        if df[col].dtype != object:
            continue

        original = df[col].copy()

        if strip_whitespace:
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

        if normalize_case:
            df[col] = df[col].apply(
                lambda x: x.title() if isinstance(x, str) else x
            )

        if remove_special_chars:
            df[col] = df[col].apply(
                lambda x: re.sub(r"[^\w\s]", "", x) if isinstance(x, str) else x
            )

        # Only log the column if something actually changed
        if not df[col].equals(original):
            cleaned_cols.append(col)

    return cleaned_cols, df


def strip_whitespace(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from all string columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to process. If None, applies to all object columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    for col in target_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    return df


def normalize_case(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    case: str = "title",
) -> pd.DataFrame:
    """
    Normalise the case of string columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to process. If None, applies to all object columns.
    case : {"title", "lower", "upper"}, optional
        The case style to apply. Default is "title".

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If case is not one of "title", "lower", "upper".
    """
    valid_cases = {"title", "lower", "upper"}
    if case not in valid_cases:
        raise ValueError(
            f"Invalid case '{case}'. Choose from: {valid_cases}"
        )

    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    case_fn = {
        "title": str.title,
        "lower": str.lower,
        "upper": str.upper,
    }[case]

    for col in target_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: case_fn(x) if isinstance(x, str) else x
            )
    return df


def remove_special_chars(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    keep_spaces: bool = True,
) -> pd.DataFrame:
    """
    Remove non-alphanumeric characters from string columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Columns to process. If None, applies to all object columns.
    keep_spaces : bool, optional
        If True, preserves spaces. Default is True.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include="object").columns.tolist()

    pattern = r"[^\w\s]" if keep_spaces else r"[^\w]"

    for col in target_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: re.sub(pattern, "", x) if isinstance(x, str) else x
            )
    return df