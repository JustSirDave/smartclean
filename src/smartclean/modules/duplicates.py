"""
smartclean/modules/duplicates.py

Duplicate detection and removal module.

Responsible for identifying and removing duplicate rows in a DataFrame.
Supports full-row duplicate detection and subset-based detection
(duplicates based on specific columns only).

Functions:
    detect_duplicates()  — returns a boolean mask of duplicate rows
    remove_duplicates()  — returns a cleaned DataFrame with duplicates removed
"""

from __future__ import annotations

import pandas as pd
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last", False] = "first",
) -> pd.Series:
    """
    Return a boolean Series marking duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    subset : list of str, optional
        Column labels to consider for identifying duplicates.
        If None, all columns are used.
    keep : {"first", "last", False}, optional
        Which occurrence to mark as NOT a duplicate:
        - "first" : mark all duplicates except the first occurrence
        - "last"  : mark all duplicates except the last occurrence
        - False   : mark all occurrences of duplicates
        Default is "first".

    Returns
    -------
    pd.Series
        Boolean Series — True where the row is a duplicate.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If any column in subset does not exist in df.

    Examples
    --------
    >>> mask = detect_duplicates(df)
    >>> print(f"Found {mask.sum()} duplicate rows")

    >>> # Check duplicates based on specific columns only
    >>> mask = detect_duplicates(df, subset=["name", "email"])
    """
    _validate_inputs(df, subset)
    return df.duplicated(subset=subset, keep=keep)


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Literal["first", "last", False] = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    This function does not modify the original DataFrame — it returns
    a new DataFrame with duplicates removed.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    subset : list of str, optional
        Column labels to consider for identifying duplicates.
        If None, all columns are used.
    keep : {"first", "last", False}, optional
        Which occurrence to keep:
        - "first" : keep the first occurrence, drop the rest (default)
        - "last"  : keep the last occurrence, drop the rest
        - False   : drop all occurrences of duplicated rows
        Default is "first".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with duplicate rows removed and the index reset.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If any column in subset does not exist in df.

    Examples
    --------
    >>> cleaned = remove_duplicates(df)

    >>> # Remove duplicates based on specific columns
    >>> cleaned = remove_duplicates(df, subset=["name", "email"])

    >>> # Drop ALL occurrences of duplicated rows (keep none)
    >>> cleaned = remove_duplicates(df, keep=False)
    """
    _validate_inputs(df, subset)

    cleaned = df.drop_duplicates(subset=subset, keep=keep)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def count_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> int:
    """
    Return the number of duplicate rows in a DataFrame.

    Uses keep="first" — counts all rows beyond the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list of str, optional
        Columns to consider. If None, uses all columns.

    Returns
    -------
    int
        Number of duplicate rows.

    Examples
    --------
    >>> n = count_duplicates(df)
    >>> print(f"{n} duplicate rows found")
    """
    _validate_inputs(df, subset)
    return int(df.duplicated(subset=subset, keep="first").sum())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    df: pd.DataFrame,
    subset: Optional[List[str]],
) -> None:
    """Validate DataFrame type and subset columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if subset is not None:
        missing = [col for col in subset if col not in df.columns]
        if missing:
            raise ValueError(
                f"Columns not found in DataFrame: {missing}"
            )