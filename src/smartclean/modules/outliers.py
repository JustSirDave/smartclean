"""
smartclean/modules/outliers.py

Outlier detection and handling module.

Responsible for identifying and handling extreme values in numeric columns.

Supported detection methods:
    IQR    — Interquartile Range (default, no extra dependencies)
    zscore — Z-score (requires scipy)

Handling options:
    remove — drop rows containing outliers
    cap    — clip values to the boundary (default, non-destructive)
    flag   — add a boolean column marking outlier rows

Functions:
    remove_outliers()    — main function, uses profiler output
    detect_outliers_iqr()   — IQR-based detection
    detect_outliers_zscore() — Z-score-based detection
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple

from smartclean.profiler import ProfileResult


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def remove_outliers(
    df: pd.DataFrame,
    profile: ProfileResult,
    method: Literal["iqr", "zscore"] = "iqr",
    action: Literal["remove", "cap", "flag"] = "cap",
    columns: Optional[List[str]] = None,
    threshold: float = 1.5,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Detect and handle outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    profile : ProfileResult
        The profiler output for this DataFrame.
    method : {"iqr", "zscore"}, optional
        Detection method. Default is "iqr".
    action : {"remove", "cap", "flag"}, optional
        How to handle detected outliers. Default is "cap".
    columns : list of str, optional
        Specific numeric columns to check. If None, uses all
        numeric columns detected by the profiler.
    threshold : float, optional
        IQR multiplier (default 1.5) or Z-score cutoff (default 3.0
        when method="zscore"). Pass your own value to override.

    Returns
    -------
    tuple of (result_dict, cleaned_df)
        result_dict : {col: {count, method, action}} for affected columns
        cleaned_df  : DataFrame with outliers handled.
    """
    df = df.copy()
    result: Dict = {}

    # Resolve columns to check
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        target_cols = columns
    else:
        target_cols = [
            col for col, cp in profile.columns.items()
            if cp.dtype == "numeric" and col in df.columns
        ]

    # Use z-score default threshold of 3.0 if method is zscore
    # and user hasn't overridden from the default 1.5
    effective_threshold = threshold
    if method == "zscore" and threshold == 1.5:
        effective_threshold = 3.0

    for col in target_cols:
        if col not in df.columns:
            continue

        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.isna().all():
            continue

        # Detect outlier mask
        if method == "iqr":
            mask = detect_outliers_iqr(numeric_col, threshold=effective_threshold)
        elif method == "zscore":
            mask = detect_outliers_zscore(numeric_col, threshold=effective_threshold)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: 'iqr', 'zscore'."
            )

        outlier_count = int(mask.sum())
        if outlier_count == 0:
            continue

        # Apply action
        if action == "remove":
            df = df[~mask].reset_index(drop=True)

        elif action == "cap":
            if method == "iqr":
                q1 = numeric_col.quantile(0.25)
                q3 = numeric_col.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - effective_threshold * iqr
                upper = q3 + effective_threshold * iqr
            else:
                mean = numeric_col.mean()
                std  = numeric_col.std()
                lower = mean - effective_threshold * std
                upper = mean + effective_threshold * std
            df[col] = numeric_col.clip(lower=lower, upper=upper)

        elif action == "flag":
            flag_col = f"{col}_outlier"
            df[flag_col] = mask

        else:
            raise ValueError(
                f"Unknown action '{action}'. Choose from: 'remove', 'cap', 'flag'."
            )

        result[col] = {
            "count":  outlier_count,
            "method": method,
            "action": action,
        }

    return result, df


def detect_outliers_iqr(
    series: pd.Series,
    threshold: float = 1.5,
) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    An outlier is defined as any value below Q1 - threshold*IQR
    or above Q3 + threshold*IQR.

    Parameters
    ----------
    series : pd.Series
        A numeric Series to check.
    threshold : float, optional
        The IQR multiplier. Default is 1.5 (standard).
        Use 3.0 for extreme outlier detection only.

    Returns
    -------
    pd.Series
        Boolean Series — True where the value is an outlier.
    """
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    q1  = numeric.quantile(0.25)
    q3  = numeric.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    return (numeric < lower) | (numeric > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect outliers using the Z-score method.

    An outlier is defined as any value with |z-score| > threshold.

    Parameters
    ----------
    series : pd.Series
        A numeric Series to check.
    threshold : float, optional
        The Z-score cutoff. Default is 3.0.

    Returns
    -------
    pd.Series
        Boolean Series — True where the value is an outlier.

    Notes
    -----
    This implementation uses numpy directly and does not require scipy.
    scipy is only needed for more advanced statistical tests.
    """
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = numeric.mean()
    std  = numeric.std()

    if std == 0:
        # All values are identical — no outliers possible
        return pd.Series(False, index=series.index)

    z_scores = (numeric - mean) / std
    return z_scores.abs() > threshold