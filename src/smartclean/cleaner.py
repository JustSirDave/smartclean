"""
smartclean/cleaner.py

The Cleaner class — the backbone of the SmartClean manual API.

Usage:
    import smartclean as sc

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
"""

from __future__ import annotations

import pandas as pd
from typing import List, Optional, Literal

from smartclean.profiler import profile, ProfileResult
from smartclean.report import CleaningReport


class Cleaner:
    """
    Manual cleaning interface for SmartClean.

    On initialisation, a copy of the DataFrame is made immediately.
    The caller's original DataFrame is never modified.

    Each method mutates the internal copy and returns `self`,
    enabling method chaining. Call `.output()` to retrieve the
    cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    drop_if_missing_pct : float, optional
        Columns with a missing value percentage above this threshold
        will be dropped automatically during `handle_missing()`.
        Default is 0.95 (95%). Set to 1.0 to disable auto-dropping.

    Example
    -------
    >>> cleaner = Cleaner(df, drop_if_missing_pct=0.90)
    >>> clean_df = (
    ...     cleaner
    ...     .clean_columns()
    ...     .fix_types()
    ...     .handle_missing()
    ...     .remove_duplicates()
    ...     .clean_text()
    ...     .remove_outliers()
    ...     .output()
    ... )
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drop_if_missing_pct: float = 0.95,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            raise ValueError(
                "Cannot clean an empty DataFrame."
            )
        if not (0.0 < drop_if_missing_pct <= 1.0):
            raise ValueError(
                "drop_if_missing_pct must be between 0.0 (exclusive) and 1.0 (inclusive)."
            )

        # ── Core state ──────────────────────────────────────────────────────
        self._df: pd.DataFrame = df.copy()          # protected copy — original never touched
        self._original_shape: tuple = df.shape
        self._drop_if_missing_pct: float = drop_if_missing_pct
        self._profile: Optional[ProfileResult] = None
        self._report: CleaningReport = CleaningReport()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_profile(self) -> ProfileResult:
        """Return cached profile, running it if not yet available."""
        if self._profile is None:
            self._profile = profile(self._df)
        return self._profile

    def _invalidate_profile(self) -> None:
        """Invalidate cached profile after any structural change to the DataFrame."""
        self._profile = None

    # ── Public API ──────────────────────────────────────────────────────────

    def clean_columns(self) -> "Cleaner":
        """
        Normalise all column names to snake_case.

        Transformations applied:
        - Strip leading/trailing whitespace
        - Convert to lowercase
        - Replace spaces, hyphens, and dots with underscores
        - Remove any remaining special characters
        - Collapse multiple consecutive underscores into one

        Returns
        -------
        Cleaner
            Returns self for method chaining.

        Example
        -------
        >>> # "First Name" → "first_name"
        >>> # "TOTAL-SALES" → "total_sales"
        >>> # "customer.id" → "customer_id"
        """
        from smartclean.modules.columns import clean_column_names

        old_columns = list(self._df.columns)
        self._df = clean_column_names(self._df)
        new_columns = list(self._df.columns)

        renamed = {
            old: new
            for old, new in zip(old_columns, new_columns)
            if old != new
        }

        if renamed:
            self._report.log("columns_renamed", renamed)

        self._invalidate_profile()
        return self

    def fix_types(self) -> "Cleaner":
        """
        Detect and correct incorrect data types.

        Attempts to convert:
        - String-encoded numbers → int or float
        - String-encoded dates  → datetime
        - String-encoded bools  → boolean

        Returns
        -------
        Cleaner
            Returns self for method chaining.
        """
        from smartclean.modules.types import fix_types

        conversions, self._df = fix_types(self._df, self._get_profile())

        if conversions:
            self._report.log("types_converted", conversions)

        self._invalidate_profile()
        return self

    def handle_missing(
        self,
        strategy: Literal["auto", "mean", "median", "mode"] = "auto",
        columns: Optional[List[str]] = None,
    ) -> "Cleaner":
        """
        Detect and fill missing values.

        Columns exceeding `drop_if_missing_pct` are dropped first and
        logged in the report. Remaining missing values are filled using
        the specified strategy.

        Auto strategy defaults:
        - numeric     → median
        - categorical → mode  (fallback: "unknown")
        - datetime    → forward fill

        Parameters
        ----------
        strategy : str, optional
            One of "auto", "mean", "median", "mode". Default is "auto".
        columns : list of str, optional
            Specific columns to apply the strategy to.
            If None, applies to all columns with missing values.

        Returns
        -------
        Cleaner
            Returns self for method chaining.
        """
        from smartclean.modules.missing import handle_missing

        result, self._df = handle_missing(
            self._df,
            profile=self._get_profile(),
            strategy=strategy,
            columns=columns,
            drop_threshold=self._drop_if_missing_pct,
        )

        if result.get("dropped"):
            self._report.log("columns_dropped", result["dropped"])
        if result.get("filled"):
            self._report.log("missing_values_filled", result["filled"])

        self._invalidate_profile()
        return self

    def remove_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last", False] = "first",
    ) -> "Cleaner":
        """
        Detect and remove duplicate rows.

        Parameters
        ----------
        subset : list of str, optional
            Column labels to consider for identifying duplicates.
            If None, uses all columns.
        keep : {"first", "last", False}, optional
            Which duplicate to keep. Default is "first".

        Returns
        -------
        Cleaner
            Returns self for method chaining.
        """
        from smartclean.modules.duplicates import remove_duplicates

        before = len(self._df)
        self._df = remove_duplicates(self._df, subset=subset, keep=keep)
        removed = before - len(self._df)

        if removed > 0:
            self._report.log("duplicates_removed", removed)

        return self

    def clean_text(
        self,
        columns: Optional[List[str]] = None,
        normalize_case: bool = True,
        strip_whitespace: bool = True,
        remove_special_chars: bool = False,
    ) -> "Cleaner":
        """
        Normalise string columns.

        Parameters
        ----------
        columns : list of str, optional
            Specific columns to clean. If None, applies to all
            columns inferred as text or categorical.
        normalize_case : bool, optional
            Title-case string values. Default is True.
        strip_whitespace : bool, optional
            Strip leading and trailing whitespace. Default is True.
        remove_special_chars : bool, optional
            Remove non-alphanumeric characters (excluding spaces).
            Default is False — opt-in only since it can be destructive.

        Returns
        -------
        Cleaner
            Returns self for method chaining.
        """
        from smartclean.modules.text import clean_text

        cleaned_cols, self._df = clean_text(
            self._df,
            profile=self._get_profile(),
            columns=columns,
            normalize_case=normalize_case,
            strip_whitespace=strip_whitespace,
            remove_special_chars=remove_special_chars,
        )

        if cleaned_cols:
            self._report.log("text_cleaned", cleaned_cols)

        return self

    def remove_outliers(
        self,
        method: Literal["iqr", "zscore"] = "iqr",
        action: Literal["remove", "cap", "flag"] = "cap",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
    ) -> "Cleaner":
        """
        Detect and handle outliers in numeric columns.

        Parameters
        ----------
        method : {"iqr", "zscore"}, optional
            Detection method. Default is "iqr".
            Note: "zscore" requires scipy to be installed.
        action : {"remove", "cap", "flag"}, optional
            How to handle detected outliers:
            - "remove" : drop rows containing outliers
            - "cap"    : clip values to the boundary (default)
            - "flag"   : add a boolean column marking outlier rows
        columns : list of str, optional
            Specific numeric columns to check. If None, applies to
            all numeric columns detected by the profiler.
        threshold : float, optional
            IQR multiplier (default 1.5) or Z-score threshold (default 3.0
            when method="zscore"). Ignored if you pass a custom value.

        Returns
        -------
        Cleaner
            Returns self for method chaining.
        """
        from smartclean.modules.outliers import remove_outliers

        result, self._df = remove_outliers(
            self._df,
            profile=self._get_profile(),
            method=method,
            action=action,
            columns=columns,
            threshold=threshold,
        )

        if result:
            self._report.log("outliers_handled", result)

        self._invalidate_profile()
        return self

    def output(self) -> pd.DataFrame:
        """
        Return the cleaned DataFrame.

        This is the terminal method of the chaining API.
        After calling output(), the Cleaner instance is no longer
        needed and can be discarded.

        Returns
        -------
        pd.DataFrame
            The cleaned copy of the original DataFrame.
        """
        return self._df.copy()

    def get_report(self) -> CleaningReport:
        """
        Return the CleaningReport accumulated during cleaning.

        Returns
        -------
        CleaningReport
            The report object. Call .summary(), .to_dict(), or .to_df()
            on the returned object.

        Example
        -------
        >>> cleaner.get_report().summary()
        >>> cleaner.get_report().to_dict()
        >>> cleaner.get_report().to_df()
        """
        return self._report

    # ── Dunder helpers ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        rows, cols = self._df.shape
        orig_rows, orig_cols = self._original_shape
        return (
            f"Cleaner("
            f"original={orig_rows}×{orig_cols}, "
            f"current={rows}×{cols}, "
            f"drop_threshold={self._drop_if_missing_pct})"
        )