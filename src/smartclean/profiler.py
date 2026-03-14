"""
smartclean/profiler.py

Responsible for inspecting a DataFrame and returning a structured
ProfileResult object. This is the internal contract that all cleaning
modules depend on — run this first before any cleaning step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Profile of a single DataFrame column."""

    name: str
    dtype: str                    # semantic type: 'numeric' | 'categorical' | 'datetime' | 'text'
    missing_count: int
    missing_pct: float            # 0.0 – 1.0
    unique_count: int
    is_constant: bool             # True when all non-null values are identical
    potential_outlier_count: int  # estimated via IQR; 0 for non-numeric columns


@dataclass
class ProfileResult:
    """Full profile of a DataFrame, returned by profile()."""

    row_count: int
    col_count: int
    duplicate_row_count: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def missing_columns(self) -> list[ColumnProfile]:
        """Return columns that have at least one missing value."""
        return [c for c in self.columns.values() if c.missing_count > 0]

    def columns_by_dtype(self, dtype: str) -> list[ColumnProfile]:
        """Return columns matching a semantic dtype."""
        return [c for c in self.columns.values() if c.dtype == dtype]

    def columns_above_missing_threshold(self, threshold: float = 0.95) -> list[ColumnProfile]:
        """Return columns whose missing_pct exceeds the given threshold."""
        return [c for c in self.columns.values() if c.missing_pct > threshold]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "Dataset Profile",
            "=" * 40,
            f"Rows               : {self.row_count:,}",
            f"Columns            : {self.col_count:,}",
            f"Duplicate rows     : {self.duplicate_row_count:,}",
            "",
            f"{'Column':<20} {'dtype':<14} {'missing':>8} {'missing%':>10} {'outliers':>10}",
            "-" * 66,
        ]
        for col in self.columns.values():
            flag = "  <- exceeds drop threshold" if col.missing_pct > 0.95 else ""
            lines.append(
                f"{col.name:<20} {col.dtype:<14} {col.missing_count:>8,} "
                f"{col.missing_pct * 100:>9.2f}% {col.potential_outlier_count:>10,}{flag}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_dtype(series: pd.Series) -> str:
    """
    Infer a semantic dtype for a column — independent of the raw pandas dtype.

    Returns one of: 'numeric', 'datetime', 'categorical', 'text'
    """
    # Drop nulls before inspection
    sample = series.dropna()

    if sample.empty:
        return "categorical"  # nothing to infer; treat as categorical

    # Already a numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # String columns — attempt numeric coercion
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        coerced = pd.to_numeric(sample, errors="coerce")
        if coerced.notna().mean() >= 0.9:
            return "numeric"

        # Attempt datetime coercion on a small sample for performance
        try:
            pd.to_datetime(sample.head(50), errors="raise")
            return "datetime"
        except (ValueError, TypeError):
            pass

        # High cardinality → free text; low cardinality → categorical
        unique_ratio = series.nunique(dropna=True) / max(len(sample), 1)
        avg_length = sample.astype(str).str.len().mean()

        if unique_ratio > 0.5 and avg_length > 20:
            return "text"

        return "categorical"

    return "categorical"


def _count_outliers_iqr(series: pd.Series) -> int:
    """
    Estimate outlier count using the IQR method.
    """
    try:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 4:
            return 0

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return 0

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((clean < lower) | (clean > upper)).sum())
    except (TypeError, ValueError):
        return 0


def _profile_column(series: pd.Series) -> ColumnProfile:
    """Build a ColumnProfile for a single column."""
    total = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = missing_count / total if total > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    dtype = _infer_dtype(series)

    non_null = series.dropna()
    is_constant = (unique_count <= 1) and (len(non_null) > 0)

    is_boolean = pd.api.types.is_bool_dtype(series) or str(series.dtype) == "boolean"
    outlier_count = (
        _count_outliers_iqr(series.copy()) if dtype == "numeric" and not is_boolean else 0
    )

    return ColumnProfile(
        name=series.name,
        dtype=dtype,
        missing_count=missing_count,
        missing_pct=round(missing_pct, 6),
        unique_count=unique_count,
        is_constant=is_constant,
        potential_outlier_count=outlier_count,
    )


def _count_duplicates(df: pd.DataFrame) -> int:
    """Count fully duplicated rows (all columns identical)."""
    return int(df.duplicated().sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile(df: pd.DataFrame) -> ProfileResult:
    """
    Inspect a DataFrame and return a ProfileResult.

    This is the entry point for the profiling module and should be
    called before any cleaning step.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.

    Returns
    -------
    ProfileResult
        Structured profile containing row/col counts, duplicate count,
        and a ColumnProfile for every column.

    Example
    -------
    >>> import pandas as pd
    >>> import smartclean as sc
    >>> df = pd.read_csv("data.csv")
    >>> result = sc.profile(df)
    >>> print(result.summary())
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    columns: dict[str, ColumnProfile] = {
        col: _profile_column(df[col]) for col in df.columns
    }

    return ProfileResult(
        row_count=len(df),
        col_count=len(df.columns),
        duplicate_row_count=_count_duplicates(df),
        columns=columns,
    )