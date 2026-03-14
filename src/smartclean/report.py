"""
smartclean/report.py

CleaningReport — accumulates and presents a summary of all
transformations applied during a cleaning session.

Usage:
    report = cleaner.get_report()
    report.summary()        # prints human-readable table
    report.to_dict()        # returns raw dict
    report.to_df()          # returns pandas DataFrame
"""

from __future__ import annotations

import pandas as pd
from typing import Any


class CleaningReport:
    """
    Accumulates a log of all cleaning operations performed by Cleaner.

    Each cleaning method on the Cleaner class calls .log() as it runs,
    appending structured records. The report can be inspected at any
    point during or after cleaning.

    Keys logged by each Cleaner method:
    - columns_renamed        : dict  {old_name: new_name}
    - types_converted        : dict  {column: new_dtype}
    - columns_dropped        : dict  {column: {reason, value}}
    - missing_values_filled  : dict  {column: {count, strategy}}
    - duplicates_removed     : int
    - text_cleaned           : list  [column, ...]
    - outliers_handled       : dict  {column: {count, method, action}}
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def log(self, key: str, value: Any) -> None:
        """
        Append or update an entry in the report.

        Parameters
        ----------
        key : str
            The operation key (e.g. "missing_values_filled").
        value : Any
            The structured value to store for that operation.
        """
        self._data[key] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Return the full report as a Python dict.

        Returns
        -------
        dict
            Raw structured report data.
        """
        return dict(self._data)

    def to_df(self) -> pd.DataFrame:
        """
        Return the report as a flat pandas DataFrame.

        Each row represents one operation and one affected column
        (or a scalar value for operations like duplicate removal).

        Returns
        -------
        pd.DataFrame
            Columns: operation, column, detail
        """
        rows = []

        for operation, value in self._data.items():
            if isinstance(value, dict):
                for col, detail in value.items():
                    rows.append({
                        "operation": operation,
                        "column": col,
                        "detail": detail,
                    })
            elif isinstance(value, list):
                for col in value:
                    rows.append({
                        "operation": operation,
                        "column": col,
                        "detail": None,
                    })
            else:
                rows.append({
                    "operation": operation,
                    "column": None,
                    "detail": value,
                })

        if not rows:
            return pd.DataFrame(columns=["operation", "column", "detail"])

        return pd.DataFrame(rows)

    def summary(self) -> None:
        """
        Print a human-readable cleaning summary to stdout.

        Example output:
            ╔══════════════════════════════════════════╗
            ║         SmartClean — Cleaning Summary    ║
            ╚══════════════════════════════════════════╝
            Columns renamed          :  4
            Types converted          :  3
            Columns dropped          :  1  (notes — 95.02% missing)
            Missing values filled    : 42
            Duplicate rows removed   :  5
            Text columns cleaned     :  6
            Outliers handled         :  7
        """
        print("\n" + "═" * 46)
        print("      SmartClean — Cleaning Summary")
        print("═" * 46)

        if not self._data:
            print("  No cleaning operations were recorded.")
            print("═" * 46 + "\n")
            return

        # ── Columns renamed ─────────────────────────────────────────────
        renamed = self._data.get("columns_renamed", {})
        if renamed:
            print(f"  {'Columns renamed':<28}: {len(renamed):>4}")
            for old, new in renamed.items():
                print(f"    {old}  →  {new}")

        # ── Types converted ─────────────────────────────────────────────
        types = self._data.get("types_converted", {})
        if types:
            print(f"  {'Types converted':<28}: {len(types):>4}")
            for col, dtype in types.items():
                print(f"    {col}  →  {dtype}")

        # ── Columns dropped ─────────────────────────────────────────────
        dropped = self._data.get("columns_dropped", {})
        if dropped:
            print(f"  {'Columns dropped':<28}: {len(dropped):>4}")
            for col, info in dropped.items():
                pct = info.get("value", 0) * 100
                reason = info.get("reason", "")
                print(f"    {col}  ({pct:.1f}% missing — {reason})")

        # ── Missing values filled ────────────────────────────────────────
        filled = self._data.get("missing_values_filled", {})
        if filled:
            total_filled = sum(v.get("count", 0) for v in filled.values())
            print(f"  {'Missing values filled':<28}: {total_filled:>4}")
            for col, info in filled.items():
                count = info.get("count", 0)
                strategy = info.get("strategy", "")
                print(f"    {col}  ({count} values — {strategy})")

        # ── Duplicates removed ───────────────────────────────────────────
        dupes = self._data.get("duplicates_removed")
        if dupes is not None:
            print(f"  {'Duplicate rows removed':<28}: {dupes:>4}")

        # ── Text cleaned ─────────────────────────────────────────────────
        text = self._data.get("text_cleaned", [])
        if text:
            print(f"  {'Text columns cleaned':<28}: {len(text):>4}")
            print(f"    {', '.join(text)}")

        # ── Outliers handled ─────────────────────────────────────────────
        outliers = self._data.get("outliers_handled", {})
        if outliers:
            total_outliers = sum(
                v.get("count", 0) for v in outliers.values()
            )
            print(f"  {'Outliers handled':<28}: {total_outliers:>4}")
            for col, info in outliers.items():
                count = info.get("count", 0)
                method = info.get("method", "")
                action = info.get("action", "")
                print(f"    {col}  ({count} — {method}, {action})")

        print("═" * 46 + "\n")

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"CleaningReport(operations={keys})"