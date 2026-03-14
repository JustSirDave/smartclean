"""
smartclean/io.py

Dataset ingestion module.

Responsible for loading datasets from common file formats into
pandas DataFrames. Supports CSV, Excel, JSON, and passing through
an existing DataFrame directly.

Functions:
    read()        — smart loader, detects format from file extension
    read_csv()    — load a CSV file
    read_excel()  — load an Excel file
    read_json()   — load a JSON file
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def read(
    source: Union[str, Path, pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """
    Load a dataset from a file path or pass through a DataFrame.

    Automatically detects the file format from the extension and
    delegates to the appropriate reader.

    Supported formats: .csv, .xlsx, .xls, .json

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        The file path to load, or an existing DataFrame to pass through.
    **kwargs
        Additional keyword arguments passed to the underlying
        pandas reader (e.g. sep=";", sheet_name="Sheet2").

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    TypeError
        If source is not a string, Path, or DataFrame.
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file path does not exist.

    Examples
    --------
    >>> df = sc.read("data.csv")
    >>> df = sc.read("data.xlsx", sheet_name="Sales")
    >>> df = sc.read("data.json")
    >>> df = sc.read(existing_df)   # pass-through
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    readers = {
        ".csv":  read_csv,
        ".xlsx": read_excel,
        ".xls":  read_excel,
        ".json": read_json,
    }

    if extension not in readers:
        raise ValueError(
            f"Unsupported file format '{extension}'. "
            f"Supported formats: {list(readers.keys())}"
        )

    return readers[extension](path, **kwargs)


def read_csv(
    path: Union[str, Path],
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Attempts to detect the delimiter automatically if not specified.

    Parameters
    ----------
    path : str or Path
    encoding : str, optional
        File encoding. Default is "utf-8".
        Falls back to "latin-1" if utf-8 fails.
    **kwargs
        Additional arguments passed to pd.read_csv().

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)

    try:
        return pd.read_csv(path, encoding=encoding, engine="python", **kwargs)
    except UnicodeDecodeError:
        # Fallback encoding for files with special characters
        return pd.read_csv(path, encoding="latin-1", engine="python", **kwargs)


def read_excel(
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Load an Excel file into a DataFrame.

    Parameters
    ----------
    path : str or Path
    sheet_name : str or int, optional
        The sheet to load. Default is 0 (first sheet).
    **kwargs
        Additional arguments passed to pd.read_excel().

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_excel(Path(path), sheet_name=sheet_name, **kwargs)


def read_json(
    path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load a JSON file into a DataFrame.

    Parameters
    ----------
    path : str or Path
    **kwargs
        Additional arguments passed to pd.read_json().

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_json(Path(path), **kwargs)