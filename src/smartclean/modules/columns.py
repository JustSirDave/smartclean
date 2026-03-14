"""
smartclean/modules/columns.py

Column name normalisation module.

Responsible for converting messy column names into clean,
consistent snake_case identifiers that are safe to use as
Python variable names and pandas column accessors.

Transformations applied (in order):
1. Strip leading/trailing whitespace
2. Convert to lowercase
3. Replace spaces, hyphens, and dots with underscores
4. Remove any remaining non-alphanumeric characters (except underscores)
5. Collapse multiple consecutive underscores into one
6. Strip leading/trailing underscores
7. Prefix with "col_" if the result starts with a digit

Examples:
    "First Name"     → "first_name"
    "TOTAL SALES"    → "total_sales"
    "customer-id"    → "customer_id"
    "customer.id"    → "customer_id"
    "  Age  "        → "age"
    "100score"       → "col_100score"
    "__weird__"      → "weird"
    "A"              → "a"
"""

from __future__ import annotations

import re
import pandas as pd


def snake_case(name: str) -> str:
    """
    Convert a single column name string to snake_case.

    Parameters
    ----------
    name : str
        The original column name.

    Returns
    -------
    str
        The normalised snake_case column name.

    Examples
    --------
    >>> snake_case("First Name")
    'first_name'
    >>> snake_case("TOTAL-SALES")
    'total_sales'
    >>> snake_case("customer.id")
    'customer_id'
    >>> snake_case("  Age  ")
    'age'
    >>> snake_case("100score")
    'col_100score'
    """
    # 1. Ensure string type
    name = str(name)

    # 2. Strip leading/trailing whitespace
    name = name.strip()

    # 3. Convert to lowercase
    name = name.lower()

    # 4. Replace spaces, hyphens, dots, and slashes with underscores
    name = re.sub(r"[\s\-\./\\]+", "_", name)

    # 5. Remove any remaining characters that are not alphanumeric or underscore
    name = re.sub(r"[^\w]", "", name)

    # 6. Collapse multiple consecutive underscores into one
    name = re.sub(r"_+", "_", name)

    # 7. Strip leading/trailing underscores
    name = name.strip("_")

    # 8. If result starts with a digit, prefix with "col_"
    if name and name[0].isdigit():
        name = "col_" + name

    # 9. Fallback for empty string result
    if not name:
        name = "unnamed"

    return name


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise all column names in a DataFrame to snake_case.

    This function does not modify the original DataFrame — it returns
    a new DataFrame with renamed columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose column names should be normalised.

    Returns
    -------
    pd.DataFrame
        A DataFrame with all column names converted to snake_case.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=["First Name", "TOTAL SALES", "customer-id"])
    >>> clean_column_names(df).columns.tolist()
    ['first_name', 'total_sales', 'customer_id']
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )

    new_columns = [snake_case(col) for col in df.columns]

    # Handle duplicate column names after normalisation
    # e.g. "Name" and "name" would both become "name"
    seen: dict[str, int] = {}
    deduplicated = []
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            deduplicated.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduplicated.append(col)

    df = df.copy()
    df.columns = deduplicated
    return df


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias for clean_column_names().

    Provided for API consistency with the SRS function spec.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    return clean_column_names(df)