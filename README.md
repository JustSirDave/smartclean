# SmartClean

**An intelligent, automatic data cleaning library for pandas DataFrames.**

SmartClean reduces the time data scientists and analysts spend on repetitive data preparation by providing an automatic cleaning pipeline that detects and resolves common data quality issues — with full transparency about every transformation applied.

```python
import smartclean as sc

# One-liner automatic cleaning
clean_df = sc.auto_clean(df)

# With full cleaning report
clean_df, report = sc.auto_clean(df, return_report=True)
report.summary()
```

---

## Table of Contents

- [Why SmartClean](#why-smartclean)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Auto Clean Pipeline](#auto-clean-pipeline)
- [Manual Cleaning API](#manual-cleaning-api)
- [Cleaning Modules](#cleaning-modules)
  - [Dataset Profiling](#dataset-profiling)
  - [Column Name Standardisation](#column-name-standardisation)
  - [Missing Value Handling](#missing-value-handling)
  - [Duplicate Detection and Removal](#duplicate-detection-and-removal)
  - [Data Type Correction](#data-type-correction)
  - [Text Cleaning](#text-cleaning)
  - [Outlier Detection and Handling](#outlier-detection-and-handling)
- [Cleaning Reports](#cleaning-reports)
- [Loading Data](#loading-data)
- [Configuration Reference](#configuration-reference)
- [Architecture](#architecture)
- [Development](#development)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why SmartClean

In most data science workflows, 60–80% of time is spent cleaning data. The problems are well known and repetitive:

| Problem | Example |
|---|---|
| Missing values | `age: NaN`, `salary: NaN` |
| Duplicate rows | Same record appears twice |
| Incorrect types | `"25"` stored as string instead of int |
| Inconsistent column names | `First Name`, `TOTAL-SALES`, `customer.id` |
| Text formatting issues | `"  USA "`, `"female"` vs `"Female"` |
| Outliers | Salary of `1,000,000` in a dataset where median is `55,000` |

Existing tools like pandas provide the primitives to fix these — but not a cohesive, automatic pipeline that handles all of them intelligently and transparently.

SmartClean addresses this gap.

---

## Installation

```bash
pip install dfsmartclean
```

With optional Z-score outlier detection (requires scipy):

```bash
pip install dfsmartclean[outliers]
```

For development:

```bash
pip install dfsmartclean[dev]
```

**Requirements:** Python 3.9+, pandas >= 1.5.0, numpy >= 1.23.0

---

## Quick Start

### Automatic cleaning (recommended for most users)

```python
import pandas as pd
import smartclean as sc

df = pd.read_csv("data.csv")

# Clean in one line
clean_df = sc.auto_clean(df)
```

### With a cleaning report

```python
clean_df, report = sc.auto_clean(df, return_report=True)

# Print human-readable summary
report.summary()

# Get structured data
report.to_dict()

# Get as a pandas DataFrame
report.to_df()
```

### Manual control via chained API

```python
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

# Access the report
cleaner.get_report().summary()
```

### Loading data

```python
df = sc.read("data.csv")
df = sc.read("data.xlsx", sheet_name="Sales")
df = sc.read("data.json")
```

---

## Core Concepts

### Automatic but transparent

SmartClean applies sensible defaults automatically — but records every transformation in a `CleaningReport` so you always know exactly what changed and why.

### Original data is never modified

Every operation works on an internal copy. Your original DataFrame is always safe:

```python
original_df = pd.read_csv("data.csv")
clean_df = sc.auto_clean(original_df)
# original_df is unchanged
```

### Semantic types

SmartClean uses its own semantic type system — independent of pandas dtypes — to make intelligent decisions about each column:

| Semantic type | Description | Auto fill strategy |
|---|---|---|
| `numeric` | Numbers (int, float, or string-encoded) | Median |
| `categorical` | Low-cardinality strings | Mode |
| `datetime` | Dates and timestamps | Forward fill |
| `text` | High-cardinality free text | Constant "unknown" |

### Modular architecture

Every cleaning step is an independent module. You can use any module directly without going through the pipeline:

```python
from smartclean.modules.missing import fill_median
from smartclean.modules.columns import clean_column_names
from smartclean.modules.duplicates import remove_duplicates
```

---

## Auto Clean Pipeline

`sc.auto_clean()` runs all cleaning steps in the correct sequence:

```
1. Profile dataset
2. Drop columns exceeding missing threshold
3. Clean column names → snake_case
4. Fix data types
5. Handle missing values
6. Remove duplicates
7. Clean text fields
8. Detect and handle outliers
9. Generate CleaningReport
```

### Signature

```python
sc.auto_clean(
    df,
    drop_if_missing_pct=0.95,
    outlier_method="iqr",
    outlier_action="cap",
    return_report=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | required | The DataFrame to clean |
| `drop_if_missing_pct` | `float` | `0.95` | Drop columns with more than this fraction of missing values. Set to `1.0` to disable. |
| `outlier_method` | `"iqr"` or `"zscore"` | `"iqr"` | Detection method for outliers |
| `outlier_action` | `"cap"`, `"remove"`, or `"flag"` | `"cap"` | How to handle detected outliers |
| `return_report` | `bool` | `False` | If `True`, returns `(cleaned_df, report)` tuple |

### Examples

```python
# Default — drops columns >95% empty, caps outliers using IQR
clean_df = sc.auto_clean(df)

# More aggressive column dropping
clean_df = sc.auto_clean(df, drop_if_missing_pct=0.50)

# Remove outlier rows instead of capping
clean_df = sc.auto_clean(df, outlier_action="remove")

# Flag outliers with a boolean column instead of modifying values
clean_df = sc.auto_clean(df, outlier_action="flag")

# Use Z-score instead of IQR for outlier detection
clean_df = sc.auto_clean(df, outlier_method="zscore")

# Get the full cleaning report
clean_df, report = sc.auto_clean(df, return_report=True)
report.summary()
```

---

## Manual Cleaning API

For users who need precise control over each step, the `Cleaner` class provides a fluent chained API.

### Creating a Cleaner

```python
cleaner = sc.Cleaner(df)

# With custom missing value threshold
cleaner = sc.Cleaner(df, drop_if_missing_pct=0.80)
```

The `Cleaner` copies the DataFrame on initialisation — your original is never modified.

### Chaining methods

Every method returns `self`, allowing full method chaining:

```python
clean_df = (
    sc.Cleaner(df)
    .clean_columns()
    .fix_types()
    .handle_missing()
    .remove_duplicates()
    .clean_text()
    .remove_outliers()
    .output()
)
```

### Method reference

#### `.clean_columns()`
Normalises all column names to snake_case.

```python
cleaner.clean_columns()
# "First Name" → "first_name"
# "TOTAL-SALES" → "total_sales"
# "customer.id" → "customer_id"
```

#### `.fix_types()`
Detects and converts incorrectly typed columns.

```python
cleaner.fix_types()
# "25" → 25 (int)
# "2023-01-15" → Timestamp
# "true" → True (boolean)
```

#### `.handle_missing(strategy="auto", columns=None)`
Fills missing values using the specified strategy.

```python
# Auto strategy (recommended) — picks per column type
cleaner.handle_missing()

# Manual strategy for specific columns
cleaner.handle_missing(strategy="median", columns=["age", "salary"])
cleaner.handle_missing(strategy="mode", columns=["department"])
```

| Strategy | Description |
|---|---|
| `"auto"` | Median for numeric, mode for categorical, forward fill for datetime |
| `"mean"` | Fill with column mean |
| `"median"` | Fill with column median |
| `"mode"` | Fill with most frequent value |

#### `.remove_duplicates(subset=None, keep="first")`
Removes duplicate rows.

```python
# Remove full-row duplicates
cleaner.remove_duplicates()

# Remove duplicates based on specific columns
cleaner.remove_duplicates(subset=["name", "email"])

# Keep last occurrence instead of first
cleaner.remove_duplicates(keep="last")

# Remove ALL occurrences of duplicated rows
cleaner.remove_duplicates(keep=False)
```

#### `.clean_text(columns=None, normalize_case=True, strip_whitespace=True, remove_special_chars=False)`
Normalises string columns.

```python
# Default — strip whitespace and title-case
cleaner.clean_text()
# "  alice  " → "Alice"
# "FEMALE" → "Female"

# Also remove special characters (opt-in — can be destructive)
cleaner.clean_text(remove_special_chars=True)

# Target specific columns only
cleaner.clean_text(columns=["name", "city"])
```

#### `.remove_outliers(method="iqr", action="cap", columns=None, threshold=1.5)`
Detects and handles outliers in numeric columns.

```python
# Default — IQR method, cap outliers
cleaner.remove_outliers()

# Remove outlier rows
cleaner.remove_outliers(action="remove")

# Flag outliers with a boolean column
cleaner.remove_outliers(action="flag")

# Z-score method
cleaner.remove_outliers(method="zscore")

# Stricter IQR threshold
cleaner.remove_outliers(threshold=3.0)

# Specific columns only
cleaner.remove_outliers(columns=["salary", "age"])
```

#### `.output()`
Returns the cleaned DataFrame. Always call this last.

```python
clean_df = cleaner.output()
```

#### `.get_report()`
Returns the `CleaningReport` accumulated during cleaning.

```python
report = cleaner.get_report()
report.summary()
```

---

## Cleaning Modules

All modules can be used independently of the pipeline and `Cleaner` class.

### Dataset Profiling

```python
from smartclean.profiler import profile

result = profile(df)
print(result.summary())
```

**Output:**
```
Dataset Profile
========================================
Rows               : 12,450
Columns            : 10
Duplicate rows     : 3

Column               dtype          missing   missing%   outliers
------------------------------------------------------------------
Age                  numeric             32      0.26%          4
Salary               numeric              5      0.04%          7
Department           categorical          0      0.00%          0
Notes                text             11830     95.02%          0  <- exceeds drop threshold
```

**ProfileResult attributes:**

```python
result.row_count              # int
result.col_count              # int
result.duplicate_row_count    # int
result.columns                # dict[str, ColumnProfile]

# Helpers
result.missing_columns()                          # columns with missing values
result.columns_by_dtype("numeric")                # columns by semantic type
result.columns_above_missing_threshold(0.95)      # columns to be dropped
```

**ColumnProfile attributes:**

```python
col = result.columns["Age"]
col.name                    # "Age"
col.dtype                   # "numeric"
col.missing_count           # 32
col.missing_pct             # 0.002566
col.unique_count            # 67
col.is_constant             # False
col.potential_outlier_count # 4
```

---

### Column Name Standardisation

```python
from smartclean.modules.columns import clean_column_names, snake_case

# Clean all columns in a DataFrame
df = clean_column_names(df)

# Convert a single name
snake_case("First Name")     # "first_name"
snake_case("TOTAL-SALES")    # "total_sales"
snake_case("customer.id")    # "customer_id"
snake_case("100score")       # "col_100score"
snake_case("__weird__")      # "weird"
```

**Transformations applied (in order):**
1. Strip leading/trailing whitespace
2. Convert to lowercase
3. Replace spaces, hyphens, dots, slashes with underscores
4. Remove remaining non-alphanumeric characters
5. Collapse multiple consecutive underscores
6. Strip leading/trailing underscores
7. Prefix with `col_` if result starts with a digit
8. Fallback to `unnamed` if result is empty

**Duplicate handling:** If two columns normalise to the same name (e.g. `"Name"` and `"name"` both become `"name"`), the second is automatically suffixed: `"name"` and `"name_1"`.

---

### Missing Value Handling

```python
from smartclean.modules.missing import (
    handle_missing,
    fill_mean,
    fill_median,
    fill_mode,
    fill_auto,
)

# Main function — uses profiler output for intelligent defaults
result, clean_df = handle_missing(df, profile=p, strategy="auto")

# Standalone fill functions
df = fill_mean(df, columns=["salary"])
df = fill_median(df, columns=["age", "salary"])
df = fill_mode(df, columns=["department", "gender"])
df = fill_auto(df, profile=p)
```

**Auto strategy defaults:**

| Column type | Strategy | Fallback |
|---|---|---|
| Numeric | Median | — |
| Categorical | Mode | Constant `"unknown"` |
| Datetime | Forward fill | Backward fill |
| Text | Constant `"unknown"` | — |

**Drop threshold:**

Columns with `missing_pct > drop_if_missing_pct` are dropped before any filling is attempted:

```python
result, clean_df = handle_missing(df, profile=p, drop_threshold=0.95)

# result["dropped"] → {"notes": {"reason": "missing_pct", "value": 0.9502}}
# result["filled"]  → {"age": {"count": 32, "strategy": "median"}}
```

---

### Duplicate Detection and Removal

```python
from smartclean.modules.duplicates import (
    detect_duplicates,
    remove_duplicates,
    count_duplicates,
)

# Count duplicates
n = count_duplicates(df)
print(f"{n} duplicate rows found")

# Get a boolean mask of duplicate rows
mask = detect_duplicates(df)
print(df[mask])  # view the duplicate rows

# Remove duplicates — returns a new DataFrame with reset index
clean_df = remove_duplicates(df)

# Subset-based: duplicates on specific columns only
clean_df = remove_duplicates(df, subset=["name", "email"])

# Keep last occurrence
clean_df = remove_duplicates(df, keep="last")

# Remove ALL occurrences of duplicated rows (keep none)
clean_df = remove_duplicates(df, keep=False)
```

---

### Data Type Correction

```python
from smartclean.modules.types import (
    fix_types,
    convert_numeric,
    convert_datetime,
    convert_boolean,
)

# Main function — uses profiler output
conversions, clean_df = fix_types(df, profile=p)
# conversions → {"age": "int64", "salary": "float64", "active": "boolean"}

# Standalone converters
df = convert_numeric(df, columns=["age", "salary"])
df = convert_datetime(df, columns=["signup_date"])
df = convert_boolean(df, columns=["active", "verified"])
```

**Detected conversions:**

| Before | After |
|---|---|
| `"25"` | `int64` |
| `"3.14"` | `float64` |
| `"2023-01-15"` | `datetime64` |
| `"true"` / `"false"` | `boolean` |
| `"yes"` / `"no"` | `boolean` |
| `"1"` / `"0"` | `boolean` |

Conversions are only applied when at least 80% of non-null values in the column successfully convert — preventing accidental corruption of mixed-type columns.

---

### Text Cleaning

```python
from smartclean.modules.text import (
    clean_text,
    strip_whitespace,
    normalize_case,
    remove_special_chars,
)

# Main function — uses profiler output to auto-select text/categorical columns
cleaned_cols, clean_df = clean_text(df, profile=p)

# Standalone functions
df = strip_whitespace(df)                          # all object columns
df = strip_whitespace(df, columns=["name", "city"])

df = normalize_case(df, case="title")              # default
df = normalize_case(df, case="lower")
df = normalize_case(df, case="upper")

df = remove_special_chars(df)                      # keeps spaces by default
df = remove_special_chars(df, keep_spaces=False)   # removes spaces too
```

**Examples:**

```python
" USA "    → "Usa"       # strip + title case
"FEMALE"   → "Female"    # title case
"hello!"   → "hello"     # remove special chars
"Bob@work" → "Bobwork"   # remove special chars
```

**Note:** `remove_special_chars` is opt-in in `clean_text()` — set `remove_special_chars=True` to enable it. It defaults to off because it can be destructive on columns like email addresses or product codes.

---

### Outlier Detection and Handling

```python
from smartclean.modules.outliers import (
    remove_outliers,
    detect_outliers_iqr,
    detect_outliers_zscore,
)

# Main function — uses profiler output
result, clean_df = remove_outliers(df, profile=p)
# result → {"salary": {"count": 7, "method": "iqr", "action": "cap"}}

# Detection only — returns a boolean mask
mask = detect_outliers_iqr(df["salary"])
mask = detect_outliers_iqr(df["salary"], threshold=3.0)  # stricter

mask = detect_outliers_zscore(df["salary"])
mask = detect_outliers_zscore(df["salary"], threshold=2.5)
```

**Detection methods:**

| Method | Description | Default threshold |
|---|---|---|
| `"iqr"` | Values outside Q1 − 1.5×IQR or Q3 + 1.5×IQR | 1.5 |
| `"zscore"` | Values with \|z-score\| above threshold | 3.0 |

**Handling actions:**

| Action | Description |
|---|---|
| `"cap"` | Clip values to the IQR/z-score boundary (default — non-destructive) |
| `"remove"` | Drop rows containing outliers |
| `"flag"` | Add a boolean `{column}_outlier` column marking outlier rows |

**IQR is the recommended default** for most use cases — it is robust to non-normal distributions and works well on small datasets. Z-score assumes normality and requires larger samples (50+ rows) to be reliable.

---

## Cleaning Reports

The `CleaningReport` object is accumulated incrementally as each cleaning step runs. Access it via `return_report=True` or `cleaner.get_report()`.

### Output modes

```python
# Human-readable summary printed to stdout
report.summary()

# Raw Python dict — for programmatic use
d = report.to_dict()

# Flat pandas DataFrame — one row per operation per column
df = report.to_df()
```

### Example summary output

```
══════════════════════════════════════════════
      SmartClean — Cleaning Summary
══════════════════════════════════════════════
  Columns renamed              :    4
    First Name  →  first_name
    TOTAL SALES  →  total_sales
  Types converted              :    3
    age  →  int64
    active  →  boolean
  Columns dropped              :    1
    notes  (95.0% missing — missing_pct)
  Missing values filled        :   42
    Age  (32 values — median)
    Department  (10 values — mode)
  Duplicate rows removed       :    5
  Text columns cleaned         :    3
    sex, embarked, name
  Outliers handled             :    7
    Fare  (7 — iqr, cap)
══════════════════════════════════════════════
```

### Report dict structure

```python
{
    "columns_renamed": {
        "First Name": "first_name",
        "TOTAL SALES": "total_sales",
    },
    "types_converted": {
        "age": "int64",
        "active": "boolean",
    },
    "columns_dropped": {
        "notes": {"reason": "missing_pct", "value": 0.9502}
    },
    "missing_values_filled": {
        "Age":        {"count": 32, "strategy": "median"},
        "Department": {"count": 10, "strategy": "mode"},
    },
    "duplicates_removed": 5,
    "text_cleaned": ["sex", "embarked", "name"],
    "outliers_handled": {
        "Fare": {"count": 7, "method": "iqr", "action": "cap"}
    },
}
```

---

## Loading Data

```python
import smartclean as sc

# Auto-detect format from extension
df = sc.read("data.csv")
df = sc.read("data.xlsx")
df = sc.read("data.json")

# Pass keyword arguments to the underlying pandas reader
df = sc.read("data.csv", sep=";")
df = sc.read("data.xlsx", sheet_name="Q3 Sales")
df = sc.read("data.json", orient="records")

# Pass through an existing DataFrame (returns a copy)
df = sc.read(existing_df)
```

**Supported formats:** `.csv`, `.xlsx`, `.xls`, `.json`

CSV files automatically fall back to `latin-1` encoding if `utf-8` fails — useful for files exported from Excel with special characters.

---

## Configuration Reference

### `auto_clean()` parameters

| Parameter | Default | Description |
|---|---|---|
| `drop_if_missing_pct` | `0.95` | Drop columns with > 95% missing values. Set to `1.0` to disable. |
| `outlier_method` | `"iqr"` | `"iqr"` or `"zscore"` |
| `outlier_action` | `"cap"` | `"cap"`, `"remove"`, or `"flag"` |
| `return_report` | `False` | Return `(df, report)` tuple if `True` |

### `Cleaner()` parameters

| Parameter | Default | Description |
|---|---|---|
| `drop_if_missing_pct` | `0.95` | Threshold for auto-dropping sparse columns |

### `handle_missing()` strategies

| Strategy | Best for |
|---|---|
| `"auto"` | General use — picks per column type |
| `"mean"` | Normally distributed numeric data |
| `"median"` | Skewed numeric data or data with outliers |
| `"mode"` | Categorical data |

### `remove_outliers()` thresholds

| Method | Conservative | Standard | Strict |
|---|---|---|---|
| IQR | `3.0` | `1.5` (default) | `1.0` |
| Z-score | `3.5` | `3.0` (default) | `2.5` |

---

## Architecture

SmartClean follows a modular pipeline design. Each module is independent and can be used standalone or via the pipeline.

```
Data Input (CSV / Excel / JSON / DataFrame)
    ↓
IO Module          sc.read()
    ↓
Profiling Module   sc.profile()  →  ProfileResult
    ↓
Cleaning Modules
    ├── columns.py     clean_column_names()
    ├── types.py       fix_types()
    ├── missing.py     handle_missing()
    ├── duplicates.py  remove_duplicates()
    ├── text.py        clean_text()
    └── outliers.py    remove_outliers()
    ↓
Pipeline Engine    sc.auto_clean()  /  sc.Cleaner()
    ↓
Report Generator   CleaningReport
```

**Project structure:**

```
smartclean/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── smartclean/
│       ├── __init__.py        # public API
│       ├── io.py              # dataset loading
│       ├── profiler.py        # dataset inspection
│       ├── cleaner.py         # manual chained API
│       ├── pipeline.py        # auto_clean()
│       ├── report.py          # CleaningReport
│       └── modules/
│           ├── columns.py     # column name normalisation
│           ├── missing.py     # missing value handling
│           ├── duplicates.py  # duplicate detection/removal
│           ├── types.py       # data type correction
│           ├── text.py        # text cleaning
│           └── outliers.py    # outlier detection/handling
├── tests/
│   ├── test_profiler.py
│   ├── test_columns.py
│   ├── test_missing.py
│   ├── test_duplicates.py
│   ├── test_types.py
│   ├── test_text.py
│   ├── test_outliers.py
│   └── test_pipeline.py
└── docs/
```

**Dependencies:**

| Package | Version | Required |
|---|---|---|
| pandas | >= 1.5.0 | Yes |
| numpy | >= 1.23.0 | Yes |
| scipy | >= 1.9.0 | Optional (Z-score outliers only) |

---

## Development

### Setup

```bash
git clone https://github.com/yourname/smartclean.git
cd smartclean

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -e ".[dev]"
```

### Install with all optional dependencies

```bash
pip install -e ".[all]"
```

### Project commands

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=src/smartclean --cov-report=term-missing

# Run a specific test file
pytest tests/test_pipeline.py -v

# Lint
ruff check src/

# Type check
mypy src/
```

---

## Running Tests

SmartClean has a comprehensive test suite of **242 tests** covering all modules.

```bash
pytest tests/ -v
```

**Test coverage by module:**

| Module | Tests | Coverage |
|---|---|---|
| `profiler.py` | 27 | 84% |
| `columns.py` | 37 | 100% |
| `missing.py` | 36 | 81% |
| `duplicates.py` | 38 | 100% |
| `types.py` | 22 | 77% |
| `text.py` | 27 | 98% |
| `outliers.py` | 22 | 91% |
| `pipeline.py` | 23 | 100% |
| `report.py` | — | 86% |
| **Total** | **242** | **79%** |

Tests use the Titanic dataset for real-world validation — it contains missing values, high-missing columns, outliers, inconsistent text, and known data quality issues that exercise every code path.

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Add or update tests to cover your changes
5. Ensure all tests pass: `pytest tests/ -v`
6. Run the linter: `ruff check src/`
7. Commit with a descriptive message: `git commit -m "feat: add X"`
8. Push and open a Pull Request

**Commit message format:**

```
feat: add new feature
fix: fix a bug
docs: update documentation
test: add or update tests
refactor: refactor code without changing behaviour
chore: update dependencies or configuration
```

**Adding a new cleaning module:**

1. Create `src/smartclean/modules/your_module.py`
2. Follow the existing module pattern — return `(result_dict, cleaned_df)`
3. Add it to `src/smartclean/__init__.py`
4. Add a step in `src/smartclean/pipeline.py`
5. Add a method in `src/smartclean/cleaner.py`
6. Write tests in `tests/test_your_module.py`

---

## Roadmap

### Version 0.1.0 (current)
- [x] Dataset profiling with `ProfileResult`
- [x] Column name normalisation to snake_case
- [x] Missing value handling with auto strategy
- [x] Duplicate detection and removal
- [x] Data type correction
- [x] Text cleaning and normalisation
- [x] Outlier detection (IQR and Z-score)
- [x] Auto-clean pipeline
- [x] CleaningReport with three output modes
- [x] Fluent chained `Cleaner` API
- [x] 242 tests, 79% coverage

### Version 0.5.0 (planned)
- [ ] Configurable cleaning profiles (save and reuse settings)
- [ ] Column value standardisation (e.g. `"USA"` / `"US"` / `"United States"` → `"USA"`)
- [ ] Schema validation — assert expected columns and types
- [ ] HTML cleaning report export
- [ ] Jupyter notebook integration — inline report display

### Version 1.0.0 (planned)
- [ ] Stable public API
- [ ] Full documentation site
- [ ] Performance optimisation for datasets > 1 million rows
- [ ] Plugin architecture for custom cleaning modules
- [ ] CLI interface: `smartclean data.csv --output clean.csv`

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/).
Tested against the [Titanic dataset](https://www.kaggle.com/c/titanic) from Kaggle.

---

*SmartClean is an open-source project. If it saves you time, consider giving it a ⭐ on GitHub.*