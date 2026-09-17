"""
Semantic type inference for columns (numeric, categorical, datetime, text, etc.).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from .config import ColumnTypeConfig

logger = logging.getLogger(__name__)


SemanticType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "boolean",
    "datetime",
    "time",
    "text",
    "identifier",
    "constant",
]

# Spanish labels for the inferred types, shown per column in the HTML report and the CSV tables.
SEMANTIC_TYPE_LABELS: dict[str, str] = {
    "numeric_continuous": "Continuous",
    "numeric_discrete": "Discrete",
    "categorical": "Categorical",
    "boolean": "Boolean",
    "datetime": "Date/time",
    "time": "Time of day",
    "text": "Free text",
    "identifier": "Identifier",
    "constant": "Constant",
}


def semantic_type_label(semantic_type: str | None) -> str:
    """Report label for an inferred type; an unknown or missing type is shown as a dash."""
    if not semantic_type:
        return "—"
    return SEMANTIC_TYPE_LABELS.get(semantic_type, semantic_type)


# Short names for the storage dtypes pandas reports. The loader reads with the nullable backend,
# so a column of whole numbers arrives as "Int64" and one of text as "string"; the report shows
# "int" and "text" and keeps the exact dtype alongside.
_DTYPE_LABEL_PREFIXES: tuple[tuple[str, str], ...] = (
    ("int", "int"),
    ("uint", "int"),
    ("float", "float"),
    ("complex", "complex"),
    ("bool", "bool"),
    ("datetime", "date"),
    ("timedelta", "duration"),
    ("period", "period"),
    ("category", "category"),
    ("string", "text"),
    ("str", "text"),
    ("object", "text"),
)


def dtype_label(dtype: object) -> str:
    """Short name for a pandas dtype ("Int64" -> "int", "string" -> "texto"); others as they are."""
    if not dtype:
        return "—"
    name = str(dtype).lower()
    for prefix, label in _DTYPE_LABEL_PREFIXES:
        if name.startswith(prefix):
            return label
    return str(dtype)


# Cheap regex pre-filter so obviously non-date strings (e.g. "24,598029558215444")
# never reach pd.to_datetime. Restricted to the separators used by the explicit
# formats below ('-', '/'), which also keeps decimal-comma numbers out.
_DATE_LIKE_PATTERN = re.compile(r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*$")

# Explicit candidate formats tried in order. ISO forms come first (unambiguous),
# then day-first forms (preferred for Spanish-speaking users), then month-first
# forms as a last resort. Using an explicit `format=` avoids the slow/ambiguous
# dateutil fallback and the associated "Could not infer format" UserWarning.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m-%d-%Y",
    # Two-digit years, last so a four-digit year always wins, and day-first before month-first as
    # above. Date Egg in penguins_lter ("11/11/07") was read as a category with 50 values; across
    # the 19 datasets in data/raw these formats change no other column.
    "%d/%m/%y",
    "%d-%m-%y",
    "%m/%d/%y",
    "%m-%d-%y",
)

_DATE_SUCCESS_RATIO_THRESHOLD = 0.9


def _looks_like_date_string(value: object) -> bool:
    """Cheap regex pre-check: does this look like a date at all?"""
    if not isinstance(value, str):
        return False
    return bool(_DATE_LIKE_PATTERN.match(value))


def _detect_datetime_series(valid: pd.Series, sample_size: int = 200) -> bool:
    """
    Decide whether a string series is predominantly dates.

    Only strings that pass a cheap date-like regex are handed to
    ``pd.to_datetime``, and only with explicit formats (never the free-form
    dateutil fallback), so numeric-looking text is never misclassified and no
    warnings are emitted.
    """
    sample = valid.iloc[: min(sample_size, len(valid))]
    if len(sample) == 0:
        return False

    date_like = sample.map(_looks_like_date_string)
    if date_like.mean() <= _DATE_SUCCESS_RATIO_THRESHOLD:
        return False

    candidates = sample[date_like]
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(candidates, format=fmt, errors="coerce")
        if parsed.notna().mean() > _DATE_SUCCESS_RATIO_THRESHOLD:
            return True

    return False


def coerce_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a series to datetime, for an explicit ``column_types.datetime`` override and wherever
    a column inferred as a date is read as one (statistics, time series).

    Tries the same day-first-preferring explicit formats used for detection
    (``_DATE_FORMATS``) and keeps whichever parses the most values. Plain
    ``pd.to_datetime(series, errors="coerce")`` would instead lock onto
    whatever format pandas infers from the first valid value and silently
    turn every non-matching (but still valid, e.g. day-first) date into NaT.
    Falls back to pandas' own inference only if none of the explicit formats
    parse anything, so unusual formats are still handled leniently.
    """
    best_parsed, best_count = None, -1
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        count = int(parsed.notna().sum())
        if count > best_count:
            best_parsed, best_count = parsed, count

    if best_count > 0:
        return best_parsed
    return pd.to_datetime(series, errors="coerce")


# A time of day with no date, such as "11:43:47" or "09:30". The date formats above are tried
# first, so a full timestamp ("2024-01-01 11:43:47") stays a datetime.
_TIME_LIKE_PATTERN = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*$")
_TIME_FORMATS: tuple[str, ...] = ("%H:%M:%S", "%H:%M", "%H:%M:%S.%f")


def parse_time_of_day(series: pd.Series) -> pd.Series:
    """
    Parse "HH:MM[:SS]" text into timestamps on a placeholder date; only their clock part means anything.

    Every format is tried and the results are combined, so a column mixing "09:30" and "09:30:15"
    parses whole. Values no clock shows, like "25:30", become NaT.
    """
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in _TIME_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    return parsed


def _detect_time_series(valid: pd.Series, sample_size: int = 200) -> bool:
    """Decide whether a string series is predominantly times of day, the way dates are detected."""
    sample = valid.iloc[: min(sample_size, len(valid))]
    if len(sample) == 0:
        return False

    time_like = sample.str.match(_TIME_LIKE_PATTERN).fillna(False).astype(bool)
    if time_like.mean() <= _DATE_SUCCESS_RATIO_THRESHOLD:
        return False

    # The shape is not enough: "25:30" looks like a time, but no clock shows it.
    return parse_time_of_day(sample[time_like]).notna().mean() > _DATE_SUCCESS_RATIO_THRESHOLD


# A number that names an entity instead of measuring it: "id", "customer_id", "id_cliente",
# "orderId", "row_key". Deliberately narrow: "numero_creditos" in the Mexican data is a count, and
# a column called "humid" is not a key.
_IDENTIFIER_NAME_PATTERN = re.compile(r"^(?:id|uuid|guid)$|^id[_\-]|[_\-]id$|[_\-]key$", re.IGNORECASE)
_CAMEL_CASE_ID_PATTERN = re.compile(r"[a-z0-9]Id$")


def _has_identifier_name(name: object) -> bool:
    """Does this column name mark it as a key (id, customer_id, id_cliente, orderId, row_key)?"""
    if name is None:
        return False
    text = str(name).strip()
    return bool(_IDENTIFIER_NAME_PATTERN.search(text) or _CAMEL_CASE_ID_PATTERN.search(text))


# A counter numbers the rows instead of measuring them: from one row to the next it goes up by exactly
# one. Sample Number in penguins_lter (1 to 152) does so on 99.4% of its rows. Across every integer
# column with more than 20 values in the 19 datasets of data/raw, no measure comes close: the highest
# is numero_mensualidades_no_pagadas at 18.3%. Covering every value from 1 up is no signal on its own:
# units_sold and days_to_close in sales.csv do, in no particular order.
SEQUENCE_MIN_STEP_SHARE = 0.9


def _looks_like_sequence(valid: pd.Series) -> bool:
    """Does this integer column go up by exactly one from one row to the next, almost everywhere?"""
    steps = valid.diff().dropna()
    return len(steps) > 0 and float((steps == 1).mean()) >= SEQUENCE_MIN_STEP_SHARE


# Above numeric_discrete_threshold, a string column is free text only if most of its values are
# distinct; values that repeat are categories however long or numerous the labels are.
FREE_TEXT_MIN_UNIQUE_RATIO = 0.5

# An entity code such as "CUST-00001" or "ORD-2024-000001": letters and digits (both required, so
# plain labels like "Monterrey" and bare numbers are excluded), optionally joined by "-" or "_",
# and never containing spaces.
_CODE_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]+$")

# Share of a column's values that must look like codes before it is read as a key.
CODE_MIN_SHARE = 0.9

# A foreign key repeats, so the "mostly unique" rule cannot recognise it. Above this many distinct
# values, a column of codes names entities instead of describing them. The floor is the
# high-cardinality threshold (the pipeline passes the configured one), so a column of codes is
# either analysed as a category or recognised as an ID, but never flagged as high cardinality.
IDENTIFIER_MIN_UNIQUE = 100


def _looks_like_codes(valid: pd.Series) -> bool:
    """Do (almost) all of these values look like entity codes, e.g. "CUST-00001"?"""
    return float(valid.str.match(_CODE_PATTERN).fillna(False).mean()) >= CODE_MIN_SHARE


def infer_semantic_type(
    series: pd.Series,
    numeric_discrete_threshold: int = 20,
    identifier_min_unique: int = IDENTIFIER_MIN_UNIQUE,
) -> SemanticType:
    """
    Infer semantic type of a column.

    Args:
        series: The column to classify.
        numeric_discrete_threshold: Above this many distinct values a number is continuous.
        identifier_min_unique: Above this many distinct values, a column of codes is an
            identifier rather than a category (see IDENTIFIER_MIN_UNIQUE).

    Returns one of:
        - numeric_continuous: float or int with many unique values
        - numeric_discrete: int with few unique values (≤threshold)
        - categorical: string values that repeat (any label length)
        - boolean: True/False values
        - datetime: datetime or date string
        - time: a time of day without a date ("11:43:47", "09:30")
        - text: free text, i.e. string values that rarely repeat
        - identifier: codes that name an entity (IDs), either unique per row or repeated as a
          foreign key; also whole numbers whose column name marks them as keys, or that go up by
          one from row to row (a sample or row counter)
        - constant: all values the same
    """
    # Remove nulls
    valid = series.dropna()

    if len(valid) == 0:
        return "constant"

    # Check for constant
    if valid.nunique() == 1:
        return "constant"

    # Check for boolean
    unique_vals = set(valid.unique())
    if unique_vals <= {True, False, 0, 1, "True", "False", "true", "false", "yes", "no", "Yes", "No", "YES", "NO"}:
        return "boolean"

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Try to parse as datetime (string columns only; see _detect_datetime_series)
    if pd.api.types.is_string_dtype(series) and _detect_datetime_series(valid):
        return "datetime"

    # A time of day without a date: read as text, transaction_time on Vistara ("11:43:47") became a
    # category with 31702 values, flagged as high cardinality and drawn as a bar chart of instants.
    if pd.api.types.is_string_dtype(series) and _detect_time_series(valid):
        return "time"

    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        n_unique = valid.nunique()
        if n_unique <= numeric_discrete_threshold:
            return "numeric_discrete"
        # A key stored as a number is not a measurement: analysing it yields a histogram of
        # customer numbers, a VIF entry and scatter plots that mean nothing. Only whole numbers
        # whose name marks them as keys, or that count the rows one by one, qualify. Near-uniqueness
        # alone would not do: on the Mexican data, monto_siniestro holds 451 distinct amounts over
        # 452 rows.
        if pd.api.types.is_integer_dtype(series) and (_has_identifier_name(series.name) or _looks_like_sequence(valid)):
            return "identifier"
        return "numeric_continuous"

    # String-like columns
    if pd.api.types.is_string_dtype(series):
        n_unique = valid.nunique()
        n_total = len(valid)
        avg_length = valid.str.len().mean()

        # Check for identifier (mostly unique, alphanumeric, not too long)
        if n_unique / n_total > 0.9 and avg_length < 50:
            # Check if mostly alphanumeric
            alphanumeric_count = valid.str.match(r"^[a-zA-Z0-9_-]+$").sum()
            if alphanumeric_count / n_total > 0.8:
                return "identifier"

        # A foreign key repeats, so the rule above cannot see it: customer_id held 7063 codes
        # over 219432 rows and was read as a category, then flagged as high cardinality. Past the
        # floor, a column of codes names entities (CUST-00001) instead of describing them.
        if n_unique > identifier_min_unique and _looks_like_codes(valid):
            return "identifier"

        # Repetition, not label length or a fixed count, is what separates a category from free
        # text: 8 long event names, or 28 causes of default spread over 452 rows, are categories.
        if n_unique <= numeric_discrete_threshold:
            return "categorical"
        if n_unique / n_total > FREE_TEXT_MIN_UNIQUE_RATIO:
            return "text"
        return "categorical"

    # Default
    return "categorical"


def infer_all_types(
    df: pd.DataFrame,
    numeric_discrete_threshold: int = 20,
    config_overrides: dict[str, str] | None = None,
    identifier_min_unique: int = IDENTIFIER_MIN_UNIQUE,
) -> dict[str, SemanticType]:
    """
    Infer semantic types for all columns, with config overrides.

    Args:
        df: DataFrame
        numeric_discrete_threshold: Threshold for treating numeric as discrete
        config_overrides: Dict mapping column names to desired types
        identifier_min_unique: Distinct values above which a column of codes is an identifier;
            the pipeline passes `data_quality.cardinality_threshold` so that an ID is never
            flagged as high cardinality

    Returns:
        Dictionary mapping column name to inferred type
    """
    config_overrides = config_overrides or {}
    types = {}

    for col in df.columns:
        if col in config_overrides:
            types[col] = config_overrides[col]
        else:
            types[col] = infer_semantic_type(df[col], numeric_discrete_threshold, identifier_min_unique)

    return types


def get_numeric_columns(type_map: dict[str, SemanticType]) -> list[str]:
    """Get columns that are numeric (continuous or discrete)."""
    return [col for col, t in type_map.items() if "numeric" in t]


def get_categorical_columns(type_map: dict[str, SemanticType]) -> list[str]:
    """Get columns that are categorical or boolean."""
    return [col for col, t in type_map.items() if t in ("categorical", "boolean", "numeric_discrete")]


def get_time_columns(type_map: dict[str, SemanticType]) -> list[str]:
    """Get columns that hold a time of day without a date."""
    return [col for col, t in type_map.items() if t == "time"]


def get_text_columns(type_map: dict[str, SemanticType]) -> list[str]:
    """Get columns that are text (including identifiers)."""
    return [col for col, t in type_map.items() if t in ("text", "identifier")]


def get_datetime_columns(type_map: dict[str, SemanticType]) -> list[str]:
    """Get columns that are datetime."""
    return [col for col, t in type_map.items() if t == "datetime"]


@dataclass
class ColumnOverrideResult:
    """Outcome of applying user-specified `column_types` overrides."""

    column_types: dict[str, SemanticType] = field(default_factory=dict)
    coercion_failures: dict[str, int] = field(default_factory=dict)  # col -> values that failed to convert
    ignored_columns: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)


def apply_column_type_overrides(
    df: pd.DataFrame,
    column_types: dict[str, SemanticType],
    overrides: "ColumnTypeConfig",
) -> tuple[pd.DataFrame, ColumnOverrideResult]:
    """
    Apply user-specified column type overrides on top of inferred types.

    - ``numeric``/``datetime`` overrides coerce the underlying data
      (unconvertible values become NaN/NaT), reporting how many values failed.
    - ``categorical``/``text`` overrides only relabel the semantic type.
    - ``ignore`` drops the column from the DataFrame (and from
      ``column_types``) so no later analysis step sees it.
    - Column names listed in the config but absent from the DataFrame are
      collected in ``unknown_columns`` and logged as a warning.

    Args:
        df: Source DataFrame (not mutated; a copy is returned).
        column_types: Inferred semantic types (from ``infer_all_types``).
        overrides: Per-column type overrides from configuration.

    Returns:
        (new_df, ColumnOverrideResult)
    """
    df = df.copy()
    types = dict(column_types)
    result = ColumnOverrideResult(column_types=types)
    known_cols = set(df.columns)

    def _known(cols: list[str]) -> list[str]:
        unknown = [c for c in cols if c not in known_cols]
        result.unknown_columns.extend(unknown)
        return [c for c in cols if c in known_cols]

    for col in _known(overrides.numeric):
        before_na = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_failed = int(df[col].isna().sum()) - before_na
        if n_failed > 0:
            result.coercion_failures[col] = n_failed
            logger.warning(f"Column '{col}': {n_failed} value(s) could not be converted to a number (they stayed NaN).")
        types[col] = "numeric_continuous"

    for col in _known(overrides.datetime):
        before_na = int(df[col].isna().sum())
        df[col] = coerce_to_datetime(df[col])
        n_failed = int(df[col].isna().sum()) - before_na
        if n_failed > 0:
            result.coercion_failures[col] = n_failed
            logger.warning(
                f"Column '{col}': {n_failed} value(s) could not be converted to a date (they stayed NaT)."
            )
        types[col] = "datetime"

    for col in _known(overrides.categorical):
        types[col] = "categorical"

    for col in _known(overrides.text):
        types[col] = "text"

    ignore_cols = _known(overrides.ignore)
    if ignore_cols:
        df = df.drop(columns=ignore_cols)
        for col in ignore_cols:
            types.pop(col, None)
        result.ignored_columns = ignore_cols

    if result.unknown_columns:
        logger.warning(
            "Column(s) named in 'column_types' are not in the dataset: "
            + ", ".join(sorted(set(result.unknown_columns)))
        )

    result.column_types = types
    return df, result
