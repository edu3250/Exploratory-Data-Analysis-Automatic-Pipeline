"""
Data loading with automatic format, delimiter, encoding, and decimal detection.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from chardet import detect as chardet_detect

logger = logging.getLogger(__name__)

# Extensions recognized by detect_file_format / discover_batch_files.
SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl")

_DELIMITER_CANDIDATES = ",;\t|"

# A cell holding nothing but punctuation marks a missing value, not a category: penguins sex has one
# "." among MALE and FEMALE, which the report counted as a third sex. pandas already reads "", "NA",
# "N/A", "null" and similar as missing; these are the ones it leaves as text. Across the 19 datasets
# in data/raw they occur only in the two penguin sex columns.
_MISSING_PLACEHOLDER_PATTERN = re.compile(r"^\s*[.\-?_*/]+\s*$")


def replace_missing_placeholders(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Turn cells that hold only punctuation (".", "-", "?", "--") into missing values.

    A text column whose remaining values are all numbers becomes numeric: a "?" in a column of
    weights is what made pandas read every weight as text in the first place.

    Returns:
        (the frame, changed only in the affected columns; {column: {placeholder: count}} of what
        was replaced, empty when nothing was)
    """
    found: dict[str, dict[str, int]] = {}
    result = df
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_string_dtype(series):
            continue
        text = series.dropna().astype(str)
        placeholders = text[text.str.match(_MISSING_PLACEHOLDER_PATTERN)]
        if placeholders.empty:
            continue

        found[str(col)] = {str(k): int(v) for k, v in placeholders.str.strip().value_counts().items()}
        cleaned = series.mask(series.index.isin(placeholders.index))
        numbers = pd.to_numeric(cleaned, errors="coerce")
        if cleaned.notna().any() and numbers.notna().sum() == cleaned.notna().sum():
            cleaned = numbers.convert_dtypes(dtype_backend="numpy_nullable")
        if result is df:
            result = df.copy()  # the caller's frame is never modified
        result[col] = cleaned
    return result, found


def detect_encoding(file_path: Path, chardet_sample_size: int = 200_000, min_confidence: float = 0.5) -> str:
    """
    Detect a file's text encoding.

    Strategy, in order:
        1. Try to decode the *entire* file as strict UTF-8. This is the most
           common case and avoids false positives from short/ambiguous
           samples (a small sample may simply not contain the byte that would
           give away a non-UTF-8 encoding).
        2. Run chardet on a larger sample; trust it only above a confidence
           threshold.
        3. Fall back to cp1252 (a superset of latin-1, and the common case for
           Windows-authored files with accented characters).
    """
    raw = Path(file_path).read_bytes()

    try:
        raw.decode("utf-8")
        logger.debug("Detected encoding: utf-8 (strict decode of full file)")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    detection = chardet_detect(raw[:chardet_sample_size])
    encoding = detection.get("encoding")
    confidence = detection.get("confidence") or 0

    if encoding and confidence >= min_confidence:
        logger.debug(f"Detected encoding: {encoding} (chardet, confidence={confidence:.2f})")
        return encoding

    logger.debug(f"chardet inconclusive (encoding={encoding}, confidence={confidence:.2f}); falling back to cp1252")
    return "cp1252"


def _is_consistent(lines: list[str], delimiter: str) -> bool:
    """A delimiter is trustworthy if it appears the same positive number of times on every line."""
    counts = [line.count(delimiter) for line in lines]
    return counts[0] > 0 and len(set(counts)) == 1


def detect_delimiter(file_path: Path, encoding: str, sample_lines: int = 20) -> str:
    """
    Detect the CSV field delimiter.

    Uses ``csv.Sniffer`` restricted to a small set of plausible delimiters
    (``, ; \\t |``  — deliberately excluding space, which false-triggers on
    free-text columns) and cross-checks the guess with a per-line field-count
    consistency check. Falls back to ',' whenever detection is inconclusive
    (e.g. text-heavy columns, or files where numeric fields use a comma as
    decimal separator).
    """
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            raw_lines = [f.readline() for _ in range(sample_lines)]
        lines = [line for line in raw_lines if line.strip()]

        if len(lines) < 2:
            return ","

        sample = "".join(lines)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=_DELIMITER_CANDIDATES)
            candidate = dialect.delimiter
        except csv.Error:
            candidate = None

        if candidate and _is_consistent(lines, candidate):
            logger.debug(f"Detected delimiter: '{candidate}' (csv.Sniffer)")
            return candidate

        # Sniffer was inconclusive or failed its own consistency check: try
        # every candidate directly and keep the most frequent consistent one.
        best_delimiter, best_score = None, 0
        for delim in _DELIMITER_CANDIDATES:
            if _is_consistent(lines, delim):
                score = lines[0].count(delim)
                if score > best_score:
                    best_delimiter, best_score = delim, score

        result = best_delimiter or ","
        logger.debug(f"Detected delimiter: '{result}' (consistency fallback)")
        return result
    except Exception as e:
        logger.warning(f"Failed to detect delimiter: {e}. Using comma.")
        return ","


_COMMA_DECIMAL_RE = re.compile(r"^-?\d+,\d+$")
_DOT_DECIMAL_RE = re.compile(r"^-?\d+\.\d+$")


def detect_decimal_separator(file_path: Path, encoding: str, delimiter: str, sample_lines: int = 30) -> str:
    """
    Auto-detect the decimal separator ('.' or ',') for a CSV file.

    Only considers ',' when the field delimiter is itself not a comma
    (otherwise the two would be ambiguous). A field counts as comma-decimal
    when it matches ``\\d+,\\d+`` (and analogously for '.').
    """
    if delimiter == ",":
        return "."

    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            lines = [f.readline() for _ in range(sample_lines + 1)]
    except OSError:
        return "."

    comma_hits = 0
    dot_hits = 0
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        for field_value in line.strip().split(delimiter):
            field_value = field_value.strip().strip('"')
            if _COMMA_DECIMAL_RE.match(field_value):
                comma_hits += 1
            elif _DOT_DECIMAL_RE.match(field_value):
                dot_hits += 1

    decimal = "," if comma_hits > 0 and comma_hits >= dot_hits else "."
    logger.debug(f"Detected decimal separator: '{decimal}' (comma_hits={comma_hits}, dot_hits={dot_hits})")
    return decimal


def _apply_sample(df: pd.DataFrame, sample_size: Optional[int]) -> pd.DataFrame:
    """Randomly (but deterministically) sample rows, logging the true original size."""
    if sample_size is not None and len(df) > sample_size:
        original_len = len(df)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} rows from {original_len} (deterministic)")
    return df


def _stringify_nested_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serialize any list/dict cell to a JSON string.

    Downstream analysis assumes hashable scalar values (e.g. ``value_counts``,
    ``duplicated``); a stray list/dict cell (most commonly from nested JSON)
    would otherwise crash steps far removed from the original load.
    """
    for col in df.columns:
        has_nested = df[col].map(lambda v: isinstance(v, (list, dict))).any()
        if has_nested:
            logger.info(f"Columna '{col}' contiene listas/diccionarios anidados; se serializan como texto JSON.")
            df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df


def load_csv(
    file_path: Path,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    decimal: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load CSV/TSV with automatic detection of encoding, delimiter, and decimal separator."""
    if encoding is None:
        encoding = detect_encoding(file_path)
    if delimiter is None:
        delimiter = detect_delimiter(file_path, encoding)
    if decimal is None:
        decimal = detect_decimal_separator(file_path, encoding, delimiter)

    logger.info(f"Loading CSV: {file_path} (encoding={encoding}, delimiter='{delimiter}', decimal='{decimal}')")

    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, decimal=decimal, dtype_backend="numpy_nullable")

    return _apply_sample(df, sample_size)


def _resolve_sheet(sheet: Optional[str | int]) -> str | int:
    """Default to the first sheet (index 0) when no sheet was requested."""
    return 0 if sheet is None else sheet


def load_excel(file_path: Path, sheet: Optional[str | int] = None, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load an Excel file.

    ``sheet=None`` (the default) loads the first sheet. Passing ``sheet_name``
    as ``None`` straight to pandas would return a dict of *all* sheets, which
    downstream code cannot handle; ``_resolve_sheet`` avoids that entirely.
    """
    resolved_sheet = _resolve_sheet(sheet)
    logger.info(f"Loading Excel: {file_path} (sheet={resolved_sheet})")

    try:
        df = pd.read_excel(file_path, sheet_name=resolved_sheet, dtype_backend="numpy_nullable")
    except Exception as e:
        logger.error(f"Failed to load Excel: {e}")
        raise

    return _apply_sample(df, sample_size)


def list_excel_sheets(file_path: Path) -> list[str]:
    """List all sheet names in an Excel file."""
    try:
        # pd.ExcelFile keeps the underlying file open for later .parse() calls;
        # closing it explicitly (rather than leaking it until garbage collection)
        # avoids "file in use" errors when the caller tries to delete it soon after,
        # which is common on Windows.
        with pd.ExcelFile(file_path) as excel_file:
            return excel_file.sheet_names
    except Exception as e:
        logger.error(f"Failed to read Excel sheets: {e}")
        raise


def load_parquet(file_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load Parquet file."""
    logger.info(f"Loading Parquet: {file_path}")

    df = pd.read_parquet(file_path)

    return _apply_sample(df, sample_size)


def load_json(file_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load a JSON or JSON Lines file.

    The choice of parser is driven by content/extension rather than
    trial-and-error:
        - ``.jsonl``: always parsed as JSON Lines (one record per line).
        - ``.json``: parsed as a regular JSON document first (a single
          object, or an array of — possibly nested — records), falling back
          to JSON Lines only if that fails. Arrays of records are flattened
          with ``pd.json_normalize`` so nested dicts become dotted columns
          (e.g. ``address.city``).

    A single-line JSON *array* previously "succeeded" under
    ``pd.read_json(..., lines=True)`` but produced a frame whose cells were
    raw dicts, crashing later steps with ``TypeError: unhashable type:
    'dict'``. Choosing the parser by content avoids that entirely.
    """
    logger.info(f"Loading JSON: {file_path}")
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".jsonl":
        df = pd.read_json(file_path, lines=True, dtype=False)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw if isinstance(raw, list) else [raw]
            df = pd.json_normalize(records)
        except Exception as e:
            logger.warning(f"No se pudo interpretar {file_path.name} como JSON regular ({e}); probando JSON Lines.")
            try:
                df = pd.read_json(file_path, lines=True, dtype=False)
            except Exception as e2:
                logger.error(f"Failed to load JSON: {e2}")
                raise

    return _apply_sample(df, sample_size)


def detect_file_format(file_path: Path) -> str:
    """Detect file format from extension."""
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return "csv"
    elif suffix in (".xlsx", ".xls"):
        return "excel"
    elif suffix == ".parquet":
        return "parquet"
    elif suffix in (".json", ".jsonl"):
        return "json"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_data(
    file_path: Path,
    file_format: Optional[str] = None,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    decimal: Optional[str] = None,
    excel_sheet: Optional[str | int] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load data from various formats with automatic detection.

    Args:
        file_path: Path to data file
        file_format: 'csv', 'excel', 'parquet', 'json' (auto-detect if None)
        encoding: Text encoding (auto-detect if None)
        delimiter: Delimiter for CSV (auto-detect if None)
        decimal: Decimal separator, '.' or ',' (auto-detect if None)
        excel_sheet: Sheet name or index for Excel (first sheet if None)
        sample_size: Sample this many rows (for large datasets)

    Returns:
        pandas DataFrame. Any cell holding a nested list/dict (possible with
        JSON and Parquet sources) is serialized to a JSON string so no later
        analysis step chokes on an unhashable value.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_format is None:
        file_format = detect_file_format(file_path)
        logger.info(f"Auto-detected format: {file_format}")

    if file_format == "csv":
        df = load_csv(file_path, encoding, delimiter, decimal, sample_size)
    elif file_format == "excel":
        df = load_excel(file_path, excel_sheet, sample_size)
    elif file_format == "parquet":
        df = load_parquet(file_path, sample_size)
    elif file_format == "json":
        df = load_json(file_path, sample_size)
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    return _stringify_nested_cells(df)


def discover_batch_files(folder_path: Path, pattern: Optional[str] = None) -> list[Path]:
    """
    Discover data files in a folder for batch processing.

    Args:
        folder_path: Folder to scan.
        pattern: Glob pattern (e.g. "*.csv"). If None, all files with a
            supported extension are considered (see SUPPORTED_EXTENSIONS).

    Returns:
        Sorted list of file paths, excluding hidden files (dotfiles, e.g.
        ``.gitkeep``) and files with an unsupported extension. Both are
        logged at INFO level rather than raising, so a single stray file
        never aborts a batch run.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    if pattern:
        candidates = sorted(set(folder_path.glob(pattern)))
    else:
        candidates = sorted({p for ext in SUPPORTED_EXTENSIONS for p in folder_path.glob(f"*{ext}")})

    files = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.name.startswith("."):
            logger.info(f"Omitiendo archivo oculto: {path.name}")
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.info(f"Omitiendo archivo con extensión no soportada: {path.name}")
            continue
        files.append(path)

    return sorted(files)


def load_batch(folder_path: Path, file_pattern: Optional[str] = None, **kwargs) -> dict[str, pd.DataFrame]:
    """
    Load multiple files from a folder into memory.

    Kept for convenience/backwards compatibility; ``EDAPipeline`` processes
    batches one file at a time (see ``pipeline.py``) instead of using this,
    so a large batch does not need every file resident in memory at once.

    Args:
        folder_path: Folder containing data files
        file_pattern: Glob pattern for files to load (default: all supported extensions)
        **kwargs: Additional arguments to pass to load_data

    Returns:
        Dictionary mapping dataset name to DataFrame. A file that fails to
        load is skipped (logged as an error) rather than aborting the batch.
    """
    files = discover_batch_files(folder_path, file_pattern)
    logger.info(f"Loading {len(files)} files from {folder_path}")

    results: dict[str, pd.DataFrame] = {}
    for file_path in files:
        try:
            df = load_data(file_path, **kwargs)
            dataset_name = file_path.stem
            results[dataset_name] = df
            logger.info(f"  Loaded {dataset_name}: {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"  Failed to load {file_path.name}: {e}")

    return results
