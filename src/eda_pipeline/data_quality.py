"""
Data quality analysis: missing values, duplicates, constants, cardinality, type conflicts.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Shared severity ordering, reused wherever alerts from multiple sources are merged and sorted.
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Alert:
    """Data quality alert."""

    severity: str  # 'critical', 'high', 'medium', 'low'
    column: Optional[str]
    message: str
    recommendation: Optional[str] = None


@dataclass
class DataQualityReport:
    """Data quality analysis results."""

    shape: tuple[int, int]
    missing_per_column: dict[str, float]  # percentage
    missing_per_row: dict[int, int]  # row_index: count of missing values
    duplicates: int
    duplicate_rows: pd.DataFrame
    constant_columns: list[str]
    quasi_constant_columns: dict[str, float]  # col: percentage of most common value
    high_cardinality_columns: dict[str, int]  # col: n_unique
    mixed_type_columns: list[str]
    numeric_as_text: dict[str, int]  # col: count of numeric-looking text values
    alerts: list[Alert] = field(default_factory=list)
    recommendations: dict[str, list[str]] = field(default_factory=dict)


def analyze_missing(
    df: pd.DataFrame, missing_threshold: float = 0.95
) -> tuple[dict[str, float], dict[int, int], list[Alert]]:
    """
    Analyze missing values.

    Returns:
        (missing_per_column, missing_per_row, alerts)
    """
    alerts = []

    # Missing per column
    missing_per_column = (df.isnull().sum() / len(df) * 100).to_dict()

    # Columns with excessive missing
    for col, pct in missing_per_column.items():
        if pct > missing_threshold * 100:
            alerts.append(
                Alert(
                    severity="high",
                    column=col,
                    message=f"Columna '{col}': {pct:.1f}% valores faltantes",
                    recommendation="Considere eliminar esta columna o usar imputación avanzada",
                )
            )
        elif pct > 50:
            alerts.append(
                Alert(
                    severity="medium",
                    column=col,
                    message=f"Columna '{col}': {pct:.1f}% valores faltantes",
                    recommendation="Evalúe si el patrón de falta es informativo o aleatorio",
                )
            )

    # Missing per row
    missing_per_row = df.isnull().sum(axis=1).to_dict()
    rows_with_all_missing = {i: count for i, count in missing_per_row.items() if count == len(df.columns)}
    if rows_with_all_missing:
        alerts.append(
            Alert(
                severity="high",
                column=None,
                message=f"{len(rows_with_all_missing)} fila(s) completamente vacía(s)",
                recommendation="Elimine estas filas antes del análisis",
            )
        )

    return missing_per_column, missing_per_row, alerts


def _sort_duplicate_rows(duplicate_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Sort duplicate rows for readable side-by-side comparison.

    ``sort_values`` can raise ``TypeError`` on object columns holding mixed
    types (e.g. ints and strings), which Python cannot compare with ``<``. In
    that case, fall back to sorting by the string representation of each
    column (only for ordering purposes; the returned values are unchanged).
    """
    if duplicate_rows.empty:
        return duplicate_rows

    columns = list(duplicate_rows.columns)
    try:
        return duplicate_rows.sort_values(by=columns)
    except TypeError:
        try:
            return duplicate_rows.sort_values(by=columns, key=lambda col: col.astype(str))
        except Exception as e:
            logger.warning(f"No se pudieron ordenar las filas duplicadas ({e}); se devuelven sin ordenar.")
            return duplicate_rows


def analyze_duplicates(df: pd.DataFrame, duplicate_threshold: float = 0.1) -> tuple[int, pd.DataFrame, list[Alert]]:
    """
    Analyze duplicate rows.

    Args:
        df: DataFrame to analyze
        duplicate_threshold: Fraction of duplicated rows (0-1) above which the
            alert severity is raised to 'high' (half that fraction -> 'medium').

    Returns:
        (n_duplicates, duplicate_rows_df, alerts)
    """
    alerts = []

    # Exact duplicates
    n_duplicates = df.duplicated().sum()
    duplicate_rows = _sort_duplicate_rows(df[df.duplicated(keep=False)])

    if n_duplicates > 0:
        dup_ratio = n_duplicates / len(df)
        if dup_ratio > duplicate_threshold:
            severity = "high"
        elif dup_ratio > duplicate_threshold / 2:
            severity = "medium"
        else:
            severity = "low"
        alerts.append(
            Alert(
                severity=severity,
                column=None,
                message=f"{n_duplicates} filas duplicadas exactas ({dup_ratio * 100:.2f}%)",
                recommendation="Investigue si los duplicados son legítimos o errores de entrada",
            )
        )

    return n_duplicates, duplicate_rows, alerts


def analyze_constants(
    df: pd.DataFrame, constant_threshold: float = 0.99
) -> tuple[list[str], dict[str, float], list[Alert]]:
    """
    Identify constant and quasi-constant columns.

    Returns:
        (constant_columns, quasi_constant_columns, alerts)
    """
    alerts = []
    constant_cols = []
    quasi_constant_cols = {}

    for col in df.columns:
        # Get most common value percentage
        value_counts = df[col].value_counts(dropna=False)
        if len(value_counts) == 0:
            constant_cols.append(col)
        elif len(value_counts) == 1:
            constant_cols.append(col)
        else:
            most_common_pct = value_counts.iloc[0] / len(df)
            if most_common_pct > constant_threshold:
                quasi_constant_cols[col] = most_common_pct * 100

    if constant_cols:
        alerts.append(
            Alert(
                severity="high",
                column=None,
                message=f"{len(constant_cols)} columna(s) con un único valor: {', '.join(constant_cols)}",
                recommendation="Elimine estas columnas; no tienen varianza",
            )
        )

    for col, pct in quasi_constant_cols.items():
        alerts.append(
            Alert(
                severity="medium",
                column=col,
                message=f"Columna casi constante: {pct:.1f}% un único valor",
                recommendation="Esta variable tiene poca capacidad predictiva",
            )
        )

    return constant_cols, quasi_constant_cols, alerts


# Semantic types for which many distinct values are expected (amounts, balances, dates, IDs), not
# a data-quality problem. An identifier holds one value per entity, so many distinct values is
# what an ID looks like rather than a finding worth reporting.
CARDINALITY_EXEMPT_TYPES = {
    "numeric_continuous",
    "numeric_discrete",
    "datetime",
    "boolean",
    "constant",
    "identifier",
}


def analyze_cardinality(
    df: pd.DataFrame, cardinality_threshold: int = 100, column_types: Optional[dict[str, str]] = None
) -> tuple[dict[str, int], list[Alert]]:
    """
    Identify high-cardinality columns among categorical and free-text columns.

    Numbers, dates and identifiers are skipped: thousands of distinct amounts, timestamps or
    customer codes are normal. The inferred semantic types are used when given (so dates stored
    as text, and IDs, are skipped too), otherwise the column dtypes.

    Returns:
        (high_cardinality_columns, alerts)
    """
    alerts = []
    high_cardinality = {}

    for col in df.columns:
        if column_types is not None and col in column_types:
            if column_types[col] in CARDINALITY_EXEMPT_TYPES:
                continue
        elif pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n_unique = df[col].nunique()
        if n_unique > cardinality_threshold:
            high_cardinality[col] = n_unique

    for col, n_unique in sorted(high_cardinality.items(), key=lambda x: -x[1])[:10]:
        alerts.append(
            Alert(
                severity="low",
                column=col,
                message=f"Columna '{col}': {n_unique} valores únicos (alta cardinalidad)",
                recommendation="Considere agrupar valores o usar hashing si esta es categórica",
            )
        )

    return high_cardinality, alerts


def detect_mixed_types(df: pd.DataFrame) -> tuple[list[str], list[Alert]]:
    """
    Detect columns with mixed data types.

    Returns:
        (mixed_type_columns, alerts)
    """
    alerts = []
    mixed_cols = []

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Sample to check for type mixing
            non_null = df[col].dropna()
            if len(non_null) > 0:
                types_in_col = set()
                for val in non_null.iloc[:100]:  # Sample first 100 non-null values
                    if isinstance(val, str):
                        types_in_col.add("string")
                    elif isinstance(val, (int, np.integer)):
                        types_in_col.add("int")
                    elif isinstance(val, (float, np.floating)):
                        types_in_col.add("float")
                    elif isinstance(val, bool):
                        types_in_col.add("bool")
                    else:
                        types_in_col.add(type(val).__name__)

                if len(types_in_col) > 1:
                    mixed_cols.append(col)
                    alerts.append(
                        Alert(
                            severity="medium",
                            column=col,
                            message=f"Columna '{col}' contiene tipos mixtos: {', '.join(sorted(types_in_col))}",
                            recommendation="Estandarice los tipos o investigue la causa",
                        )
                    )

    return mixed_cols, alerts


def detect_numeric_as_text(df: pd.DataFrame) -> tuple[dict[str, int], list[Alert]]:
    """
    Detect numeric values stored as text.

    Returns:
        (numeric_as_text_columns, alerts)
    """
    alerts = []
    numeric_as_text = {}

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Count numeric-looking strings
                numeric_count = 0
                for val in non_null:
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass

                if numeric_count > 0.8 * len(non_null):
                    numeric_as_text[col] = numeric_count
                    alerts.append(
                        Alert(
                            severity="medium",
                            column=col,
                            message=f"Columna '{col}': {numeric_count}/{len(non_null)} valores se ven numéricos pero están como texto",
                            recommendation="Convierta a tipo numérico para análisis correcto",
                        )
                    )

    return numeric_as_text, alerts


def analyze_data_quality(
    df: pd.DataFrame,
    missing_threshold: float = 0.95,
    cardinality_threshold: int = 100,
    constant_threshold: float = 0.99,
    duplicate_threshold: float = 0.1,
    column_types: Optional[dict[str, str]] = None,
) -> DataQualityReport:
    """
    Perform comprehensive data quality analysis.

    Args:
        column_types: Inferred semantic types, so the cardinality check can skip numbers and
            dates even when they are stored as text.

    Returns:
        DataQualityReport with all metrics and alerts
    """
    # Run all analyses
    missing_per_col, missing_per_row, missing_alerts = analyze_missing(df, missing_threshold)
    n_duplicates, dup_rows, dup_alerts = analyze_duplicates(df, duplicate_threshold)
    const_cols, quasi_const_cols, const_alerts = analyze_constants(df, constant_threshold)
    high_card_cols, card_alerts = analyze_cardinality(df, cardinality_threshold, column_types)
    mixed_cols, mixed_alerts = detect_mixed_types(df)
    numeric_text, numeric_alerts = detect_numeric_as_text(df)

    # Combine all alerts, sorted by severity
    all_alerts = missing_alerts + dup_alerts + const_alerts + card_alerts + mixed_alerts + numeric_alerts
    all_alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 999), a.column or ""))

    return DataQualityReport(
        shape=df.shape,
        missing_per_column=missing_per_col,
        missing_per_row=missing_per_row,
        duplicates=n_duplicates,
        duplicate_rows=dup_rows,
        constant_columns=const_cols,
        quasi_constant_columns=quasi_const_cols,
        high_cardinality_columns=high_card_cols,
        mixed_type_columns=mixed_cols,
        numeric_as_text=numeric_text,
        alerts=all_alerts,
    )
