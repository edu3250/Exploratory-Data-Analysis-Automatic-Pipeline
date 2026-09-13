"""
CSV table exports: spreadsheet-friendly copies of the per-run analysis results.
"""

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .type_inference import dtype_label, semantic_type_label

logger = logging.getLogger(__name__)


def _records_from_stats(stats_dict: dict[str, Any], id_col: str) -> list[dict]:
    """Turn a ``{column_name: dataclass_instance}`` mapping into flat row dicts."""
    records = []
    for name, stats in stats_dict.items():
        record = {id_col: name}
        record.update(asdict(stats) if is_dataclass(stats) else dict(stats))
        records.append(record)
    return records


def _write_csv(tables_dir: Path, name: str, df: pd.DataFrame) -> str:
    path = tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def write_result_tables(
    output_dir: Path,
    numeric_stats: dict,
    categorical_stats: dict,
    missing_per_column: dict,
    numeric_pairs: list,
    categorical_pairs: list,
    mixed_pairs: list,
    outliers_report,
    alerts: list,
    column_types: dict[str, str] | None = None,
    column_dtypes: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Write CSV tables summarizing one dataset's analysis.

    Args:
        output_dir: The dataset's report directory (a ``tables/`` subfolder is created).
        numeric_stats: ``{column: NumericStats}`` from univariate analysis.
        categorical_stats: ``{column: CategoricalStats}`` from univariate analysis.
        missing_per_column: ``{column: pct_missing}`` from data quality analysis.
        numeric_pairs, categorical_pairs, mixed_pairs: ``CorrelationPair`` lists from relationships analysis.
        outliers_report: ``OutlierReport`` from outlier detection.
        alerts: Combined list of ``Alert`` (data quality + target-derived).
        column_types: Inferred semantic type per column, written next to its missing percentage.
        column_dtypes: Storage dtype per column, written next to the inferred type.

    Returns:
        Dict mapping table name to the written CSV file path.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    written = {
        "numeric_stats": _write_csv(
            tables_dir, "numeric_stats", pd.DataFrame(_records_from_stats(numeric_stats, "columna"))
        ),
        "categorical_stats": _write_csv(
            tables_dir, "categorical_stats", pd.DataFrame(_records_from_stats(categorical_stats, "columna"))
        ),
        "missing_per_column": _write_csv(
            tables_dir,
            "missing_per_column",
            pd.DataFrame(
                [
                    {
                        "columna": col,
                        "tipo_dato": dtype_label((column_dtypes or {}).get(col)),
                        "categoria_inferida": semantic_type_label((column_types or {}).get(col)),
                        "pct_faltante": pct,
                    }
                    for col, pct in missing_per_column.items()
                ]
            ),
        ),
        "correlations": _write_csv(
            tables_dir,
            "correlations",
            pd.DataFrame([asdict(pair) for pair in (numeric_pairs + categorical_pairs + mixed_pairs)]),
        ),
        "outlier_summary": _write_csv(tables_dir, "outlier_summary", _outlier_summary_table(outliers_report)),
        "alerts": _write_csv(tables_dir, "alerts", pd.DataFrame([asdict(alert) for alert in alerts])),
    }

    logger.info(f"Tablas CSV escritas en: {tables_dir}")
    return written


def _outlier_summary_table(outliers_report) -> pd.DataFrame:
    """Flatten IQR/MAD per-column results and the multivariate result into one summary table."""
    rows = []
    for method_name, info_map in (
        ("iqr", outliers_report.iqr_outliers),
        ("mad_zscore", outliers_report.mad_outliers),
    ):
        for col, info in info_map.items():
            rows.append({"columna": col, "metodo": method_name, "n_outliers": info.n_outliers, "nota": info.note})

    if outliers_report.multivariate_outliers or outliers_report.multivariate_note:
        # All entries share the same multivariate outlier set; one summary row is enough.
        flagged = outliers_report.multivariate_outliers
        rows.append(
            {
                "columna": "(multivariado)",
                "metodo": "isolation_forest",
                "n_outliers": flagged[0].n_outliers if flagged else 0,
                "nota": outliers_report.multivariate_note,
            }
        )

    return pd.DataFrame(rows)
