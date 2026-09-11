"""
Main EDA Pipeline Orchestrator.
"""

import difflib
import json
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config
from .data_loader import discover_batch_files, load_data
from .data_quality import SEVERITY_ORDER, Alert, DataQualityReport, analyze_data_quality
from .html_report import generate_html_report
from .logging_util import generate_correlation_id, setup_logging
from .outlier_detection import OutlierReport, analyze_outliers
from .relationships import RelationshipsReport, analyze_relationships
from .tables import write_result_tables
from .target_analysis import analyze_target
from .type_inference import (
    apply_column_type_overrides,
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    get_text_columns,
    infer_all_types,
)
from .univariate_analysis import UnivariateReport, analyze_univariate
from .visualizations import generate_all_visualizations


class TargetColumnNotFoundError(ValueError):
    """Raised when the configured target column is missing (single-file mode fails fast on this)."""


def _missing_target_message(target_column: str, available_columns: list[str]) -> str:
    """Build a Spanish error message for a missing target column, with fuzzy-match suggestions."""
    suggestions = difflib.get_close_matches(target_column, available_columns, n=3)
    message = (
        f"La columna target '{target_column}' no existe en el dataset. "
        f"Columnas disponibles: {', '.join(available_columns)}."
    )
    if suggestions:
        message += f" ¿Quiso decir: {', '.join(suggestions)}?"
    return message


def _build_target_alerts(target_report) -> list[Alert]:
    """Turn class-imbalance and leakage findings into Alerts for the main alerts list."""
    alerts: list[Alert] = []
    cb = target_report.class_balance
    if cb and cb.is_imbalanced:
        majority_pct = max(cb.class_proportions.values()) if cb.class_proportions else 0.0
        alerts.append(
            Alert(
                severity="high",
                column=target_report.target_column,
                message=(
                    f"Desbalance de clases en target '{target_report.target_column}': "
                    f"la clase mayoritaria representa {majority_pct:.1f}% de las muestras."
                ),
                recommendation="Considere resampling (SMOTE, undersampling) o métricas robustas al desbalance (F1, AUC-PR).",
            )
        )
    for leak_message in target_report.leakage_alerts:
        alerts.append(
            Alert(
                severity="high",
                column=target_report.target_column,
                message=leak_message,
                recommendation="Revise si esta variable debería excluirse del modelo por fuga de información.",
            )
        )
    return alerts


def _empty_data_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """Fallback used when the data_quality step itself fails, so later steps still run."""
    return DataQualityReport(
        shape=df.shape,
        missing_per_column={},
        missing_per_row={},
        duplicates=0,
        duplicate_rows=pd.DataFrame(),
        constant_columns=[],
        quasi_constant_columns={},
        high_cardinality_columns={},
        mixed_type_columns=[],
        numeric_as_text={},
        alerts=[],
    )


def _empty_relationships_report() -> RelationshipsReport:
    """Fallback used when the relationships step itself fails."""
    return RelationshipsReport(
        numeric_pairs=[], categorical_pairs=[], mixed_pairs=[], correlation_matrix=pd.DataFrame(), vif={}
    )


def _empty_outlier_report() -> OutlierReport:
    """Fallback used when the outliers step itself fails."""
    return OutlierReport(iqr_outliers={}, mad_outliers={}, multivariate_outliers=[], outlier_indices_union=set())


class EDAPipeline:
    """Main EDA Pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.correlation_id = generate_correlation_id()
        self.log_file = None
        self.results = {}

    def setup(self):
        """Initialize logging and output directories."""
        log_dir = Path(self.config.output_dir) / ".." / "logs"
        self.logger, self.log_file = setup_logging(log_dir, self.correlation_id, verbose=self.config.verbose)
        self.logger.info(f"EDA Pipeline started (correlation_id={self.correlation_id})")

    def run(self) -> dict:
        """
        Run the complete EDA pipeline (single file or batch folder).

        Returns:
            Dictionary mapping dataset name to its result dict. A dataset
            entry is either ``{'error': str, 'success': False}`` (loading or
            analysis failed outright) or a success dict that also carries a
            ``failed_steps`` list (possibly empty) for partial failures.
        """
        try:
            try:
                if self.config.input_file:
                    file_path = Path(self.config.input_file)
                    dataset_name = file_path.stem
                    self.logger.info(f"Loading data from: {file_path}")
                    all_results = {dataset_name: self._load_and_analyze(file_path, dataset_name)}
                else:
                    all_results = self._run_batch()

                self.results = all_results
                self.logger.info(f"Pipeline completed. Processed {len(all_results)} dataset(s)")
                return all_results

            except Exception as e:
                self.logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
                if self.config.strict_mode:
                    raise
                return {"error": str(e)}
        finally:
            # Release the log file handle now rather than waiting for the next
            # setup_logging() call. Otherwise the file stays open (locked, on
            # Windows) until some later run happens to replace it, which can
            # make an output directory undeletable in the meantime.
            self._close_logging()

    def _close_logging(self) -> None:
        """Close and detach this run's log handlers from the shared 'eda_pipeline' logger."""
        if not self.logger:
            return
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def _run_batch(self) -> dict:
        """Discover files in the input folder and process them one at a time (not preloaded)."""
        folder_path = Path(self.config.input_folder)
        files = discover_batch_files(folder_path, self.config.batch_pattern)
        self.logger.info(f"Se encontraron {len(files)} archivo(s) para procesar en {folder_path}")

        stem_counts = Counter(f.stem for f in files)
        all_results = {}
        for file_path in files:
            dataset_name = self._unique_dataset_name(file_path, stem_counts)
            all_results[dataset_name] = self._load_and_analyze(file_path, dataset_name)

        return all_results

    @staticmethod
    def _unique_dataset_name(file_path: Path, stem_counts: Counter) -> str:
        """Disambiguate same-stem files (e.g. sales.csv / sales.parquet) with a suffix hint."""
        stem = file_path.stem
        if stem_counts[stem] > 1:
            return f"{stem}_{file_path.suffix.lstrip('.').lower()}"
        return stem

    def _load_and_analyze(self, file_path: Path, dataset_name: str) -> dict:
        """Load a single file and, if that succeeds, run the full per-dataset analysis."""
        try:
            df = load_data(
                file_path,
                file_format=self.config.file_format,
                encoding=self.config.encoding,
                delimiter=self.config.delimiter,
                decimal=self.config.decimal,
                excel_sheet=self.config.excel_sheet,
                sample_size=self.config.sample_size,
            )
        except Exception as e:
            self.logger.error(f"No se pudo cargar {file_path.name}: {e}\n{traceback.format_exc()}")
            if self.config.strict_mode:
                raise
            return {"error": str(e), "success": False}

        try:
            # df.shape and the analysis itself are both inside this dataset's error
            # handling (previously df.shape was logged outside the try, so a loader
            # returning something unexpected, e.g. the old Excel dict-of-sheets bug,
            # would escape to the outer handler and abort the whole run).
            self.logger.info(f"Analyzing {dataset_name}: {df.shape[0]} rows × {df.shape[1]} columns")
            return self._analyze_dataset(dataset_name, df)
        except TargetColumnNotFoundError as e:
            # A mistyped --target is a user error: show the message, not a traceback.
            if self.config.strict_mode:
                raise
            self.logger.error(str(e))
            return {"error": str(e), "success": False}
        except Exception as e:
            if self.config.strict_mode:
                raise
            self.logger.error(f"Error analyzing {dataset_name}: {e}\n{traceback.format_exc()}")
            return {"error": str(e), "success": False}

    def _run_step(self, step_name: str, step_failures: list[str], func, *args, default=None, **kwargs):
        """
        Run one analysis step in isolation.

        On failure, the traceback is logged (every log line already carries
        this run's correlation id), ``step_name`` is recorded in
        ``step_failures``, and ``default`` is returned so later steps can
        still run. In strict mode, the exception is re-raised immediately.

        Note: the tracking list parameter is named ``step_failures`` (not
        ``failed_steps``) so it never collides with a same-named keyword
        argument intended for ``func`` (e.g. ``generate_html_report`` also
        takes a ``failed_steps=`` argument).
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if self.config.strict_mode:
                raise
            self.logger.error(
                f"Paso '{step_name}' falló (correlation_id={self.correlation_id}): {e}\n{traceback.format_exc()}"
            )
            step_failures.append(step_name)
            return default

    def _analyze_dataset(self, dataset_name: str, df: pd.DataFrame) -> dict:
        """Analyze a single dataset, isolating failures per analysis step (see _run_step)."""
        failed_steps: list[str] = []

        # Type inference + user overrides. Not step-isolated: every later step depends on it,
        # so a failure here should surface as a dataset-level error (via _load_and_analyze).
        self.logger.info("Inferring column types...")
        column_types = infer_all_types(df)
        df, override_result = apply_column_type_overrides(df, column_types, self.config.column_types)
        column_types = override_result.column_types

        numeric_cols = get_numeric_columns(column_types)
        categorical_cols = get_categorical_columns(column_types)
        text_cols = get_text_columns(column_types)
        datetime_cols = get_datetime_columns(column_types)

        self.logger.info(
            f"  Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, "
            f"Text: {len(text_cols)}, Datetime: {len(datetime_cols)}"
        )

        # Target column existence: single-file mode fails fast; batch mode warns and continues.
        target_missing_alert = None
        target_column = self.config.target.target_column
        if target_column and target_column not in df.columns:
            message = _missing_target_message(target_column, list(df.columns))
            if self.config.input_file:
                raise TargetColumnNotFoundError(message)
            self.logger.warning(message)
            target_missing_alert = Alert(
                severity="medium",
                column=target_column,
                message=message,
                recommendation="Verifique el nombre de la columna target o quite la opción --target.",
            )
            target_column = None

        self.logger.info("Analyzing data quality...")
        dq_report = self._run_step(
            "data_quality",
            failed_steps,
            analyze_data_quality,
            df,
            missing_threshold=self.config.data_quality.missing_threshold,
            cardinality_threshold=self.config.data_quality.cardinality_threshold,
            constant_threshold=self.config.data_quality.constant_threshold,
            duplicate_threshold=self.config.data_quality.duplicate_threshold,
            column_types=column_types,
            default=_empty_data_quality_report(df),
        )

        self.logger.info("Performing univariate analysis...")
        univariate_report = self._run_step(
            "univariate", failed_steps, analyze_univariate, df, column_types, default=UnivariateReport()
        )

        self.logger.info("Analyzing relationships...")
        relationships_report = self._run_step(
            "relationships",
            failed_steps,
            analyze_relationships,
            df,
            numeric_cols,
            categorical_cols,
            min_correlation=self.config.visualizations.correlation_threshold,
            default=_empty_relationships_report(),
        )

        self.logger.info("Detecting outliers...")
        outliers_report = self._run_step(
            "outliers",
            failed_steps,
            analyze_outliers,
            df,
            numeric_cols,
            iqr_multiplier=self.config.outliers.iqr_multiplier,
            z_score_threshold=self.config.outliers.z_score_threshold,
            isolation_forest_enabled=self.config.outliers.isolation_forest_enabled,
            isolation_forest_contamination=self.config.outliers.isolation_forest_contamination,
            default=_empty_outlier_report(),
        )

        target_report = None
        if target_column:
            self.logger.info(f"Analyzing target: {target_column}")
            target_report = self._run_step(
                "target",
                failed_steps,
                analyze_target,
                df,
                target_column,
                self.config.target.target_type,
                imbalance_threshold=self.config.target.class_imbalance_threshold,
                default=None,
            )

        # Merge data-quality alerts with target-derived alerts (imbalance, leakage, missing target).
        combined_alerts = list(dq_report.alerts)
        if target_report:
            combined_alerts.extend(_build_target_alerts(target_report))
        if target_missing_alert:
            combined_alerts.append(target_missing_alert)
        combined_alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 999), a.column or ""))

        self.logger.info("Generating visualizations...")
        output_dir = Path(self.config.output_dir) / f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plots_dir = output_dir / "plots"

        plot_files = (
            self._run_step(
                "visualizations",
                failed_steps,
                generate_all_visualizations,
                df,
                column_types,
                numeric_cols,
                categorical_cols,
                datetime_cols,
                text_cols,
                relationships_report.correlation_matrix,
                plots_dir,
                self.config.visualizations,
                default={},
            )
            or {}
        )

        self.logger.info("Writing tablas CSV...")
        self._run_step(
            "tables",
            failed_steps,
            write_result_tables,
            output_dir,
            numeric_stats=univariate_report.numeric_stats,
            categorical_stats=univariate_report.categorical_stats,
            missing_per_column=dq_report.missing_per_column,
            numeric_pairs=relationships_report.numeric_pairs,
            categorical_pairs=relationships_report.categorical_pairs,
            mixed_pairs=relationships_report.mixed_pairs,
            outliers_report=outliers_report,
            alerts=combined_alerts,
            default=None,
        )

        # HTML Report
        self.logger.info("Generating HTML report...")
        html_path = output_dir / "report.html"

        html_file = self._run_step(
            "html_report",
            failed_steps,
            generate_html_report,
            html_path,
            dataset_name=dataset_name,
            correlation_id=self.correlation_id,
            df_shape=df.shape,
            data_quality={
                "missing_per_column": dq_report.missing_per_column,
                "duplicates": dq_report.duplicates,
                "constant_columns": dq_report.constant_columns,
                "quasi_constant_columns": dq_report.quasi_constant_columns,
            },
            numeric_stats=univariate_report.numeric_stats,
            categorical_stats=univariate_report.categorical_stats,
            correlations={
                "numeric_pairs": relationships_report.numeric_pairs,
                "categorical_pairs": relationships_report.categorical_pairs,
                "mixed_pairs": relationships_report.mixed_pairs,
            },
            outliers=outliers_report,
            target_analysis=target_report,
            plot_files=plot_files,
            alerts=combined_alerts,
            ignored_columns=override_result.ignored_columns,
            failed_steps=list(failed_steps),
            default=None,
        )

        # Summary JSON
        self.logger.info("Writing summary.json...")
        summary = {
            "dataset_name": dataset_name,
            "correlation_id": self.correlation_id,
            "timestamp": datetime.now().isoformat(),
            "shape": df.shape,
            "column_types": column_types,
            "column_overrides": {
                "ignored_columns": override_result.ignored_columns,
                "coercion_failures": override_result.coercion_failures,
                "unknown_columns": override_result.unknown_columns,
            },
            "data_quality": {
                "missing_per_column": dq_report.missing_per_column,
                "duplicates": dq_report.duplicates,
                "duplicate_rows_count": len(dq_report.duplicate_rows),
                "constant_columns": dq_report.constant_columns,
                "quasi_constant_columns": dq_report.quasi_constant_columns,
                "high_cardinality_columns": dq_report.high_cardinality_columns,
                "mixed_type_columns": dq_report.mixed_type_columns,
            },
            "alerts": [
                {"severity": a.severity, "column": a.column, "message": a.message, "recommendation": a.recommendation}
                for a in combined_alerts
            ],
            "univariate": {
                "numeric_columns": len(univariate_report.numeric_stats),
                "categorical_columns": len(univariate_report.categorical_stats),
                "datetime_columns": len(univariate_report.datetime_stats),
                "text_columns": len(univariate_report.text_stats),
            },
            "relationships": {
                "numeric_pairs_count": len(relationships_report.numeric_pairs),
                "categorical_pairs_count": len(relationships_report.categorical_pairs),
                "mixed_pairs_count": len(relationships_report.mixed_pairs),
                "multicollinearity_vif": relationships_report.vif,
            },
            "outliers": {"total_outlier_rows": len(outliers_report.outlier_indices_union)},
            "failed_steps": failed_steps,
        }

        if target_report:
            summary["target"] = {
                "target_column": target_report.target_column,
                "target_type": target_report.target_type,
                "n_missing": target_report.n_missing,
                "class_balance": {
                    "counts": target_report.class_balance.class_counts,
                    "is_imbalanced": target_report.class_balance.is_imbalanced,
                    "imbalance_ratio": target_report.class_balance.imbalance_ratio,
                }
                if target_report.class_balance
                else None,
                "leakage_alerts": target_report.leakage_alerts,
            }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Report saved to: {output_dir}")

        return {
            "success": len(failed_steps) == 0,
            "dataset_name": dataset_name,
            "output_dir": str(output_dir),
            "html_report": html_file,
            "summary_json": str(summary_path),
            "n_alerts": len(combined_alerts),
            "n_plots": sum(len(v) for v in plot_files.values()),
            "failed_steps": failed_steps,
        }
