"""
Main EDA Pipeline Orchestrator.
"""

import difflib
import json
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config
from .data_loader import discover_batch_files, load_data, replace_missing_placeholders
from .data_quality import SEVERITY_ORDER, Alert, DataQualityReport, analyze_data_quality
from .html_report import generate_html_report
from .logging_util import generate_correlation_id, setup_logging
from .multicollinearity import analyze_multicollinearity
from .outlier_detection import OutlierReport, analyze_outliers
from .recommendations import build_recommendations
from .relationships import RelationshipsReport, analyze_relationships
from .tables import write_result_tables
from .target_analysis import analyze_target
from .transforms import analyze_transforms
from .type_inference import (
    apply_column_type_overrides,
    get_categorical_columns,
    get_datetime_columns,
    get_numeric_columns,
    get_text_columns,
    get_time_columns,
    infer_all_types,
)
from .univariate_analysis import UnivariateReport, analyze_univariate
from .visualizations import (
    choose_pair_plot,
    floor_split_columns,
    generate_all_visualizations,
    log_scale_columns,
)

# Stems that name a split of a dataset rather than the dataset itself. Kaggle ships train.csv,
# test.csv and sample_submission.csv inside a folder whose name is the only thing that says what
# the data is, so a report called "train" says nothing about which competition it came from.
# Measured over the 31 data files under data/raw: the only stems that identify nothing are the
# three in "Kaggle Titanic" (train, test, gender_submission); every other file already names its
# own dataset (clientes, penguins_size, Plant_1_Generation_Data, Sales_Receipts...).
# "data" is deliberately absent: it is a neutral filename rather than a split, and qualifying it
# would rename datasets whose names are already fine.
SPLIT_STEMS = frozenset(
    {
        "train",
        "test",
        "val",
        "valid",
        "validation",
        "dev",
        "holdout",
        "submission",
        "sample_submission",
        "gender_submission",
    }
)

# Folders that say where a file lives, not what it holds. Qualifying train.csv with one of these
# would give "raw_train", which is no better than "train", so the plain stem is kept instead.
CONTAINER_FOLDERS = frozenset(
    {
        "data",
        "raw",
        "dataset",
        "datasets",
        "input",
        "inputs",
        "files",
        "csv",
        "tmp",
        "temp",
        "downloads",
    }
)


def _slug(name: str) -> str:
    """
    A folder or file name as a path-safe piece of another name ("power Bi" -> "power_Bi").

    Runs of replaced characters collapse into one underscore, so "house prices (2024)" reads as
    "house_prices_2024" rather than "house_prices__2024_".
    """
    cleaned = "".join(char if (char.isalnum() or char in "-._") else "_" for char in name)
    return "_".join(part for part in cleaned.split("_") if part).strip("._")


def dataset_name_for(file_path: Path) -> str:
    """
    What one dataset's report is called: the file's stem, qualified by its folder when the stem
    only names a split.

    ``data/raw/Kaggle Titanic/train.csv`` becomes ``Kaggle_Titanic_train``, while
    ``data/raw/mx/clientes.csv`` stays ``clientes``: a stem that already names its dataset is
    never touched. The folder is left out when it only says where the file lives (``data/raw``),
    or when it repeats the stem.
    """
    stem = file_path.stem
    if stem.lower() not in SPLIT_STEMS:
        return stem

    try:
        parent = file_path.resolve().parent.name
    except OSError:  # an unresolvable path (a dead network drive, a name Windows rejects)
        parent = file_path.parent.name

    if parent.lower() in CONTAINER_FOLDERS or parent.lower() == stem.lower():
        return stem

    slug = _slug(parent)
    return f"{slug}_{stem}" if slug else stem


class TargetColumnNotFoundError(ValueError):
    """Raised when the configured target column is missing (single-file mode fails fast on this)."""


def _missing_target_message(target_column: str, available_columns: list[str]) -> str:
    """Build the error message for a missing target column, with fuzzy-match suggestions."""
    suggestions = difflib.get_close_matches(target_column, available_columns, n=3)
    message = (
        f"The target column '{target_column}' is not in the dataset. Available columns: {', '.join(available_columns)}."
    )
    if suggestions:
        message += f" Did you mean: {', '.join(suggestions)}?"
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
                    f"Imbalanced classes in target '{target_report.target_column}': "
                    f"the majority class covers {majority_pct:.1f}% of the rows."
                ),
                recommendation="Consider resampling (SMOTE, undersampling) or metrics that survive imbalance (F1, AUC-PR).",
            )
        )
    for leak_message in target_report.leakage_alerts:
        alerts.append(
            Alert(
                severity="high",
                column=target_report.target_column,
                message=leak_message,
                recommendation="Check whether this column should be kept out of the model as leakage.",
            )
        )
    return alerts


def _placeholder_alerts(missing_placeholders: dict[str, dict[str, int]]) -> list[Alert]:
    """One alert per column where punctuation-only cells were read as missing values."""
    alerts = []
    for column, counts in missing_placeholders.items():
        total = sum(counts.values())
        values = ", ".join(f"'{value}' ({count})" for value, count in counts.items())
        alerts.append(
            Alert(
                severity="low",
                column=column,
                message=f"Column '{column}': {total} punctuation-only value(s) were read as missing: {values}",
                recommendation="Check where the data came from: those markers usually stand for a value nobody recorded.",
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
        numeric_pairs=[], categorical_pairs=[], mixed_pairs=[], correlation_matrix=pd.DataFrame()
    )


def _empty_outlier_report() -> OutlierReport:
    """Fallback used when the outliers step itself fails."""
    return OutlierReport(iqr_outliers={}, mad_outliers={}, multivariate_outliers=[], outlier_indices_union=set())


def _timestamp() -> str:
    """Timestamp used to name report folders."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class EDAPipeline:
    """Main EDA Pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.correlation_id = generate_correlation_id()
        self.log_file = None
        self.results = {}
        # On a batch run, the folder that holds every report of the run (None for a single file).
        self.batch_output_dir = None

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
                    dataset_name = dataset_name_for(file_path)
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
        self.logger.info(f"Found {len(files)} file(s) to process in {folder_path}")

        # One folder for the whole run: a folder of eight files used to scatter eight timestamped
        # report folders across the output directory, mixed with those of every previous run.
        self.batch_output_dir = Path(self.config.output_dir) / self._batch_run_folder_name(folder_path, _timestamp())
        self.logger.info(f"This batch will write its reports to: {self.batch_output_dir}")

        name_counts = Counter(dataset_name_for(f) for f in files)
        all_results = {}
        for file_path in files:
            dataset_name = self._unique_dataset_name(file_path, name_counts)
            all_results[dataset_name] = self._load_and_analyze(file_path, dataset_name, self.batch_output_dir)

        return all_results

    @staticmethod
    def _batch_run_folder_name(folder_path: Path, timestamp: str) -> str:
        """
        Name of the folder that holds every report of one batch run: `<input folder>_batch_<timestamp>`.

        The input folder's name is slugified, so spaces and other characters that are awkward in a
        path never leak into it ("power Bi" -> "power_Bi").
        """
        slug = _slug(folder_path.resolve().name)
        return f"{slug or 'data'}_batch_{timestamp}"

    def _dataset_output_dir(self, dataset_name: str, batch_dir: Path | None) -> Path:
        """
        Where one dataset's report goes.

        In a batch, every report is a subfolder of the run's folder, which already carries the
        timestamp. A single file keeps its own timestamped folder in the output directory.
        """
        if batch_dir is not None:
            return batch_dir / dataset_name
        return Path(self.config.output_dir) / f"{dataset_name}_{_timestamp()}"

    @staticmethod
    def _unique_dataset_name(file_path: Path, name_counts: Counter) -> str:
        """Disambiguate same-name files (e.g. sales.csv / sales.parquet) with a suffix hint."""
        name = dataset_name_for(file_path)
        if name_counts[name] > 1:
            return f"{name}_{file_path.suffix.lstrip('.').lower()}"
        return name

    def _load_and_analyze(self, file_path: Path, dataset_name: str, batch_dir: Path | None = None) -> dict:
        """
        Load a single file and, if that succeeds, run the full per-dataset analysis.

        `batch_dir` is the run's folder when this file is part of a batch, and None for a single file.
        """
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
            self.logger.error(f"Could not load {file_path.name}: {e}\n{traceback.format_exc()}")
            if self.config.strict_mode:
                raise
            return {"error": str(e), "success": False}

        try:
            # df.shape and the analysis itself are both inside this dataset's error
            # handling (previously df.shape was logged outside the try, so a loader
            # returning something unexpected, e.g. the old Excel dict-of-sheets bug,
            # would escape to the outer handler and abort the whole run).
            self.logger.info(f"Analyzing {dataset_name}: {df.shape[0]} rows × {df.shape[1]} columns")
            return self._analyze_dataset(dataset_name, df, batch_dir)
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
                f"Step '{step_name}' failed (correlation_id={self.correlation_id}): {e}\n{traceback.format_exc()}"
            )
            step_failures.append(step_name)
            return default

    def _analyze_dataset(self, dataset_name: str, df: pd.DataFrame, batch_dir: Path | None = None) -> dict:
        """Analyze a single dataset, isolating failures per analysis step (see _run_step)."""
        failed_steps: list[str] = []

        # Type inference + user overrides. Not step-isolated: every later step depends on it,
        # so a failure here should surface as a dataset-level error (via _load_and_analyze).
        # Cells that only hold punctuation (".", "?", "-") are missing values, not categories. They
        # are replaced before anything reads the data, and reported below as alerts.
        df, missing_placeholders = replace_missing_placeholders(df)

        self.logger.info("Inferring column types...")
        column_types = infer_all_types(df, identifier_min_unique=self.config.data_quality.cardinality_threshold)
        df, override_result = apply_column_type_overrides(df, column_types, self.config.column_types)
        column_types = override_result.column_types

        numeric_cols = get_numeric_columns(column_types)
        categorical_cols = get_categorical_columns(column_types)
        text_cols = get_text_columns(column_types)
        datetime_cols = get_datetime_columns(column_types)
        time_cols = get_time_columns(column_types)

        self.logger.info(
            f"  Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, "
            f"Text: {len(text_cols)}, Datetime: {len(datetime_cols)}, Time: {len(time_cols)}"
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
                recommendation="Check the target column name, or drop the --target option.",
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

        self.logger.info("Measuring multicollinearity...")
        multicollinearity_report = self._run_step(
            "multicollinearity",
            failed_steps,
            analyze_multicollinearity,
            df,
            column_types,
            target_column=target_column,
            default=None,
        )
        multicollinearity_summary = multicollinearity_report.to_summary() if multicollinearity_report else None

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
                column_types=column_types,
                default=None,
            )

        # Merge data-quality alerts with target-derived alerts (imbalance, leakage, missing target).
        combined_alerts = list(dq_report.alerts)
        combined_alerts.extend(_placeholder_alerts(missing_placeholders))
        if target_report:
            combined_alerts.extend(_build_target_alerts(target_report))
        if target_missing_alert:
            combined_alerts.append(target_missing_alert)
        combined_alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 999), a.column or ""))

        self.logger.info("Checking which columns need a log scale...")
        log_scale = self._run_step("log_scale", failed_steps, log_scale_columns, df, column_types, default={}) or {}

        self.logger.info("Checking which columns are filled by their lowest value...")
        floor_split = (
            self._run_step("floor_split", failed_steps, floor_split_columns, df, column_types, default={}) or {}
        )

        self.logger.info("Checking which skewed columns a transformation fixes...")
        transforms = self._run_step(
            "transforms",
            failed_steps,
            analyze_transforms,
            df,
            column_types,
            target_column=target_column,
            target_type=target_report.target_type if target_report else None,
            default=None,
        )

        self.logger.info("Choosing the pair plot...")
        pair_plot = self._run_step(
            "pair_plot",
            failed_steps,
            choose_pair_plot,
            df,
            column_types,
            target_column=target_column,
            target_type=target_report.target_type if target_report else None,
            default=None,
        )

        self.logger.info("Building the preprocessing plan...")
        recommendations = (
            self._run_step(
                "recommendations",
                failed_steps,
                build_recommendations,
                df,
                column_types,
                quality=dq_report,
                numeric_stats=univariate_report.numeric_stats,
                correlation_matrix=relationships_report.correlation_matrix,
                log_scale_columns=log_scale,
                transforms=transforms,
                target_column=target_column,
                default=[],
            )
            or []
        )

        self.logger.info("Generating visualizations...")
        output_dir = self._dataset_output_dir(dataset_name, batch_dir)
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
                assoc_matrix=relationships_report.association_matrix,
                target_column=target_column,
                target_type=target_report.target_type if target_report else None,
                time_cols=time_cols,
                pair_plot=pair_plot,
                log_scale=log_scale,
                floor_split=floor_split,
                transforms=transforms,
                default={},
            )
            or {}
        )

        # How each column ended up stored, shown in the report next to its inferred type.
        column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        self.logger.info("Writing the CSV tables...")
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
            column_types=column_types,
            column_dtypes=column_dtypes,
            recommendations=recommendations,
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
            column_types=column_types,
            column_dtypes=column_dtypes,
            data_preview=df,
            time_stats=univariate_report.time_stats,
            pair_plot=pair_plot,
            log_scale_columns=list(log_scale),
            floor_split_columns={col: asdict(check) for col, check in floor_split.items()},
            recommendations=recommendations,
            multicollinearity=multicollinearity_report,
            transforms=transforms,
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
                "missing_placeholders": missing_placeholders,
            },
            "alerts": [
                {"severity": a.severity, "column": a.column, "message": a.message, "recommendation": a.recommendation}
                for a in combined_alerts
            ],
            "univariate": {
                "numeric_columns": len(univariate_report.numeric_stats),
                "categorical_columns": len(univariate_report.categorical_stats),
                "datetime_columns": len(univariate_report.datetime_stats),
                "time_columns": len(univariate_report.time_stats),
                "text_columns": len(univariate_report.text_stats),
            },
            "relationships": {
                "numeric_pairs_count": len(relationships_report.numeric_pairs),
                "categorical_pairs_count": len(relationships_report.categorical_pairs),
                "mixed_pairs_count": len(relationships_report.mixed_pairs),
                "multicollinearity_vif": multicollinearity_summary["vif"] if multicollinearity_summary else {},
            },
            "multicollinearity": multicollinearity_summary,
            "outliers": {
                "total_outlier_rows": len(outliers_report.outlier_indices_union),
                "isolation_forest": {
                    "outlier_rows": outliers_report.multivariate_outliers[0].n_outliers
                    if outliers_report.multivariate_outliers
                    else 0,
                    "note": outliers_report.multivariate_note,
                },
            },
            "pair_plot": asdict(pair_plot) if pair_plot else None,
            "log_scale_columns": {col: asdict(check) for col, check in log_scale.items()},
            "floor_split_columns": {col: asdict(check) for col, check in floor_split.items()},
            "transforms": transforms.to_summary() if transforms else None,
            "recommendations": [asdict(rec) for rec in recommendations],
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
