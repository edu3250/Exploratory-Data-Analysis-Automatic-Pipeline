"""
Outlier detection: IQR, MAD-based z-score, isolation forest.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class OutlierInfo:
    """Information about detected outliers."""

    method: str  # 'iqr', 'mad_zscore', 'isolation_forest'
    column: str
    n_outliers: int
    outlier_indices: list[int]
    values: list[float]


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> OutlierInfo:
    """
    Detect outliers using IQR method.

    Args:
        series: Numeric series
        multiplier: IQR multiplier (1.5 is standard, 3.0 is extreme)

    Returns:
        OutlierInfo with indices and values
    """
    valid = series.dropna()
    if len(valid) < 4:
        return OutlierInfo(method="iqr", column=series.name or "unknown", n_outliers=0, outlier_indices=[], values=[])

    q25 = valid.quantile(0.25)
    q75 = valid.quantile(0.75)
    iqr = q75 - q25

    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr

    outlier_mask = (valid < lower_bound) | (valid > upper_bound)
    outlier_idx = valid[outlier_mask].index.tolist()
    outlier_values = valid[outlier_mask].tolist()

    return OutlierInfo(
        method="iqr",
        column=series.name or "unknown",
        n_outliers=len(outlier_idx),
        outlier_indices=outlier_idx,
        values=outlier_values,
    )


def detect_outliers_mad(series: pd.Series, z_threshold: float = 3.0) -> OutlierInfo:
    """
    Detect outliers using MAD (Median Absolute Deviation) z-score.

    Args:
        series: Numeric series
        z_threshold: Z-score threshold (3.0 = 99.7% for normal data)

    Returns:
        OutlierInfo with indices and values
    """
    valid = series.dropna()
    if len(valid) < 2:
        return OutlierInfo(
            method="mad_zscore", column=series.name or "unknown", n_outliers=0, outlier_indices=[], values=[]
        )

    median = valid.median()
    mad = np.median(np.abs(valid - median))

    if mad == 0:
        # No variation; all values are identical to median
        return OutlierInfo(
            method="mad_zscore", column=series.name or "unknown", n_outliers=0, outlier_indices=[], values=[]
        )

    # Modified z-score (0.6745 constant for normal distribution)
    modified_z_scores = 0.6745 * (valid - median) / mad
    outlier_mask = np.abs(modified_z_scores) > z_threshold

    outlier_idx = valid[outlier_mask].index.tolist()
    outlier_values = valid[outlier_mask].tolist()

    return OutlierInfo(
        method="mad_zscore",
        column=series.name or "unknown",
        n_outliers=len(outlier_idx),
        outlier_indices=outlier_idx,
        values=outlier_values,
    )


def detect_outliers_isolation_forest(
    df: pd.DataFrame, numeric_cols: list[str], contamination: float = 0.1, random_state: int = 42
) -> list[OutlierInfo]:
    """
    Detect multivariate outliers using Isolation Forest.

    Args:
        df: DataFrame
        numeric_cols: Numeric columns to use
        contamination: Fraction of outliers to expect (0.0-0.5)
        random_state: Random seed

    Returns:
        List of OutlierInfo objects (one per numeric column with outliers found)
    """
    if not numeric_cols or len(df) < 2:
        return []

    # Prepare data
    df_numeric = df[numeric_cols].dropna()
    if len(df_numeric) < 2:
        return []

    try:
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=min(contamination, 0.5), random_state=random_state, n_estimators=100)
        predictions = iso_forest.fit_predict(df_numeric)

        # Get outlier indices (-1 indicates outlier)
        outlier_mask = predictions == -1
        if not outlier_mask.any():
            return []

        # Create result per column (mark overall multivariate outliers)
        results = []
        outlier_indices = df_numeric[outlier_mask].index.tolist()

        # Return as a single entry for all numeric columns
        if outlier_indices:
            for col in numeric_cols:
                results.append(
                    OutlierInfo(
                        method="isolation_forest",
                        column=col,
                        n_outliers=len(outlier_indices),
                        outlier_indices=outlier_indices,
                        values=df.loc[outlier_indices, col].dropna().tolist(),
                    )
                )

        return results
    except Exception as e:
        logger.warning(f"Isolation Forest failed: {e}")
        return []


@dataclass
class OutlierReport:
    """Complete outlier analysis results."""

    iqr_outliers: dict[str, OutlierInfo]
    mad_outliers: dict[str, OutlierInfo]
    multivariate_outliers: list[OutlierInfo]
    outlier_indices_union: set[int]  # All unique indices flagged as outliers


def analyze_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
    isolation_forest_enabled: bool = True,
    isolation_forest_contamination: float = 0.1,
) -> OutlierReport:
    """
    Perform comprehensive outlier analysis.
    """
    iqr_results = {}
    mad_results = {}

    # Apply IQR and MAD to each numeric column
    for col in numeric_cols:
        series = df[col].copy()
        series.name = col

        iqr_info = detect_outliers_iqr(series, iqr_multiplier)
        iqr_results[col] = iqr_info

        mad_info = detect_outliers_mad(series, z_score_threshold)
        mad_results[col] = mad_info

    # Multivariate outlier detection
    multivariate_results = []
    if isolation_forest_enabled:
        multivariate_results = detect_outliers_isolation_forest(df, numeric_cols, isolation_forest_contamination)

    # Collect all unique outlier indices
    all_indices = set()
    for info in iqr_results.values():
        all_indices.update(info.outlier_indices)
    for info in mad_results.values():
        all_indices.update(info.outlier_indices)
    for info in multivariate_results:
        all_indices.update(info.outlier_indices)

    return OutlierReport(
        iqr_outliers=iqr_results,
        mad_outliers=mad_results,
        multivariate_outliers=multivariate_results,
        outlier_indices_union=all_indices,
    )
