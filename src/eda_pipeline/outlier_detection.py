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
    note: str = ""  # why the method could not be applied, when it could not


# When at least half of a column holds one value, IQR (and MAD) measure no spread at all. The
# "normal" range then shrinks to that single value and every other value becomes an outlier: on
# Vistara, every discount of discount_pct (0 on 77% of the rows) did. The same happened to waste and
# waste_pct in Inventory and prima_cedida in the mortgage data. There is nothing to scale against,
# so the method is not applied and the report says why.
NO_SPREAD_NOTE_IQR = "No spread (IQR = 0): at least half of the values are the same"
NO_SPREAD_NOTE_MAD = "No spread (MAD = 0): at least half of the values are the same"

# A value that fills a quarter of a column lands on a quartile and the box then measures that value
# instead of the spread. On the spaceship titanic data, RoomService is 0 on 65% of the rows, so Q1
# was 0, the fence fell at 2.5 x Q3 and a fifth of the column came back as outliers; over the six
# numeric columns that was 5 030 of the 8 693 rows. Measured over the 111 numeric columns of
# data/raw, 24 have a quartile pinned to a repeated value and only 9 of them flag more than 1%;
# drawing the fence over the rows that hold a different value brings those 9 from 14-22% to 2-10%,
# and leaves every other column untouched.
PINNED_VALUE_MIN_SHARE = 0.25


def pinned_value(valid: pd.Series, q25: float, q75: float) -> float | None:
    """
    The repeated value that took over a quartile, if there is one.

    A quartile sitting exactly on a value that fills at least a quarter of the column is measuring
    that value, not the spread around it.
    """
    if valid.empty:
        return None
    counts = valid.value_counts()
    top_value = counts.index[0]
    if counts.iloc[0] / len(valid) < PINNED_VALUE_MIN_SHARE:
        return None
    return top_value if top_value in (q25, q75) else None


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
    if iqr == 0:
        return OutlierInfo(
            method="iqr",
            column=series.name or "unknown",
            n_outliers=0,
            outlier_indices=[],
            values=[],
            note=NO_SPREAD_NOTE_IQR,
        )

    note = ""
    pinned = pinned_value(valid, q25, q75)
    if pinned is not None:
        rest = valid[valid != pinned]
        rest_q25, rest_q75 = rest.quantile(0.25), rest.quantile(0.75)
        if rest_q75 > rest_q25:
            share = float((valid == pinned).mean() * 100)
            note = (
                f"The value {pinned:g} fills {share:.1f}% of the rows and takes a quartile: the "
                f"normal range was measured over the others"
            )
            q25, q75 = rest_q25, rest_q75
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
        note=note,
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
        return OutlierInfo(
            method="mad_zscore",
            column=series.name or "unknown",
            n_outliers=0,
            outlier_indices=[],
            values=[],
            note=NO_SPREAD_NOTE_MAD,
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


# Isolation Forest used to flag a fixed share of every dataset (contamination=0.1): 10% of the rows
# whatever the data looked like. The cut now comes from each dataset's own anomaly scores, with the
# same Tukey fence the report applies to single columns. Measured across the 17 datasets in data/raw,
# the flagged share then ranges from 0.3% (Order_Details) to about 12% (siniestros), and on clean
# synthetic data it drops under 2%.
#
# 300 trees: with 100, the random seed alone moved Dates between 0 and 31 flagged rows and siniestros
# between 37 and 63; with 300 they hold at 0 and at 50-59. 500 trees steadied nothing further and
# doubled the time on Order_Details (219432 rows: 4.6 s against 9.5 s).
ISOLATION_FOREST_TREES = 300

# Below a few dozen rows a share of flagged rows says nothing: one row of Sales_Outlet is 20%, and the
# quartiles of five scores are not a distribution. Products (40 rows) already behaves, moving by a
# single row across seeds.
ISOLATION_FOREST_MIN_ROWS = 30


def isolation_forest_cutoff(scores: np.ndarray, multiplier: float = 1.5) -> float | None:
    """
    Tukey's upper fence over the anomaly scores (higher = more isolated): Q3 + multiplier * IQR.

    None when the scores have no spread, for the same reason IQR itself is not applied then: the fence
    would collapse onto the common value and flag whatever sits above it.
    """
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q3 + multiplier * iqr)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float | None = None,
    random_state: int = 42,
    iqr_multiplier: float = 1.5,
) -> tuple[list[OutlierInfo], str]:
    """
    Detect multivariate outliers with Isolation Forest.

    Args:
        df: DataFrame
        numeric_cols: Numeric columns to use (rows missing any of them are left out)
        contamination: A fixed share of rows to flag, when set explicitly. None (the default) draws
            the cut from the anomaly scores instead (see isolation_forest_cutoff).
        random_state: Random seed
        iqr_multiplier: The fence multiplier applied to the scores

    Returns:
        (one OutlierInfo per numeric column, all sharing the same flagged rows; a Spanish note saying
        how the cut was drawn, or why the method was not applied)
    """
    if not numeric_cols:
        return [], ""

    df_numeric = df[numeric_cols].dropna()
    if len(df_numeric) < ISOLATION_FOREST_MIN_ROWS:
        return [], (
            f"Not applied: {len(df_numeric)} complete rows, and at least {ISOLATION_FOREST_MIN_ROWS} are needed."
        )

    try:
        if contamination is not None:
            share = min(contamination, 0.5)
            forest = IsolationForest(
                contamination=share, random_state=random_state, n_estimators=ISOLATION_FOREST_TREES
            )
            outlier_mask = forest.fit_predict(df_numeric) == -1
            note = f"Fixed share: {share:.0%} of the rows (isolation_forest_contamination)."
        else:
            forest = IsolationForest(random_state=random_state, n_estimators=ISOLATION_FOREST_TREES)
            scores = -forest.fit(df_numeric).score_samples(df_numeric)  # in (0, 1]; higher = more isolated
            cutoff = isolation_forest_cutoff(scores, iqr_multiplier)
            if cutoff is None:
                return [], "No spread in the anomaly scores: no row is flagged."
            outlier_mask = scores > cutoff
            note = f"Anomaly score above {cutoff:.3f} (Q3 + {iqr_multiplier:g}·IQR of this dataset's own scores)."
    except Exception as e:
        logger.warning(f"Isolation Forest failed: {e}")
        return [], ""

    outlier_indices = df_numeric[outlier_mask].index.tolist()
    if not outlier_indices:
        return [], note

    results = [
        OutlierInfo(
            method="isolation_forest",
            column=col,
            n_outliers=len(outlier_indices),
            outlier_indices=outlier_indices,
            values=df.loc[outlier_indices, col].dropna().tolist(),
            note=note,
        )
        for col in numeric_cols
    ]
    return results, note


@dataclass
class OutlierReport:
    """Complete outlier analysis results."""

    iqr_outliers: dict[str, OutlierInfo]
    mad_outliers: dict[str, OutlierInfo]
    multivariate_outliers: list[OutlierInfo]
    outlier_indices_union: set[int]  # All unique indices flagged as outliers
    multivariate_note: str = ""  # how the Isolation Forest cut was drawn, or why it was not applied


def analyze_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    iqr_multiplier: float = 1.5,
    z_score_threshold: float = 3.0,
    isolation_forest_enabled: bool = True,
    isolation_forest_contamination: float | None = None,
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
    multivariate_results, multivariate_note = [], ""
    if isolation_forest_enabled:
        multivariate_results, multivariate_note = detect_outliers_isolation_forest(
            df, numeric_cols, isolation_forest_contamination, iqr_multiplier=iqr_multiplier
        )

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
        multivariate_note=multivariate_note,
    )
