"""
Univariate analysis: descriptive statistics, distributions, frequency tables.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .type_inference import coerce_to_datetime, parse_time_of_day

logger = logging.getLogger(__name__)


@dataclass
class NumericStats:
    """Statistics for numeric columns."""

    count: int
    missing: int
    mean: float
    median: float
    std: float
    min: float
    q25: float
    q75: float
    max: float
    iqr: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation
    zeros: int
    negatives: int
    normality_p_value: Optional[float] = None  # From Shapiro-Wilk or D'Agostino-Pearson


@dataclass
class CategoricalStats:
    """Statistics for categorical columns."""

    count: int
    missing: int
    nunique: int
    mode: str
    mode_frequency: int
    frequency_table: dict[str, int]  # Top categories


@dataclass
class DatetimeStats:
    """Statistics for datetime columns."""

    count: int
    missing: int
    min_date: str
    max_date: str
    n_days: int
    gaps: list[tuple[str, str]]  # Periods with no data
    frequency: Optional[str]  # Inferred frequency (daily, monthly, etc.)


@dataclass
class TimeOfDayStats:
    """Statistics for a time-of-day column ("11:43:47"): when in the day its rows happen."""

    count: int
    missing: int
    nunique: int
    earliest: str
    latest: str
    peak_hour: Optional[int]
    hour_counts: dict[int, int]  # all 24 hours, zero where nothing happens


@dataclass
class TextStats:
    """Statistics for text columns."""

    count: int
    missing: int
    nunique: int
    mean_length: float
    median_length: float
    min_length: int
    max_length: int
    empty_strings: int


@dataclass
class UnivariateReport:
    """Complete univariate analysis results."""

    numeric_stats: dict[str, NumericStats] = field(default_factory=dict)
    categorical_stats: dict[str, CategoricalStats] = field(default_factory=dict)
    datetime_stats: dict[str, DatetimeStats] = field(default_factory=dict)
    time_stats: dict[str, TimeOfDayStats] = field(default_factory=dict)
    text_stats: dict[str, TextStats] = field(default_factory=dict)


def analyze_numeric(series: pd.Series) -> NumericStats:
    """Analyze a numeric column."""
    valid = series.dropna()
    count = len(valid)

    if count == 0:
        return NumericStats(
            count=0,
            missing=len(series),
            mean=np.nan,
            median=np.nan,
            std=np.nan,
            min=np.nan,
            q25=np.nan,
            q75=np.nan,
            max=np.nan,
            iqr=np.nan,
            skewness=np.nan,
            kurtosis=np.nan,
            cv=np.nan,
            zeros=0,
            negatives=0,
        )

    mean = float(valid.mean())
    median = float(valid.median())
    std = float(valid.std())
    q25 = float(valid.quantile(0.25))
    q75 = float(valid.quantile(0.75))
    iqr = q75 - q25

    # Skewness and kurtosis
    skewness = float(valid.skew()) if count > 2 else np.nan
    kurtosis_val = float(valid.kurtosis()) if count > 3 else np.nan

    # Coefficient of variation
    cv = (std / mean * 100) if mean != 0 else 0

    # Counts
    zeros = int((valid == 0).sum())
    negatives = int((valid < 0).sum())

    # Normality test (use appropriate test based on sample size)
    normality_p_value = None
    if count >= 3:
        if count <= 5000:
            try:
                # Shapiro-Wilk for smaller samples
                _, p = stats.shapiro(valid)
                normality_p_value = float(p)
            except Exception:
                pass
        else:
            try:
                # D'Agostino-Pearson for larger samples
                _, p = stats.normaltest(valid)
                normality_p_value = float(p)
            except Exception:
                pass

    return NumericStats(
        count=count,
        missing=len(series) - count,
        mean=mean,
        median=median,
        std=std,
        min=float(valid.min()),
        q25=q25,
        q75=q75,
        max=float(valid.max()),
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis_val,
        cv=cv,
        zeros=zeros,
        negatives=negatives,
        normality_p_value=normality_p_value,
    )


def analyze_categorical(series: pd.Series, top_n: int = 20) -> CategoricalStats:
    """Analyze a categorical column."""
    valid = series.dropna()
    count = len(valid)

    if count == 0:
        return CategoricalStats(count=0, missing=len(series), nunique=0, mode="", mode_frequency=0, frequency_table={})

    nunique = series.nunique()
    mode_val = valid.mode()
    mode = str(mode_val[0]) if len(mode_val) > 0 else ""
    mode_freq = int(valid.value_counts().iloc[0]) if len(valid.value_counts()) > 0 else 0

    # Top N categories
    freq_table = valid.value_counts().head(top_n).to_dict()

    return CategoricalStats(
        count=count,
        missing=len(series) - count,
        nunique=nunique,
        mode=mode,
        mode_frequency=mode_freq,
        frequency_table=freq_table,
    )


def analyze_datetime(series: pd.Series) -> DatetimeStats:
    """Analyze a datetime column."""
    valid = series.dropna()
    count = len(valid)

    if count == 0:
        return DatetimeStats(count=0, missing=len(series), min_date="", max_date="", n_days=0, gaps=[], frequency=None)

    # Dates stored as text are read with the formats that recognised them, not by pandas guessing
    # one: a bare pd.to_datetime reads "05/03/07" as May 3rd and warns while doing so.
    if not pd.api.types.is_datetime64_any_dtype(valid):
        valid = coerce_to_datetime(valid).dropna()
        count = len(valid)
        if count == 0:
            return DatetimeStats(
                count=0, missing=len(series), min_date="", max_date="", n_days=0, gaps=[], frequency=None
            )

    min_date = str(valid.min().date())
    max_date = str(valid.max().date())
    n_days = (valid.max() - valid.min()).days

    # Detect gaps (periods with no data)
    gaps = []
    if count > 1:
        sorted_dates = valid.sort_values().reset_index(drop=True)
        diffs = sorted_dates.diff()
        large_gaps_mask = diffs > pd.Timedelta(days=7)
        for i in large_gaps_mask[large_gaps_mask].index:
            try:
                prev_date = sorted_dates.iloc[i - 1]
                curr_date = sorted_dates.iloc[i]
                gaps.append((str(prev_date.date()), str(curr_date.date())))
            except Exception:
                pass

    # Infer frequency
    frequency = None
    if count > 1:
        try:
            freq = pd.infer_freq(sorted_dates)
            frequency = freq
        except Exception:
            pass

    return DatetimeStats(
        count=count,
        missing=len(series) - count,
        min_date=min_date,
        max_date=max_date,
        n_days=n_days,
        gaps=gaps[:5],  # Limit to 5 gaps
        frequency=frequency,
    )


def analyze_time_of_day(series: pd.Series) -> TimeOfDayStats:
    """Analyze a time-of-day column: its earliest and latest time and how the rows spread over the hours."""
    missing = int(series.isna().sum())
    parsed = parse_time_of_day(series.dropna()).dropna()
    hour_counts = parsed.dt.hour.value_counts().reindex(range(24), fill_value=0)
    hours = {int(hour): int(n) for hour, n in hour_counts.items()}

    if parsed.empty:
        return TimeOfDayStats(
            count=0, missing=missing, nunique=0, earliest="", latest="", peak_hour=None, hour_counts=hours
        )

    return TimeOfDayStats(
        count=len(parsed),
        missing=missing,
        nunique=int(series.nunique()),
        earliest=parsed.min().strftime("%H:%M:%S"),
        latest=parsed.max().strftime("%H:%M:%S"),
        peak_hour=int(hour_counts.idxmax()),
        hour_counts=hours,
    )


def analyze_text(series: pd.Series) -> TextStats:
    """Analyze a text column."""
    valid = series.dropna()
    count = len(valid)

    if count == 0:
        return TextStats(
            count=0,
            missing=len(series),
            nunique=0,
            mean_length=0,
            median_length=0,
            min_length=0,
            max_length=0,
            empty_strings=0,
        )

    nunique = series.nunique()
    # An identifier can be a number (`id`, `customer_id`), and a number still has a length.
    text = valid.astype(str)
    lengths = text.str.len()
    empty = (lengths == 0).sum()

    return TextStats(
        count=count,
        missing=len(series) - count,
        nunique=nunique,
        mean_length=float(lengths.mean()),
        median_length=float(lengths.median()),
        min_length=int(lengths.min()),
        max_length=int(lengths.max()),
        empty_strings=int(empty),
    )


def analyze_univariate(df: pd.DataFrame, column_types: dict[str, str]) -> UnivariateReport:
    """
    Perform univariate analysis for all columns.

    Args:
        df: DataFrame
        column_types: Mapping of column name to semantic type

    Returns:
        UnivariateReport with statistics for each column
    """
    report = UnivariateReport()

    for col, dtype in column_types.items():
        if col not in df.columns:
            continue

        series = df[col]

        if "numeric" in dtype:
            report.numeric_stats[col] = analyze_numeric(series)
        elif dtype == "categorical" or dtype == "boolean" or dtype == "numeric_discrete":
            report.categorical_stats[col] = analyze_categorical(series)
        elif dtype == "datetime":
            report.datetime_stats[col] = analyze_datetime(series)
        elif dtype == "time":
            report.time_stats[col] = analyze_time_of_day(series)
        elif dtype in ("text", "identifier"):
            report.text_stats[col] = analyze_text(series)

    return report
