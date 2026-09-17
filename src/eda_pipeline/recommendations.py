"""
Preprocessing plan: the decisions the measurements settle on their own.

The report closes with four steps in the order they are applied — descartar, imputar, codificar,
escalar — and every line carries the measurement that produced it, so it can be checked instead of
believed.

Only what the data decides is here. Whether a categorical variable is ordinal, and in what order its
levels go, is not in the table: over the 89 categorical columns of ``data/raw`` an automatic rule
recognises the order of none of them. Whether the model needs scaling at all is not in the table
either: a tree does not care, a distance or a penalty does. Those are left to the person reading.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STEP_CONVERT = "convert"
STEP_DROP = "drop"
STEP_IMPUTE = "impute"
STEP_ENCODE = "encode"
STEP_SCALE = "scale"
STEP_ORDER = [STEP_CONVERT, STEP_DROP, STEP_IMPUTE, STEP_ENCODE, STEP_SCALE]

# Numbering these would leave gaps: a report without numbers kept as text starts at "Drop",
# and a "2." with no "1." above it reads as something missing. The order is the order of the sections.
STEP_LABELS = {
    STEP_CONVERT: "Convert",
    STEP_DROP: "Drop",
    STEP_IMPUTE: "Fill in",
    STEP_ENCODE: "Encode",
    STEP_SCALE: "Scale",
}

NUMERIC_TYPES = ("numeric_continuous", "numeric_discrete")
CATEGORICAL_TYPES = ("categorical", "boolean")

# Every threshold below was fixed on the 24 datasets of data/raw before the rules were written.

# Missing share above which a column carries almost nothing. Measured, the columns with gaps stop at
# 12% and the next one is penguins_lter.Comments at 92.4%, so any cut inside that gap picks the same
# column; 60% leaves room on both sides.
DROP_MISSING_PCT = 60.0

# |mean - median| / std under which both centres fill in practically the same value. Measured over
# the columns with gaps it runs from 0.03 (sales.customer_satisfaction, where the choice is
# irrelevant) to 0.82 (data_latin1.review_rating, where it is not).
MEAN_MEDIAN_GAP = 0.10

# Missing share above which filling with the most common value would pile too many rows onto it.
MODE_MAX_MISSING_PCT = 5.0

# One-hot limits: at most this many new columns, and at least ONE_HOT_MIN_ROWS_PER_COLUMN rows for
# each one. Vistara's Products.color (14 categories over 40 rows) is what the second limit catches;
# Dates.month_name (12 over 366) and clientes.sector_laboral (11 over 21572) clear both.
ONE_HOT_MAX_CATEGORIES = 15
ONE_HOT_MIN_ROWS_PER_COLUMN = 20
ONE_HOT_ALWAYS_OK = 3  # three categories are never too many, however small the table

# A category under this share of the rows leaves an almost empty one-hot column. 18 of the 89
# categorical columns have at least one.
RARE_CATEGORY_PCT = 1.0

# Ratio between the widest and the narrowest spread of a table's numeric columns. The datasets split
# in two with nothing in between: 3.1, 4.9 and 5.8 (Dates, healthcare, stroke) against 40.8 and up,
# to 2.1e9 in cobranza.
SCALING_STD_RATIO = 10.0

# How many times wider the standard deviation is than the interquartile range before the mean and the
# standard deviation stop describing the column: past that, the median and the IQR centre it better.
# A normal column sits near 0.74. Measured over the 111 numeric columns of data/raw the two groups do
# not touch: the widest column below is avg_glucose_level at 1.23 and the narrowest above is
# siniestros.valor_ultimo_avaluo at 3.60, so any cut inside that band picks the same 24 columns.
# The share of IQR outliers cannot be used for this: it moves with the fence (see outlier_detection
# .pinned_value) and, at 10%, it picked columns whose mean and standard deviation are perfectly fine
# (Order_Details.unit_price, Returns.refund_amount, both at a ratio of 1.0).
ROBUST_SCALER_SPREAD_RATIO = 2.0

# Correlation above which two columns carry the same information. Measured: 16 pairs across the
# datasets, up to r = 1.0000 between AC_POWER and DC_POWER in the solar data.
REDUNDANT_CORRELATION = 0.95
MAX_REDUNDANT_PAIRS = 15

_MAX_VALUE_CHARS = 40


@dataclass
class Recommendation:
    """One line of the preprocessing plan, with the measurement behind it."""

    step: str  # one of STEP_ORDER
    column: Optional[str]  # None when it concerns the whole table; "a / b" when it concerns a pair
    action: str
    evidence: str


def _number(value: float) -> str:
    """Format a measured value for a sentence: readable, never in scientific notation."""
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.3g}"


def _plural(count: int, singular: str, plural: str) -> str:
    """The count with the form that agrees with it: "1 category", "3 categories"."""
    return f"{count} {singular if count == 1 else plural}"


def _short(value: object) -> str:
    """A category's label, cut so one long cell cannot stretch the table."""
    text = str(value)
    return text if len(text) <= _MAX_VALUE_CHARS else text[: _MAX_VALUE_CHARS - 1] + "…"


def _missing_pct(df: pd.DataFrame, column: str, quality) -> float:
    """Missing percentage for a column, from the quality report when it has it."""
    reported = getattr(quality, "missing_per_column", {}) or {}
    if column in reported:
        return float(reported[column])
    return float(df[column].isna().mean() * 100) if len(df) else 0.0


def drop_recommendations(df, column_types, quality, target_column=None) -> list[Recommendation]:
    """Rows and columns that carry nothing into the model: repeats, gaps, single values, keys."""
    recs: list[Recommendation] = []
    n_rows = len(df)
    if n_rows == 0:
        return recs

    duplicates = int(getattr(quality, "duplicates", 0) or 0)
    if duplicates:
        recs.append(
            Recommendation(
                step=STEP_DROP,
                column=None,
                action="Remove the repeated rows before splitting train and test",
                evidence=(
                    f"{duplicates} of the {n_rows} rows are repeated "
                    f"({duplicates / n_rows * 100:.1f}% of the table); split across both sets, the "
                    f"model is scored on rows it has already seen"
                ),
            )
        )

    if target_column and target_column in df.columns:
        missing_rows = int(df[target_column].isna().sum())
        if missing_rows:
            recs.append(
                Recommendation(
                    step=STEP_DROP,
                    column=target_column,
                    action="Drop the rows with no target value",
                    evidence=(
                        f"{missing_rows} rows ({missing_rows / n_rows * 100:.1f}%) have no target; "
                        f"filling it in would invent the answer you want to predict"
                    ),
                )
            )

    constant_columns = set(getattr(quality, "constant_columns", []) or [])
    quasi_constant = dict(getattr(quality, "quasi_constant_columns", {}) or {})

    for column in df.columns:
        if column == target_column:
            continue
        semantic_type = column_types.get(column)
        missing_pct = _missing_pct(df, column, quality)

        if missing_pct >= DROP_MISSING_PCT:
            action = "Drop the column, or keep only a flag for whether the value is there"
            evidence = (
                f"{missing_pct:.1f}% of the values are missing: {int(df[column].notna().sum())} rows still have one"
            )
        elif column in constant_columns or semantic_type == "constant":
            action = "Drop the column"
            evidence = f"One single value across the {n_rows} rows: it distinguishes nothing"
        elif column in quasi_constant:
            action = "Drop the column"
            evidence = f"The same value on {quasi_constant[column]:.1f}% of the rows"
        elif semantic_type == "identifier":
            action = "Exclude from the model: it names the row, it does not describe it"
            evidence = (
                f"Classified as an identifier: {int(df[column].nunique(dropna=True))} distinct "
                f"values across {n_rows} rows"
            )
        else:
            continue

        recs.append(Recommendation(step=STEP_DROP, column=column, action=action, evidence=evidence))

    return recs


def conversion_recommendations(df, column_types, quality, skip=()) -> list[Recommendation]:
    """
    Columns whose values are numbers kept as text.

    Nothing else can be decided about them while they are text: amazon.rating holds 1464 numbers
    read as 28 categories, and without this the plan would offer to encode a rating.
    """
    numeric_as_text = dict(getattr(quality, "numeric_as_text", {}) or {})
    skip = set(skip)
    recs = []
    for column in df.columns:
        if column in skip or column not in numeric_as_text:
            continue
        # An identifier made of digits is still an identifier: it is excluded, not converted.
        if column_types.get(column) not in CATEGORICAL_TYPES + ("text",):
            continue
        present = int(df[column].notna().sum())
        recs.append(
            Recommendation(
                step=STEP_CONVERT,
                column=column,
                action="Convert it to a number and run the analysis again",
                evidence=(
                    f"{numeric_as_text[column]} of its {present} values are numbers kept as text: "
                    f"while they stay that way, the column is analysed as a category"
                ),
            )
        )
    return recs


def redundant_pairs(correlation_matrix, skip=()) -> list[tuple[str, str, float]]:
    """Pairs of numeric columns that carry the same information, strongest first."""
    if correlation_matrix is None or getattr(correlation_matrix, "empty", True):
        return []
    columns = [column for column in correlation_matrix.columns if column not in set(skip)]
    pairs = []
    for position, first in enumerate(columns):
        for second in columns[position + 1 :]:
            try:
                value = float(correlation_matrix.loc[first, second])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value) and abs(value) >= REDUNDANT_CORRELATION:
                pairs.append((first, second, value))
    pairs.sort(key=lambda pair: -abs(pair[2]))
    return pairs[:MAX_REDUNDANT_PAIRS]


def redundancy_recommendations(correlation_matrix, skip=()) -> list[Recommendation]:
    """One line per redundant pair; which of the two to keep is not in the data."""
    return [
        Recommendation(
            step=STEP_DROP,
            column=f"{first} / {second}",
            action="Keep only one of the two",
            evidence=f"r = {value:.3f}: either one can be predicted from the other",
        )
        for first, second, value in redundant_pairs(correlation_matrix, skip)
    ]


def imputation_recommendations(df, column_types, numeric_stats, quality, skip=(), target_column=None):
    """How to fill each column that has gaps: which centre, or a category of its own."""
    recs: list[Recommendation] = []
    skip = set(skip)

    for column in df.columns:
        if column in skip or column == target_column:
            continue
        missing_pct = _missing_pct(df, column, quality)
        if missing_pct <= 0:
            continue
        semantic_type = column_types.get(column)

        if semantic_type in NUMERIC_TYPES:
            stats = numeric_stats.get(column) if numeric_stats else None
            if stats is None or not np.isfinite(stats.median):
                continue
            spread = float(stats.std) if np.isfinite(stats.std) else 0.0
            gap = abs(float(stats.mean) - float(stats.median)) / spread if spread > 0 else 0.0
            action = f"Fill in with the median ({_number(stats.median)})"
            if gap >= MEAN_MEDIAN_GAP:
                evidence = (
                    f"{missing_pct:.1f}% missing. The mean ({_number(stats.mean)}) sits {gap:.2f} standard "
                    f"deviations away from the median, pulled by the tail"
                )
            else:
                evidence = (
                    f"{missing_pct:.1f}% missing. The mean ({_number(stats.mean)}) and the median agree within "
                    f"{gap:.3f} standard deviations, below {MEAN_MEDIAN_GAP:.2f}: either one will do"
                )

        elif semantic_type in CATEGORICAL_TYPES:
            counts = df[column].value_counts(dropna=True)
            if counts.empty:
                continue
            mode_share = float(counts.iloc[0] / counts.sum() * 100)
            if missing_pct <= MODE_MAX_MISSING_PCT:
                action = f"Fill in with the most common value ('{_short(counts.index[0])}')"
                evidence = (
                    f"{missing_pct:.1f}% missing, and that value already covers {mode_share:.1f}% of the "
                    f"rows that have one: the distribution barely moves"
                )
            else:
                action = "Add an explicit 'Unknown' category"
                evidence = (
                    f"{missing_pct:.1f}% missing: filling that in with the most common value would move "
                    f"all those rows onto it and give it a weight it does not have"
                )
        else:
            continue

        recs.append(Recommendation(step=STEP_IMPUTE, column=column, action=action, evidence=evidence))

    return recs


def encoding_recommendations(df, column_types, skip=(), target_column=None) -> list[Recommendation]:
    """What each categorical column admits: a 0/1 column, one-hot, or neither."""
    recs: list[Recommendation] = []
    skip = set(skip)
    n_rows = len(df)

    for column in df.columns:
        if column in skip or column == target_column or column_types.get(column) not in CATEGORICAL_TYPES:
            continue
        counts = df[column].value_counts(dropna=True)
        n_categories = int(len(counts))
        if n_categories <= 1:
            continue
        present = int(counts.sum())
        rare = [value for value, count in counts.items() if count / present * 100 < RARE_CATEGORY_PCT]

        fits_one_hot = n_categories <= ONE_HOT_MAX_CATEGORIES and (
            n_categories <= ONE_HOT_ALWAYS_OK or n_categories * ONE_HOT_MIN_ROWS_PER_COLUMN <= n_rows
        )

        if n_categories == 2:
            action = "A single 0/1 column"
            evidence = f"2 categories: '{_short(counts.index[0])}' and '{_short(counts.index[1])}'"
        elif fits_one_hot and rare:
            action = f"Group the categories under {RARE_CATEGORY_PCT:.0f}% into 'Other', then one-hot"
            evidence = (
                f"{n_categories} categories across {n_rows} rows, and "
                f"{_plural(len(rare), 'category appears', 'categories appear')} on fewer than "
                f"{RARE_CATEGORY_PCT:.0f}% of them"
            )
        elif fits_one_hot:
            action = f"One-hot ({n_categories} new columns)"
            evidence = f"{n_categories} categories across {n_rows} rows, each with enough of them"
        elif n_categories > ONE_HOT_MAX_CATEGORIES:
            action = "Avoid one-hot: group the rare categories into 'Other', or encode by frequency"
            evidence = f"{n_categories} categories across {n_rows} rows: one-hot would add {n_categories} columns"
        else:
            # Few categories, but a table too small to spend a whole column on each of them.
            action = "Avoid one-hot with so few rows: encode by frequency, or group categories"
            evidence = (
                f"{n_categories} categories across {n_rows} rows: one-hot would leave one new column "
                f"per {_plural(round(n_rows / n_categories), 'row', 'rows')}"
            )

        recs.append(Recommendation(step=STEP_ENCODE, column=column, action=action, evidence=evidence))

    return recs


def spread_ratio(stats) -> float:
    """Standard deviation over interquartile range; infinite when half the column is one value."""
    if stats is None or not np.isfinite(stats.std):
        return 0.0
    return float(stats.std) / float(stats.iqr) if stats.iqr > 0 else float("inf")


def scaling_recommendations(
    df, column_types, numeric_stats, log_scale_columns=(), skip=(), target_column=None
) -> list[Recommendation]:
    """Whether the numeric columns live on comparable scales, and which scaler suits each one."""
    skip = set(skip)
    log_columns = set(log_scale_columns or ())
    numeric_stats = numeric_stats or {}

    usable = [
        column
        for column in df.columns
        if column not in skip
        and column != target_column
        and column_types.get(column) in NUMERIC_TYPES
        and column in numeric_stats
    ]
    spreads = {
        column: float(numeric_stats[column].std)
        for column in usable
        if np.isfinite(numeric_stats[column].std) and numeric_stats[column].std > 0
    }
    if len(spreads) < 2:  # a single column has nothing to be compared against
        return []

    widest = max(spreads, key=lambda column: spreads[column])
    narrowest = min(spreads, key=lambda column: spreads[column])
    ratio = spreads[widest] / spreads[narrowest]

    if ratio < SCALING_STD_RATIO:
        return [
            Recommendation(
                step=STEP_SCALE,
                column=None,
                action="No scaling needed for a difference in magnitudes",
                evidence=(
                    f"The spread of {widest} ({_number(spreads[widest])}) is only {ratio:.1f} times "
                    f"that of {narrowest} ({_number(spreads[narrowest])})"
                ),
            )
        ]

    recs = [
        Recommendation(
            step=STEP_SCALE,
            column=None,
            action=(
                "Scale the numeric columns if your model is sensitive to scale (regularised "
                "regression, SVM, kNN, k-means, neural nets); trees and boosting do not need it"
            ),
            evidence=(
                f"The spread of {widest} ({_number(spreads[widest])}) is {_number(ratio)} times that "
                f"of {narrowest} ({_number(spreads[narrowest])}): unscaled, the first one dominates "
                f"any distance or penalty"
            ),
        )
    ]

    for column in usable:
        stats = numeric_stats[column]
        ratio = spread_ratio(stats)
        if column in log_columns:
            action = "Take log10, then StandardScaler"
            evidence = "Already drawn on a log scale: its tail eats the linear axis"
        elif not np.isfinite(ratio):
            # RobustScaler divides by the interquartile range, and here there is none to divide by.
            action = "StandardScaler (mean and standard deviation)"
            evidence = "At least half of its values are the same (IQR = 0): RobustScaler would have no divisor"
        elif ratio > ROBUST_SCALER_SPREAD_RATIO:
            action = "RobustScaler (median and IQR)"
            evidence = (
                f"Its standard deviation ({_number(stats.std)}) is {_number(ratio)} times its interquartile "
                f"range ({_number(stats.iqr)}): the tail drags the two numbers StandardScaler would use"
            )
        else:
            action = "StandardScaler (mean and standard deviation)"
            evidence = (
                f"Its standard deviation ({_number(stats.std)}) and its interquartile range "
                f"({_number(stats.iqr)}) agree: the mean and the standard deviation describe it well"
            )
        recs.append(Recommendation(step=STEP_SCALE, column=column, action=action, evidence=evidence))

    return recs


def build_recommendations(
    df,
    column_types,
    *,
    quality,
    numeric_stats,
    correlation_matrix=None,
    log_scale_columns=(),
    target_column=None,
) -> list[Recommendation]:
    """
    The whole preprocessing plan, in the order the steps are applied.

    Args:
        df: The analysed DataFrame.
        column_types: Inferred semantic type per column.
        quality: ``DataQualityReport`` (missing shares, duplicates, constant columns).
        numeric_stats: ``{column: NumericStats}`` from the univariate analysis.
        correlation_matrix: Numeric correlation matrix, to find redundant pairs.
        log_scale_columns: Columns already drawn on a log scale (rule D).
        target_column: Excluded from encoding and scaling; its gaps drop rows instead of being filled.

    Returns:
        ``Recommendation`` list ordered by step: descartar, imputar, codificar, escalar.
    """
    if df is None or len(df) == 0 or not len(df.columns):
        return []

    drops = drop_recommendations(df, column_types, quality, target_column)
    dropped = {rec.column for rec in drops if rec.column and rec.column != target_column}

    # A column still stored as text decides nothing else until it is a number, so it sits out the
    # remaining steps instead of receiving advice that assumes it is a category.
    conversions = conversion_recommendations(df, column_types, quality, skip=dropped)
    pending = dropped | {rec.column for rec in conversions}

    plan = conversions + drops + redundancy_recommendations(correlation_matrix, skip=pending)
    plan += imputation_recommendations(df, column_types, numeric_stats, quality, pending, target_column)
    plan += encoding_recommendations(df, column_types, pending, target_column)
    plan += scaling_recommendations(df, column_types, numeric_stats, log_scale_columns, pending, target_column)

    logger.info(f"Preprocessing plan: {len(plan)} recommendation(s)")
    return plan


def recommendations_by_step(plan) -> list[tuple[str, str, list[Recommendation]]]:
    """Group a plan into ``(step, label, recommendations)``, keeping only the steps that have lines."""
    grouped = []
    for step in STEP_ORDER:
        rows = [rec for rec in plan or [] if rec.step == step]
        if rows:
            grouped.append((step, STEP_LABELS[step], rows))
    return grouped
