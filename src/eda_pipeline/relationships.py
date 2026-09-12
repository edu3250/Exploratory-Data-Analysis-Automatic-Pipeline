"""
Relationship analysis: correlations, associations, multicollinearity detection.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """A pair of correlated variables."""

    var1: str
    var2: str
    correlation: float
    p_value: Optional[float] = None
    method: str = "pearson"  # pearson, spearman, cramer_v, eta


def pearson_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Pearson correlation."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    corr, p = pearsonr(valid["x"], valid["y"])
    return float(corr), float(p)


def spearman_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Spearman correlation."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    corr, p = spearmanr(valid["x"], valid["y"])
    return float(corr), float(p)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute bias-corrected Cramér's V for categorical association.
    """
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan

    # Contingency table
    ct = pd.crosstab(valid["x"], valid["y"])
    n = ct.sum().sum()

    if n == 0:
        return np.nan

    r, k = ct.shape
    if min(r, k) < 2:
        return np.nan

    # Bias-corrected Cramér's V (Bergsma & Wicher, 2013). No Yates continuity correction:
    # it deflates chi2 on 2x2 tables, and V is defined on the plain statistic.
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    phi2_corr = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        # As many categories as rows: any observed association is explainable by chance.
        return 0.0
    return float(min(1.0, np.sqrt(phi2_corr / denom)))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Compute correlation ratio (eta) for categorical vs numeric.
    """
    valid = pd.DataFrame({"cat": categories, "num": values}).dropna()
    if len(valid) < 2:
        return np.nan

    # Between-group variance
    categories = valid["cat"]
    values = valid["num"]
    cat_values = values.groupby(categories)

    # Grand mean
    grand_mean = values.mean()

    # Between-group sum of squares
    between_ss = sum(len(group) * (group.mean() - grand_mean) ** 2 for _, group in cat_values)

    # Total sum of squares
    total_ss = sum((values - grand_mean) ** 2)

    if total_ss == 0:
        return np.nan

    # Eta (correlation ratio)
    eta = np.sqrt(between_ss / total_ss)
    return float(eta)


def compute_numeric_correlations(
    df: pd.DataFrame, numeric_cols: list[str], min_correlation: float = 0.05
) -> list[CorrelationPair]:
    """
    Compute pairwise correlations between numeric columns.

    Returns:
        List of CorrelationPair objects with |correlation| > min_correlation
    """
    pairs = []

    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            corr, p = pearson_correlation(df[col1], df[col2])

            if not np.isnan(corr) and abs(corr) > min_correlation:
                pairs.append(CorrelationPair(var1=col1, var2=col2, correlation=corr, p_value=p, method="pearson"))

    # Sort by absolute correlation
    pairs.sort(key=lambda p: -abs(p.correlation))
    return pairs


def compute_categorical_associations(
    df: pd.DataFrame, categorical_cols: list[str], min_association: float = 0.05
) -> list[CorrelationPair]:
    """
    Compute pairwise associations between categorical columns (Cramér's V).

    Returns:
        List of CorrelationPair objects with V > min_association
    """
    pairs = []

    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1 :]:
            v = cramers_v(df[col1], df[col2])

            if not np.isnan(v) and v > min_association:
                pairs.append(CorrelationPair(var1=col1, var2=col2, correlation=v, method="cramer_v"))

    pairs.sort(key=lambda p: -p.correlation)
    return pairs


def compute_mixed_associations(
    df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str], min_association: float = 0.05
) -> list[CorrelationPair]:
    """
    Compute associations between categorical and numeric columns (correlation ratio).

    Returns:
        List of CorrelationPair objects with eta > min_association
    """
    pairs = []

    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            eta = correlation_ratio(df[cat_col], df[num_col])

            if not np.isnan(eta) and eta > min_association:
                pairs.append(CorrelationPair(var1=cat_col, var2=num_col, correlation=eta, method="eta"))

    pairs.sort(key=lambda p: -p.correlation)
    return pairs


ASSOCIATION_MAX_COLUMNS = 30


def association_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    max_columns: int = ASSOCIATION_MAX_COLUMNS,
) -> pd.DataFrame:
    """
    Square matrix of associations covering numeric and categorical columns alike.

    Every pair uses the measure that fits it: Pearson between two numbers (signed), Cramér's V
    between two categories, and the correlation ratio (eta) between a category and a number.

    Label-encoding the categories and running Pearson over everything, the usual shortcut, invents
    an order the categories do not have: on the stroke dataset it turns the work_type/age
    association (eta 0.68) into a misleading -0.36, a number that depends on how the five job
    types happened to be numbered.

    Columns keep the order they have in the DataFrame, and the matrix is capped at `max_columns`
    to bound the pairwise work on very wide datasets.
    """
    numeric = set(numeric_cols)
    categorical = set(categorical_cols)
    columns = [col for col in df.columns if col in numeric or col in categorical][:max_columns]
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)

    for position, col_a in enumerate(columns):
        for col_b in columns[position + 1 :]:
            a_is_numeric, b_is_numeric = col_a in numeric, col_b in numeric
            if a_is_numeric and b_is_numeric:
                value = pearson_correlation(df[col_a], df[col_b])[0]
            elif not a_is_numeric and not b_is_numeric:
                value = cramers_v(df[col_a], df[col_b])
            elif a_is_numeric:
                value = correlation_ratio(df[col_b], df[col_a])
            else:
                value = correlation_ratio(df[col_a], df[col_b])

            value = 0.0 if value is None or pd.isna(value) else float(value)
            matrix.loc[col_a, col_b] = value
            matrix.loc[col_b, col_a] = value

    return matrix


def compute_correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Compute numeric correlation matrix.
    """
    if not numeric_cols:
        return pd.DataFrame()

    df_numeric = df[numeric_cols].dropna(how="any")
    if len(df_numeric) < 2:
        return pd.DataFrame()

    return df_numeric.corr(method="pearson")


def detect_multicollinearity(correlation_matrix: pd.DataFrame, vif_threshold: float = 10.0) -> dict[str, float]:
    """
    Detect multicollinearity using VIF (computed from inverse correlation matrix).

    Args:
        correlation_matrix: Correlation matrix
        vif_threshold: VIF threshold for alerting

    Returns:
        Dictionary mapping column name to VIF
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] < 2:
        return {}

    try:
        # Compute inverse correlation matrix
        inv_corr = np.linalg.inv(correlation_matrix.values)
        # VIF is the diagonal of the inverse correlation matrix
        vif = np.diag(inv_corr)
        return {col: float(v) for col, v in zip(correlation_matrix.columns, vif)}
    except np.linalg.LinAlgError:
        # Singular matrix
        return {}


@dataclass
class RelationshipsReport:
    """Complete relationships analysis results."""

    numeric_pairs: list[CorrelationPair]
    categorical_pairs: list[CorrelationPair]
    mixed_pairs: list[CorrelationPair]
    correlation_matrix: pd.DataFrame
    vif: dict[str, float]
    # Numeric and categorical together; empty when the step could not build it.
    association_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)


def analyze_relationships(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], min_correlation: float = 0.05
) -> RelationshipsReport:
    """
    Perform comprehensive relationship analysis.
    """
    numeric_pairs = compute_numeric_correlations(df, numeric_cols, min_correlation)
    categorical_pairs = compute_categorical_associations(df, categorical_cols, min_correlation)
    mixed_pairs = compute_mixed_associations(df, categorical_cols, numeric_cols, min_correlation)

    corr_matrix = compute_correlation_matrix(df, numeric_cols)
    vif = detect_multicollinearity(corr_matrix)

    return RelationshipsReport(
        numeric_pairs=numeric_pairs,
        categorical_pairs=categorical_pairs,
        mixed_pairs=mixed_pairs,
        correlation_matrix=corr_matrix,
        vif=vif,
        association_matrix=association_matrix(df, numeric_cols, categorical_cols),
    )
