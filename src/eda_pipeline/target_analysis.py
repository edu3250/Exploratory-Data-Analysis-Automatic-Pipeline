"""
Target variable analysis: class balance, feature-target relationships, leakage detection.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr

from .relationships import adjusted_correlation_ratio, cramers_v

logger = logging.getLogger(__name__)

# An identifier names the row and free text is different on every row, so a chi-square over their
# thousands of categories measures uniqueness, not a relationship with the target: PassengerId led
# the spaceship table with an effect of 1.000 and Name with 0.999. Dates and times are not compared
# either; the report charts them over time instead.
UNTESTABLE_TYPES = ("identifier", "text", "datetime", "time", "constant")

# Which test fits depends on the kind of target and of column, and every effect runs from 0 to 1 so
# one table can rank them all:
#
#                        numeric column                       categorical column
#   numeric target       |Spearman| (its p-value)             adjusted eta (Kruskal-Wallis)
#   class target         adjusted eta (Kruskal-Wallis)        Cramér's V, bias-corrected (chi²)
#
# Chi² against a numeric target crossed each category with each distinct value of it: on Housing
# every categorical column came out between 0.69 and 0.81, Street (one value on 99.6% of the rows)
# third with 0.780, and the numeric columns, measured by mutual information on another scale, did
# not reach the top 20. Over data/raw, with every continuous column tried as a target, a shuffled
# target gave that chi² a median effect of 0.50; adjusted eta gives 0.00, and |Spearman| 0.02.
# Near-constant columns in the top 5 went from 19 of the 91 targets to 2, and numeric columns from
# 38% of the top 5 to 69%. For class targets, the corrected V and the adjusted eta take the shuffled
# effects' 99th percentile from 0.34 to 0.24 and from 0.24 to 0.14; the real targets (Spaceship,
# penguins, stroke) keep the same ranking.


@dataclass
class FeatureTargetRelationship:
    """Relationship between a feature and target."""

    feature: str
    test_statistic: float
    p_value: float
    test_name: str
    effect_size: Optional[float] = None  # 0 to 1: |Spearman|, adjusted eta or corrected Cramér's V
    support: int = 0  # Number of samples used


@dataclass
class ClassBalance:
    """Class distribution for classification target."""

    class_counts: dict[str, int]
    class_proportions: dict[str, float]
    is_imbalanced: bool
    imbalance_ratio: float  # Ratio of largest to smallest class


@dataclass
class TargetAnalysisReport:
    """Complete target analysis results."""

    target_column: str
    target_type: str  # 'classification', 'regression'
    n_samples: int
    n_missing: int

    # For classification
    class_balance: Optional[ClassBalance] = None

    # For regression
    target_stats: Optional[dict] = None

    # Feature-target relationships
    feature_relationships: list[FeatureTargetRelationship] = field(default_factory=list)

    # Potential leakage
    leakage_alerts: list[str] = field(default_factory=list)


def auto_detect_target_type(series: pd.Series, threshold: int = 20) -> str:
    """
    Auto-detect if target is classification or regression.

    Args:
        series: Target series
        threshold: If n_unique <= threshold, treat as classification

    Returns:
        'classification' or 'regression'
    """
    valid = series.dropna()
    if len(valid) == 0:
        return "unknown"

    # If numeric with few unique values, likely classification
    if pd.api.types.is_numeric_dtype(series):
        n_unique = valid.nunique()
        if n_unique <= threshold:
            return "classification"
        else:
            return "regression"

    # If string, likely classification
    if pd.api.types.is_string_dtype(series):
        return "classification"

    return "unknown"


def analyze_class_balance(series: pd.Series, imbalance_threshold: float = 0.8) -> ClassBalance:
    """
    Analyze class distribution for classification target.

    A target is considered imbalanced when its majority class share exceeds
    ``imbalance_threshold`` (e.g. threshold=0.8 flags any target where one
    class makes up more than 80% of the samples). This is independent of how
    many classes there are, unlike a rule based on the smallest class.
    """
    valid = series.dropna()
    counts = valid.value_counts().to_dict()
    proportions = (valid.value_counts(normalize=True) * 100).to_dict()

    # Imbalance ratio (largest class / smallest class)
    if counts:
        imbalance_ratio = max(counts.values()) / (min(counts.values()) + 1e-10)
        majority_share = max(proportions.values()) / 100.0
        is_imbalanced = majority_share > imbalance_threshold
    else:
        imbalance_ratio = 1.0
        is_imbalanced = False

    return ClassBalance(
        class_counts={str(k): v for k, v in counts.items()},
        class_proportions={str(k): v for k, v in proportions.items()},
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
    )


def test_feature_target_categorical(
    feature: pd.Series, target: pd.Series, is_target_numeric: bool = False
) -> FeatureTargetRelationship:
    """
    Test association between a categorical feature and the target.
    - If target is categorical: chi-square, with the bias-corrected Cramér's V as effect
    - If target is numeric: Kruskal-Wallis across the categories, with the adjusted eta as effect
    """
    valid = pd.DataFrame({"feature": feature, "target": target}).dropna()
    test_name = "kruskal_wallis" if is_target_numeric else "chi2"

    if len(valid) < 2:
        return FeatureTargetRelationship(
            feature=feature.name or "unknown", test_statistic=np.nan, p_value=np.nan, test_name=test_name
        )

    try:
        if is_target_numeric:
            groups = [group["target"].values for _, group in valid.groupby("feature")]
            statistic, p = kruskal(*groups)
            effect = adjusted_correlation_ratio(valid["feature"].astype(str), valid["target"])
        else:
            statistic, p, _, _ = chi2_contingency(pd.crosstab(valid["feature"], valid["target"]))
            effect = cramers_v(valid["feature"].astype(str), valid["target"].astype(str))

        return FeatureTargetRelationship(
            feature=feature.name or "unknown",
            test_statistic=float(statistic),
            p_value=float(p),
            test_name=test_name,
            effect_size=float(effect),
            support=len(valid),
        )
    except Exception as e:
        logger.debug(f"{test_name} failed for {feature.name}: {e}")
        return FeatureTargetRelationship(
            feature=feature.name or "unknown", test_statistic=np.nan, p_value=np.nan, test_name=test_name
        )


def test_feature_target_numeric(
    feature: pd.Series, target: pd.Series, is_target_numeric: bool
) -> FeatureTargetRelationship:
    """
    Test association between numeric feature and target.
    - If target is categorical: Kruskal-Wallis test, with the adjusted eta as effect
    - If target is numeric: Spearman's rank correlation, with its absolute value as effect
    """
    valid = pd.DataFrame({"feature": feature, "target": target}).dropna()

    if len(valid) < 2:
        return FeatureTargetRelationship(
            feature=feature.name or "unknown", test_statistic=np.nan, p_value=np.nan, test_name="unknown"
        )

    if is_target_numeric:
        # Ranks, not values: a long tail in the target or the column does not decide the effect.
        try:
            rho, p = spearmanr(valid["feature"].astype(float), valid["target"].astype(float))
            return FeatureTargetRelationship(
                feature=feature.name or "unknown",
                test_statistic=float(rho),
                p_value=float(p),
                test_name="spearman",
                effect_size=abs(float(rho)),
                support=len(valid),
            )
        except Exception as e:
            logger.debug(f"Spearman failed for {feature.name}: {e}")
            return FeatureTargetRelationship(
                feature=feature.name or "unknown", test_statistic=np.nan, p_value=np.nan, test_name="spearman"
            )
    else:
        # Kruskal-Wallis for classification (non-parametric ANOVA)
        try:
            groups = [group["feature"].values for _, group in valid.groupby("target")]
            h_stat, p = kruskal(*groups)

            return FeatureTargetRelationship(
                feature=feature.name or "unknown",
                test_statistic=h_stat,
                p_value=float(p),
                test_name="kruskal_wallis",
                # The correlation ratio, the 0-to-1 measure the association map uses for a category
                # against a number, adjusted for what the classes explain by chance.
                effect_size=float(adjusted_correlation_ratio(valid["target"].astype(str), valid["feature"])),
                support=len(valid),
            )
        except Exception as e:
            logger.debug(f"Kruskal-Wallis test failed for {feature.name}: {e}")
            return FeatureTargetRelationship(
                feature=feature.name or "unknown", test_statistic=np.nan, p_value=np.nan, test_name="kruskal_wallis"
            )


def detect_leakage(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> list[str]:
    """
    Detect potential data leakage (suspicious correlations/patterns).

    Returns:
        List of leakage alert messages
    """
    alerts = []

    # Check for features that are near-duplicates of target
    target = df[target_col].dropna()
    for feat in feature_cols:
        feature = df[feat].dropna()

        if len(feature) == 0 or len(target) == 0:
            continue

        # If both are numeric, check for very high correlation
        if pd.api.types.is_numeric_dtype(target) and pd.api.types.is_numeric_dtype(feature):
            corr = target.corr(feature)
            if abs(corr) > 0.99:
                alerts.append(f"Possible leakage: '{feat}' and '{target_col}' correlate {corr:.3f} (almost collinear)")

        # If both are categorical, check for near-perfect association
        if pd.api.types.is_string_dtype(target) and pd.api.types.is_string_dtype(feature):
            ct = pd.crosstab(target, feature)
            if ct.shape[0] == ct.shape[1] and np.diag(ct).sum() == len(target):
                alerts.append(f"Possible leakage: '{feat}' is almost identical to '{target_col}'")

    return alerts


def analyze_target(
    df: pd.DataFrame,
    target_column: str,
    target_type: Optional[str] = None,
    feature_columns: Optional[list[str]] = None,
    imbalance_threshold: float = 0.8,
    column_types: Optional[dict[str, str]] = None,
) -> TargetAnalysisReport:
    """
    Perform comprehensive target variable analysis.

    Args:
        df: DataFrame
        target_column: Name of target column
        target_type: 'classification' or 'regression'; auto-detect if None
        feature_columns: List of feature columns to test against target
        imbalance_threshold: Threshold for flagging imbalanced classification
        column_types: Inferred semantic types, so identifiers and free text are left out of the
            feature table. Every column is tested when they are not given.

    Returns:
        TargetAnalysisReport
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    target = df[target_column]
    n_missing = target.isnull().sum()

    # Auto-detect target type if not provided
    if target_type is None:
        target_type = auto_detect_target_type(target)

    # Analyze class balance for classification
    class_balance = None
    if target_type == "classification":
        class_balance = analyze_class_balance(target, imbalance_threshold)

    # Feature selection: use only numeric and categorical columns not in ignore list
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    if column_types:
        feature_columns = [col for col in feature_columns if column_types.get(col) not in UNTESTABLE_TYPES]

    # Test feature-target relationships. Which test fits depends on the kind of target, not on how it
    # happens to be stored: Transported is a boolean and stroke is 0/1, and both were read as numeric
    # targets, so their numeric features took the regression branch and came back without a p-value.
    feature_relationships = []
    is_target_numeric = target_type != "classification"

    for feat_col in feature_columns:
        feature = df[feat_col]

        if pd.api.types.is_numeric_dtype(feature):
            rel = test_feature_target_numeric(feature, target, is_target_numeric)
        else:
            rel = test_feature_target_categorical(feature, target, is_target_numeric)

        if not np.isnan(rel.p_value) or not np.isnan(rel.test_statistic):
            feature_relationships.append(rel)

    # The section answers which features are most related to the target, so it is ranked by how much
    # each one separates it (the 0-to-1 effects above) and only then by the p-value. With thousands of rows every p-value collapses to 0 and cannot rank anything: on
    # the spaceship data, Spa and RoomService both come back as p = 0 while their effects are 0.22
    # and 0.25.
    feature_relationships.sort(
        key=lambda r: (
            -(r.effect_size if r.effect_size is not None and not np.isnan(r.effect_size) else 0.0),
            r.p_value if not np.isnan(r.p_value) else float("inf"),
        )
    )

    # Detect leakage
    leakage_alerts = detect_leakage(df, target_column, feature_columns)

    return TargetAnalysisReport(
        target_column=target_column,
        target_type=target_type,
        n_samples=len(df),
        n_missing=int(n_missing),
        class_balance=class_balance,
        # Every feature: the report shows the first 20, and chart_columns ranks capped charts by all of them.
        feature_relationships=feature_relationships,
        leakage_alerts=leakage_alerts,
    )
