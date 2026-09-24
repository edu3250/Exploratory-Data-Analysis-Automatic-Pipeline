"""
Skewed columns: which ones a transformation brings close to normal, measured before and after.

A model rarely needs normal columns. Trees and boosting split on the order of the values and do not
care; a linear model assumes normal residuals, not normal columns. What a long tail does is give a
few rows most of the weight wherever a sum of squares or a distance is taken: in a linear fit, in
kNN, SVM or k-means, in PCA, and in the errors of a regression whose target is the skewed column.
So the question asked here is the one that leads to a step of the plan: does a transformation fix it?

Two transformations are tried on each continuous column with a long tail, and the first that brings
its skew within ±FIXED_SKEW and straightens its QQ plot is kept:

- log1p, log(1 + x), when the column has no negative values: the easier one to read back;
- Yeo-Johnson otherwise, which takes any sign and also pulls in a tail on the left.

Before and after are measured by the skew and by r, the correlation of the normal QQ plot: 1 when
its points lie on a straight line, that is, when the column is normal.
"""

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

LOG1P = "log1p"
YEO_JOHNSON = "yeo-johnson"
METHOD_LABELS = {LOG1P: "log1p", YEO_JOHNSON: "Yeo-Johnson"}

# Measured over the 121 distinct numeric columns of the 31 datasets under data/raw (a column that
# train and test share counts once). 57 have a skew beyond ±1, and two kinds among them gain nothing
# from a transformation:
# - 30 have their lowest value on 25% of the rows or more (PoolArea 99.5% zeros, the spending columns
#   of Spaceship Titanic around 65%). Their QQ plot is a flat run and a hook: Yeo-Johnson brings 8 of
#   the 30 within ±0.5, and lifts the median r only from 0.58 to 0.78. The floor split draws them.
# - 11 have 20 distinct values or fewer (Parch, SibSp, KitchenAbvGr): their QQ plot is a staircase.
# That leaves 26 continuous columns. log1p brings 11 of them within ±0.5 and Yeo-Johnson 22.
# A skew within ±0.5 is not enough on its own: TotalBsmtSF (Housing) goes from 1.52 to 0.23 under
# Yeo-Johnson while its QQ plot stays as bent as it was (r 0.9750 -> 0.9754), because the 37 houses
# without a basement stay apart from the rest. Every other column that reaches ±0.5 gains at least
# 0.014 (porcentaje_cobertura, 0.948 -> 0.962), so a gain of MIN_QQ_GAIN separates the two. That draws
# 21 of the 26, with the median r going from 0.90 to 0.99; the other 5 (TotalBsmtSF, monto_credito and
# two premiums of credito_asegurado, siniestros.monto_siniestro) are listed without a chart. The
# slowest file, credito_asegurado with 76 526 rows, takes about half a second.
MIN_SKEW = 1.0
MAX_FLOOR_PCT = 25.0  # the share at which the floor split takes a column over
MIN_DISTINCT = 20  # more than this many distinct values; the same line the floor split draws
FIXED_SKEW = 0.5
MIN_QQ_GAIN = 0.005
MIN_VALUES = 20

# A QQ plot of every row puts 219 432 points on the chart for Vistara's Order_Details. Above this many
# rows the plot uses the quantiles at evenly spaced probabilities instead, which draw the same line.
QQ_POINTS = 1000


@dataclass
class TransformCheck:
    """One skewed column, and what the transformation that fixes it, if any, does to it."""

    column: str
    method: Optional[str]  # LOG1P, YEO_JOHNSON, or None when neither fixes it (see is_fixed_by)
    rows: int
    skew_before: float
    qq_r_before: float
    skew_after: Optional[float]  # under the chosen method
    qq_r_after: Optional[float]
    log1p_skew: Optional[float]  # None when the column has negative values
    log1p_qq_r: Optional[float]
    yeo_johnson_skew: Optional[float]  # None when it could not be fitted
    yeo_johnson_qq_r: Optional[float]
    lmbda: Optional[float]  # Yeo-Johnson's lambda on these rows

    @property
    def fixed(self) -> bool:
        return self.method is not None

    @property
    def label(self) -> str:
        return METHOD_LABELS.get(self.method, "")


@dataclass
class TransformReport:
    columns: dict[str, TransformCheck] = field(default_factory=dict)
    target: Optional[TransformCheck] = None

    @property
    def fixed(self) -> list[TransformCheck]:
        return [check for check in self.columns.values() if check.fixed]

    @property
    def unfixed(self) -> list[TransformCheck]:
        return [check for check in self.columns.values() if not check.fixed]

    def to_summary(self) -> dict:
        return {
            "columns": {name: asdict(check) for name, check in self.columns.items()},
            "target": asdict(self.target) if self.target else None,
        }


def _values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _skew(values: np.ndarray) -> Optional[float]:
    """The same skew the numeric table shows (pandas), or None when it cannot be measured."""
    if not np.all(np.isfinite(values)):
        return None
    skew = float(pd.Series(values).skew())
    return skew if np.isfinite(skew) else None


def qq_points(values) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Normal QQ plot of ``values``: normal quantiles, the column's quantiles at the same probabilities,
    and the straight line fitted through them (slope, intercept, r).
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    points = min(len(ordered), QQ_POINTS)
    probabilities = (np.arange(1, points + 1) - 0.5) / points
    if len(ordered) > points:
        ordered = np.quantile(ordered, probabilities)
    theoretical = stats.norm.ppf(probabilities)
    fit = stats.linregress(theoretical, ordered)
    return theoretical, ordered, float(fit.slope), float(fit.intercept), float(fit.rvalue)


def apply_transform(values, method: str, lmbda: Optional[float] = None) -> np.ndarray:
    """The values after ``method``, with the lambda measured for Yeo-Johnson."""
    values = np.asarray(values, dtype=float)
    if method == LOG1P:
        return np.log1p(values)
    if method == YEO_JOHNSON:
        return stats.yeojohnson(values, lmbda)
    raise ValueError(f"Unknown transformation: {method}")


def is_fixed_by(skew_after: Optional[float], qq_r_before: float, qq_r_after: Optional[float]) -> bool:
    """A transformation fixes a column when its skew ends within ±FIXED_SKEW and its QQ plot straightens."""
    if skew_after is None or qq_r_after is None:
        return False
    return abs(skew_after) < FIXED_SKEW and qq_r_after - qq_r_before >= MIN_QQ_GAIN


def describe_attempt(check: TransformCheck, method: str) -> str:
    """What one transformation does to the column, in words: why it was not the one, when it was not."""
    if method == LOG1P:
        skew, qq_r = check.log1p_skew, check.log1p_qq_r
        if skew is None:
            return "log1p cannot take its negative values"
    else:
        skew, qq_r = check.yeo_johnson_skew, check.yeo_johnson_qq_r
        if skew is None:
            return "Yeo-Johnson could not be fitted"
    label = METHOD_LABELS[method]
    if abs(skew) >= FIXED_SKEW:
        return f"{label} leaves a skew of {skew:.2f}"
    if not is_fixed_by(skew, check.qq_r_before, qq_r):
        return (
            f"{label} brings the skew to {skew:.2f} but does not straighten its QQ plot "
            f"(r = {check.qq_r_before:.3f} → {qq_r:.3f})"
        )
    return f"{label} brings the skew to {skew:.2f}"


def describe_attempts(check: TransformCheck) -> str:
    """Both attempts: "log1p leaves a skew of 1.95; Yeo-Johnson leaves a skew of 1.58"."""
    return "; ".join(describe_attempt(check, method) for method in (LOG1P, YEO_JOHNSON))


def check_transform(series: pd.Series) -> Optional[TransformCheck]:
    """
    Measure one column; None when it is not a candidate: too few values, 20 distinct values or fewer,
    its lowest value on 25% of the rows or more, or a skew within ±1.
    """
    values = _values(series)
    if len(values) < MIN_VALUES or len(np.unique(values)) <= MIN_DISTINCT:
        return None
    if (values == values.min()).mean() * 100 >= MAX_FLOOR_PCT:
        return None
    skew = _skew(values)
    if skew is None or abs(skew) <= MIN_SKEW:
        return None

    # Very large values can overflow on the way to Yeo-Johnson's lambda: that attempt then counts as
    # failed, and the column is judged on the other one.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logged = np.log1p(values) if values.min() >= 0 else None
        try:
            yeo_johnson, lmbda = stats.yeojohnson(values)
        except (ArithmeticError, ValueError) as error:
            logger.debug(f"Yeo-Johnson failed on {series.name}: {error}")
            yeo_johnson, lmbda = None, float("nan")
    log1p_skew = _skew(logged) if logged is not None else None
    log1p_qq_r = qq_points(logged)[4] if log1p_skew is not None else None
    yeo_johnson_skew = _skew(yeo_johnson) if yeo_johnson is not None else None
    yeo_johnson_qq_r = qq_points(yeo_johnson)[4] if yeo_johnson_skew is not None else None
    qq_r = qq_points(values)[4]

    method, skew_after, qq_r_after = None, None, None
    if is_fixed_by(log1p_skew, qq_r, log1p_qq_r):
        method, skew_after, qq_r_after = LOG1P, log1p_skew, log1p_qq_r
    elif is_fixed_by(yeo_johnson_skew, qq_r, yeo_johnson_qq_r):
        method, skew_after, qq_r_after = YEO_JOHNSON, yeo_johnson_skew, yeo_johnson_qq_r

    return TransformCheck(
        column=series.name,
        method=method,
        rows=len(values),
        skew_before=skew,
        qq_r_before=qq_r,
        skew_after=skew_after,
        qq_r_after=qq_r_after,
        log1p_skew=log1p_skew,
        log1p_qq_r=log1p_qq_r,
        yeo_johnson_skew=yeo_johnson_skew,
        yeo_johnson_qq_r=yeo_johnson_qq_r,
        lmbda=float(lmbda) if np.isfinite(lmbda) else None,
    )


def analyze_transforms(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: Optional[str] = None,
    target_type: Optional[str] = None,
) -> TransformReport:
    """
    Check every continuous column, and the target apart from them.

    Args:
        df: The analysed DataFrame.
        column_types: Inferred semantic type per column; only ``numeric_continuous`` is checked.
        target_column: Kept out of the columns and checked on its own, unless it is a class label.
        target_type: ``classification`` or ``regression`` when the target analysis settled it.
    """
    report = TransformReport()
    for column, semantic_type in column_types.items():
        if semantic_type != "numeric_continuous" or column not in df.columns or column == target_column:
            continue
        check = check_transform(df[column])
        if check is not None:
            report.columns[column] = check

    if (
        target_column
        and target_column in df.columns
        and column_types.get(target_column) == "numeric_continuous"
        and target_type != "classification"
    ):
        report.target = check_transform(df[target_column])

    logger.info(
        f"Skewed columns: {len(report.fixed)} fixed by a transformation, {len(report.unfixed)} not"
        + (f"; target {report.target.label or 'not fixed'}" if report.target else "")
    )
    return report
