"""
Multicollinearity: which numeric columns the other numeric columns already explain.

Two measurements, both over the rows that have a value in every column analysed:

- Exact combinations. A column that is a linear combination of others (a total and its parts, the
  one-hot dummy trap, a rescaled copy) makes the correlation matrix singular. Each dependency is a
  direction of that matrix's null space, and is written back as an identity in the columns' own
  units: "TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF".
- The VIF of every column: 1 / (1 - R^2) of that column regressed on all the others. Infinite for a
  column in an exact combination; never negative and never missing, which is what inverting a
  singular matrix used to give.

Then a list of columns to consider dropping: the column each identity writes out, followed by the
column with the highest VIF, recomputed after every drop, until every VIF left is under HIGH_VIF. The
VIF the remaining columns have once those are gone is kept too: a column inside an exact combination
reads infinite until then, and this is the value it really keeps.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

logger = logging.getLogger(__name__)

NUMERIC_TYPES = ("numeric_continuous", "numeric_discrete")

# Share of a column the others leave unexplained (1 - R^2) under which the column counts as an exact
# combination of them. Measured over the 214 numeric columns of the 31 datasets under data/raw: the
# 16 columns in exact identities (Housing, train and test: TotalBsmtSF and GrLivArea are the sums of
# their parts) sit at 8.5e-14 or below, which is float rounding, some of it negative; the next
# closest is 6.9e-06 (AC_POWER against DC_POWER at the solar plant, a physical relation that is
# tight but not exact). 1e-10 lies three orders of magnitude from either side. The same cut applies
# to the eigenvalues of the correlation matrix, which is what finds the dependencies: 1e-15 or below
# for the exact ones, 3.5e-06 for the smallest of the rest.
EXACT_TOLERANCE = 1e-10

# VIF at or above which a column counts as largely explained by the others (R^2 >= 0.9). The
# conventional value, and the measurement agrees with it: over the same 214 columns, nothing falls
# between 9.57 and 17.6.
HIGH_VIF = 10.0

# VIF from which a column is listed in the report (R^2 >= 0.8): the columns nearing the line, not
# only the ones past it.
SHOWN_VIF = 5.0

# (p - 1) / (n - 1) is the R^2 a column reaches against p - 1 unrelated columns by chance alone. Past
# this, the report says that part of every VIF is chance.
CHANCE_R_SQUARED_WARNING = 0.1

# A null-space direction is accurate to about 1e-10, and a real term of an identity weighs about 1
# once the columns are standardised: anything under this is not part of the identity.
_TERM_TOLERANCE = 1e-6

# A column is named as explaining another when its standardised coefficient reaches this share of
# the largest one; at most this many are named.
_PARTNER_SHARE = 0.25
_MAX_PARTNERS = 3


def _number(value: float) -> str:
    """1.8 -> "1.8", 31.9999999997 -> "32": whole numbers as whole numbers, the rest to 4 digits."""
    rounded = round(value)
    if abs(value - rounded) <= 1e-6 * max(1.0, abs(value)):
        return str(int(rounded))
    return f"{value:.4g}"


@dataclass
class ExactDependency:
    """One column written as an exact linear combination of others, in the columns' own units."""

    column: str
    terms: list[tuple[str, float]]
    intercept: float
    rows: int  # rows where it holds
    rows_checked: int  # rows that have a value in all of its columns

    @property
    def columns(self) -> list[str]:
        return [self.column] + [name for name, _ in self.terms]

    def equation(self) -> str:
        """``total = a + b + c``, ``fahrenheit = 1.8*celsius + 32``, ``d_red = 1 - d_blue - d_green``."""
        parts = []
        for name, coefficient in self.terms:
            magnitude = _number(abs(coefficient))
            parts.append((coefficient < 0, name if magnitude == "1" else f"{magnitude}*{name}"))
        if self.intercept:
            constant = (self.intercept < 0, _number(abs(self.intercept)))
            # The constant leads when every term is subtracted ("1 - a - b"), and closes otherwise.
            parts = [constant] + parts if all(negative for negative, _ in parts) else parts + [constant]
        negative, text = parts[0]
        expression = f"-{text}" if negative else text
        for negative, text in parts[1:]:
            expression += f" - {text}" if negative else f" + {text}"
        return f"{self.column} = {expression}"


@dataclass
class DropSuggestion:
    """A column that could go, and the columns that explain it (any of which could go instead)."""

    column: str
    vif: float
    partners: list[str]

    @property
    def exact(self) -> bool:
        return math.isinf(self.vif)


@dataclass
class MulticollinearityReport:
    columns: list[str] = field(default_factory=list)
    rows_used: int = 0
    rows_total: int = 0
    vif: dict[str, float] = field(default_factory=dict)
    partners: dict[str, list[str]] = field(default_factory=dict)
    exact_dependencies: list[ExactDependency] = field(default_factory=list)
    suggested_drops: list[DropSuggestion] = field(default_factory=list)
    # The columns left after the suggested drops, with the VIF they keep among themselves; empty when
    # nothing is dropped.
    vif_after_drops: dict[str, float] = field(default_factory=dict)
    partners_after_drops: dict[str, list[str]] = field(default_factory=dict)
    chance_r_squared: float = 0.0
    target_left_out: Optional[str] = None
    note: str = ""  # why nothing was measured, when nothing was

    def ranked(self, minimum: float = SHOWN_VIF, include_exact: bool = True) -> list[tuple[str, float]]:
        """Columns with a VIF of at least ``minimum``, highest first (exact combinations on top)."""
        listed = [
            (name, value)
            for name, value in self.vif.items()
            if value >= minimum and (include_exact or not math.isinf(value))
        ]
        return sorted(listed, key=lambda item: -item[1])

    def ranked_after_drops(self, minimum: float = SHOWN_VIF) -> list[tuple[str, float]]:
        """Columns left after the suggested drops with a VIF of at least ``minimum``, highest first."""
        listed = [(name, value) for name, value in self.vif_after_drops.items() if value >= minimum]
        return sorted(listed, key=lambda item: -item[1])

    @property
    def exact_columns(self) -> list[str]:
        """Every column that sits in an exact combination, in the data's order."""
        inside = {name for dependency in self.exact_dependencies for name in dependency.columns}
        return [column for column in self.columns if column in inside]

    def to_summary(self) -> dict:
        """The report for summary.json. An infinite VIF becomes null: JSON has no such number."""

        def finite(value: float) -> Optional[float]:
            return None if math.isinf(value) else round(value, 4)

        return {
            "columns": self.columns,
            "rows_used": self.rows_used,
            "rows_total": self.rows_total,
            "target_left_out": self.target_left_out,
            "chance_r_squared": round(self.chance_r_squared, 4),
            "vif": {name: finite(value) for name, value in self.vif.items()},
            "exact_dependencies": [
                {
                    "column": dependency.column,
                    "equation": dependency.equation(),
                    "terms": dict(dependency.terms),
                    "intercept": dependency.intercept,
                    "rows_holding": dependency.rows,
                    "rows_checked": dependency.rows_checked,
                }
                for dependency in self.exact_dependencies
            ],
            "suggested_drops": [
                {"column": drop.column, "vif": finite(drop.vif), "exact": drop.exact, "partners": drop.partners}
                for drop in self.suggested_drops
            ],
            "vif_after_drops": {name: finite(value) for name, value in self.vif_after_drops.items()},
            "note": self.note,
        }


def model_columns(df: pd.DataFrame, column_types: dict[str, str], target_column: Optional[str] = None) -> list[str]:
    """
    The columns a model would read as numbers, in the data's order.

    Numeric columns, plus boolean ones stored as numbers or as True/False: one-hot dummies are typed
    boolean, and leaving them out would hide the dummy trap. Booleans written as text, categories,
    identifiers, dates and free text stay out, and so does the target, which is not a predictor.
    """
    columns = []
    for column, kind in column_types.items():
        if column == target_column or column not in df.columns:
            continue
        series = df[column]
        if kind in NUMERIC_TYPES or (kind == "boolean" and (is_bool_dtype(series) or is_numeric_dtype(series))):
            columns.append(column)
    return columns


def _vif(correlation: np.ndarray, columns: list[str], subset: list[str]) -> tuple[dict, dict]:
    """VIF of each column in ``subset`` against the rest of it, and the columns that explain it most."""
    if len(subset) < 2:
        return {name: 1.0 for name in subset}, {name: [] for name in subset}
    index = [columns.index(name) for name in subset]
    block = correlation[np.ix_(index, index)]
    try:
        inverse = np.linalg.inv(block)
    except np.linalg.LinAlgError:  # only reachable when a dependency sits right at the tolerance
        inverse = np.linalg.pinv(block)

    vif, partners = {}, {}
    for position, name in enumerate(subset):
        value = max(1.0, float(inverse[position, position]))
        vif[name] = value
        if value < SHOWN_VIF:
            partners[name] = []
            continue
        # Standardised coefficients of the regression of this column on the others.
        beta = -inverse[position] / inverse[position, position]
        beta[position] = 0.0
        strongest = float(np.abs(beta).max())
        order = np.argsort(-np.abs(beta))
        partners[name] = [
            subset[k] for k in order if k != position and abs(beta[k]) >= _PARTNER_SHARE * strongest
        ][:_MAX_PARTNERS]
    return vif, partners


def _choose_pivot(row: np.ndarray, involved: list[int], stds: np.ndarray) -> int:
    """
    Which column of an identity to write out in terms of the others.

    A column whose coefficient carries a sign alone against two or more of the other sign is the
    total of the rest ("total = a + b + c"). Otherwise the column with the widest spread, and on a
    tie the one further right in the data, since derived columns tend to be appended.
    """
    positive = [j for j in involved if row[j] > 0]
    negative = [j for j in involved if row[j] < 0]
    for alone, crowd in ((positive, negative), (negative, positive)):
        if len(alone) == 1 and len(crowd) >= 2:
            return alone[0]
    return max(involved, key=lambda j: (stds[j], j))


def _identity(column: str, others: list[str], data: pd.DataFrame, df: pd.DataFrame) -> ExactDependency:
    """
    Fit ``column`` on ``others`` over the rows the dependency was found on, then count on how many
    of the rows holding all those columns it holds.
    """
    design = np.column_stack([np.ones(len(data)), data[others].to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, data[column].to_numpy(), rcond=None)
    intercept, slopes = float(coefficients[0]), [float(value) for value in coefficients[1:]]

    checked = df[[column] + others].astype("float64").dropna()
    target = checked[column].to_numpy()
    scale = max(1.0, float(np.abs(target).mean())) if len(target) else 1.0
    if abs(intercept) <= 1e-9 * scale:
        intercept = 0.0
    predicted = intercept + checked[others].to_numpy() @ np.array(slopes)
    holding = int((np.abs(target - predicted) <= 1e-9 * scale).sum())
    return ExactDependency(column, list(zip(others, slopes)), intercept, rows=holding, rows_checked=len(checked))


def _exact_dependencies(null_space: np.ndarray, data: pd.DataFrame, df: pd.DataFrame) -> list[ExactDependency]:
    """
    One identity per direction of the null space.

    Gauss-Jordan elimination over the null-space basis gives every direction a pivot column of its
    own, absent from the others: two unrelated identities come out apart, and dropping the pivots
    removes every exact dependency at once.
    """
    names = list(data.columns)
    stds = data.std().to_numpy()
    basis = null_space.copy()
    pivots: list[tuple[int, int]] = []
    for r in range(basis.shape[0]):
        row = basis[r]
        scale = float(np.abs(row).max())
        if scale == 0.0:
            continue
        involved = [j for j in range(len(names)) if abs(row[j]) > _TERM_TOLERANCE * scale]
        pivot = _choose_pivot(row, involved, stds)
        basis[r] = row / row[pivot]
        for other in range(basis.shape[0]):
            if other != r:
                basis[other] -= basis[other, pivot] * basis[r]
        pivots.append((r, pivot))

    dependencies = []
    for r, pivot in pivots:
        others = [names[j] for j in range(len(names)) if j != pivot and abs(basis[r, j]) > _TERM_TOLERANCE]
        if others:
            dependencies.append(_identity(names[pivot], others, data, df))
    # The null-space basis comes out in whatever order the eigensolver gives; the data's order does not move.
    return sorted(dependencies, key=lambda dependency: names.index(dependency.column))


def analyze_multicollinearity(
    df: pd.DataFrame, column_types: dict[str, str], target_column: Optional[str] = None
) -> MulticollinearityReport:
    """Exact combinations, the VIF of every column, and the columns to consider dropping."""
    candidates = model_columns(df, column_types, target_column)
    report = MulticollinearityReport(
        rows_total=len(df), target_left_out=target_column if target_column in df.columns else None
    )
    data = df[candidates].astype("float64").dropna()
    columns = [column for column in candidates if data[column].nunique() > 1]
    data = data[columns]
    rows, width = data.shape
    report.columns, report.rows_used = columns, rows

    if width < 2:
        report.note = "Fewer than two numeric columns: there is nothing to compare."
        return report
    if rows <= width + 1:
        report.note = (
            f"Only {rows} rows have a value in all {width} numeric columns. With no more rows than "
            "columns, any column is an exact combination of the others by chance, so nothing is measured."
        )
        return report
    report.chance_r_squared = (width - 1) / (rows - 1)

    standardised = ((data - data.mean()) / data.std()).to_numpy()
    correlation = standardised.T @ standardised / (rows - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    null_space = eigenvectors[:, eigenvalues < EXACT_TOLERANCE].T
    dependencies = _exact_dependencies(null_space, data, df)

    # Every column outside the written-out ones keeps the VIF it has among all the columns: each
    # written-out column is a combination of the kept ones, so dropping it leaves their span intact.
    written_out = {dependency.column for dependency in dependencies}
    kept = [column for column in columns if column not in written_out]
    vif, partners = _vif(correlation, columns, kept)
    # A column inside an exact combination is explained by the other columns of that combination
    # (of each of them, when it sits in several).
    exact_partners: dict[str, list[str]] = {}
    for dependency in dependencies:
        for name in dependency.columns:
            others = [other for other in dependency.columns if other != name]
            exact_partners[name] = list(dict.fromkeys(exact_partners.get(name, []) + others))
    for name, others in exact_partners.items():
        vif[name] = math.inf
        partners[name] = others
    report.vif = {column: vif[column] for column in columns}
    report.partners = {column: partners.get(column, []) for column in columns}
    report.exact_dependencies = dependencies

    drops = [
        DropSuggestion(dependency.column, math.inf, [name for name, _ in dependency.terms])
        for dependency in dependencies
    ]
    current = list(kept)
    while len(current) >= 2:
        step_vif, step_partners = _vif(correlation, columns, current)
        worst = max(current, key=lambda name: (step_vif[name], columns.index(name)))
        if step_vif[worst] < HIGH_VIF:
            break
        drops.append(DropSuggestion(worst, step_vif[worst], step_partners[worst]))
        current.remove(worst)
    report.suggested_drops = drops
    if drops:
        report.vif_after_drops, report.partners_after_drops = _vif(correlation, columns, current)

    logger.info(
        f"Multicollinearity: {len(dependencies)} exact combination(s), "
        f"{sum(1 for value in report.vif.values() if value >= HIGH_VIF)} column(s) with VIF >= {HIGH_VIF:g}"
    )
    return report
