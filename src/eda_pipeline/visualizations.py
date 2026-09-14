"""
Visualization generation: plots saved as PNG with proper handling of edge cases.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Headless backend
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning

from .relationships import correlation_ratio
from .type_inference import coerce_to_datetime, parse_time_of_day

logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 9


@contextmanager
def _suppress_seaborn_boxplot_warning():
    """
    Seaborn 0.13.x internally calls Matplotlib's ``Axes.bxp()`` with the
    ``vert`` keyword, which Matplotlib >=3.11 deprecates in favor of
    ``orientation=``. There is no public ``sns.boxplot()`` argument that
    avoids this internal call (fixed in later seaborn releases), so the
    specific warning is filtered narrowly here instead of leaking into every
    run's output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*\bvert\b.*", category=MatplotlibDeprecationWarning)
        yield


@contextmanager
def _suppress_seaborn_heatmap_warning():
    """
    Seaborn 0.13.x internally calls ``Colormap.set_bad()`` when rendering
    ``sns.heatmap()``, which recent Matplotlib versions flag as a pending
    deprecation. No public ``sns.heatmap()`` argument avoids this internal
    call, so the specific warning is filtered narrowly here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*set_bad.*", category=PendingDeprecationWarning)
        yield


def safe_plot(func):
    """Decorator to safely handle plot errors."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return None
        finally:
            plt.close("all")

    return wrapper


@safe_plot
def plot_histogram(series: pd.Series, output_path: Path) -> bool:
    """Plot histogram with KDE."""
    valid = series.dropna()
    if len(valid) < 2:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(valid.unique()) > 1:
        sns.histplot(valid, kde=True, ax=ax, bins=30)
    else:
        ax.hist(valid, bins=1)
    ax.set_title(f"Distribución: {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_boxplot(series: pd.Series, output_path: Path) -> bool:
    """Plot boxplot."""
    valid = series.dropna()
    if len(valid) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    with _suppress_seaborn_boxplot_warning():
        sns.boxplot(y=valid, ax=ax)
    ax.set_title(f"Boxplot: {series.name}")
    ax.set_ylabel(series.name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_categorical(series: pd.Series, output_path: Path, top_n: int = 20) -> bool:
    """Plot categorical bar chart (top N + Others)."""
    valid = series.dropna()
    if len(valid) == 0:
        return False

    counts = valid.value_counts()
    if len(counts) > top_n:
        top_counts = counts.iloc[:top_n]
        other_count = counts.iloc[top_n:].sum()
        top_counts["Otros"] = other_count
        counts = top_counts

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Categorías: {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


# A pie shows part-to-whole at a glance only while its slices can be told apart: about six at most.
# Measured over the 19 datasets in data/raw, 66 of the 101 categorical columns have five categories or
# fewer, none has six, and the next ones hold 7 to 10 slices of similar size (año in the mortgage
# data: ten slices of about 10% each), which only the bar chart can still show.
PIE_MAX_CATEGORIES = 6

# Slices below this share get no label inside the pie, where it would sit on its neighbours'
# (discount_pct 0.15 is 0.7% of Order_Details). The legend lists every category with its share.
PIE_MIN_LABELLED_SHARE = 5.0


def _category_counts(series: pd.Series) -> pd.Series:
    """Rows per category, most frequent first, missing values left out."""
    return series.dropna().value_counts()


def pie_chart_shares(series: pd.Series) -> pd.Series | None:
    """
    Percentage of rows per category, most frequent first, over the rows that have a value.

    None when a pie would not work: a single category, or more than PIE_MAX_CATEGORIES.
    """
    counts = _category_counts(series)
    if len(counts) < 2 or len(counts) > PIE_MAX_CATEGORIES:
        return None
    return counts / counts.sum() * 100


def format_share(pct: float) -> str:
    """A share as text; a share that would round to 0.0 reads "<0.1 %", so it never looks empty."""
    if 0 < pct < 0.1:
        return "<0.1 %"
    return f"{pct:.1f} %"


def pie_slice_label(pct: float) -> str:
    """The label drawn inside a slice: its share, or nothing for a slice too thin to hold it."""
    return format_share(pct) if pct >= PIE_MIN_LABELLED_SHARE else ""


def _readable_text_color(facecolor) -> str:
    """White or near-black, whichever contrasts more with the slice it sits on."""
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in facecolor[:3])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if (1.05 / (luminance + 0.05)) >= ((luminance + 0.05) / 0.05) else "#222222"


@safe_plot
def plot_pie_chart(series: pd.Series, output_path: Path) -> bool:
    """
    Pie of a categorical column in percentages: each large slice carries its share, and the legend
    names every category with its share and row count, so no slice is identified by colour alone.
    """
    shares = pie_chart_shares(series)
    if shares is None:
        return False
    counts = _category_counts(series)

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, labels = ax.pie(
        shares.values,
        startangle=90,
        counterclock=False,  # clockwise from the top, largest slice first
        autopct=pie_slice_label,
        pctdistance=0.7,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    for wedge, label in zip(wedges, labels):
        label.set_color(_readable_text_color(wedge.get_facecolor()))

    ax.legend(
        wedges,
        [f"{category}: {format_share(pct)} ({counts[category]})" for category, pct in shares.items()],
        title=str(series.name),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    ax.set_title(f"Proporción: {series.name} (n = {int(counts.sum())})")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path, max_size: int = 30) -> bool:
    """Plot correlation heatmap."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return False

    # Limit size for readability
    if corr_matrix.shape[0] > max_size:
        corr_matrix = corr_matrix.iloc[:max_size, :max_size]

    fig, ax = plt.subplots(figsize=(12, 10))
    with _suppress_seaborn_heatmap_warning():
        sns.heatmap(
            corr_matrix, annot=False, cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"label": "Correlación"}
        )
    ax.set_title("Matriz de Correlación Pearson")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_association_heatmap(assoc_matrix: pd.DataFrame, output_path: Path, max_size: int = 30) -> bool:
    """
    Heatmap over numeric and categorical columns alike (see relationships.association_matrix).

    Values are annotated while the matrix stays readable, because the numbers are the point here:
    a 0.68 between a category and a number says far more than its colour does.
    """
    if assoc_matrix.empty or assoc_matrix.shape[0] < 2:
        return False

    if assoc_matrix.shape[0] > max_size:
        assoc_matrix = assoc_matrix.iloc[:max_size, :max_size]

    size = assoc_matrix.shape[0]
    side = min(20, max(9, size))
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    with _suppress_seaborn_heatmap_warning():
        sns.heatmap(
            assoc_matrix,
            annot=size <= 15,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            square=True,
            cbar_kws={"label": "Asociación"},
        )
    ax.set_title("Asociación entre Variables (Pearson · Cramér's V · eta)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_missing_matrix(df: pd.DataFrame, output_path: Path) -> bool:
    """Plot missing value heatmap."""
    if df.empty:
        return False

    # Sample if too large
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    missing_matrix = df.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(14, 8))
    with _suppress_seaborn_heatmap_warning():
        sns.heatmap(missing_matrix, cbar=True, ax=ax, yticklabels=False, cbar_kws={"label": "Faltante (1=sí)"})
    ax.set_title("Mapa de Valores Faltantes")
    ax.set_xlabel("Columnas")
    ax.set_ylabel("Filas (muestra)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


def top_correlated_pairs(corr_matrix: pd.DataFrame, limit: int) -> list[tuple[str, str, float]]:
    """
    The `limit` column pairs with the strongest Pearson correlation, strongest first by |r|, sign kept.

    Pairs whose correlation is undefined (a constant column) are left out. The scatter plots used to
    take the first pairs in column order instead: on penguins_lter the strongest pair (|r| = 0.87) was
    not drawn, and on credito_asegurado neither was a pair at |r| = 1.00.
    """
    if corr_matrix.empty:
        return []
    columns = list(corr_matrix.columns)
    pairs = [
        (first, second, float(corr_matrix.loc[first, second]))
        for i, first in enumerate(columns)
        for second in columns[i + 1 :]
        if pd.notna(corr_matrix.loc[first, second])
    ]
    pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)
    return pairs[:limit]


@safe_plot
def plot_scatter(
    x: pd.Series, y: pd.Series, output_path: Path, sample_size: int = 1000, r: float | None = None
) -> bool:
    """Plot scatter plot, with the pair's Pearson correlation in the title when it is known."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return False

    # Sample if large
    if len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid["x"], valid["y"], alpha=0.6, s=20)
    ax.set_title(f"Scatter: {x.name} vs {y.name}" + (f" (r = {r:.2f})" if r is not None else ""))
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


# A time series shows at most this many points: the finest of day, week, month or year that fits.
# On the datasets in data/raw that gives days for a 16-day span, weeks for a year (Sales_Receipts),
# weeks for the two seasons of Date Egg and months for the five years of customer_since.
TIME_SERIES_MAX_PERIODS = 120
_TIME_SERIES_PERIODS = (("D", "día", 1.0), ("W", "semana", 7.0), ("M", "mes", 30.44), ("Y", "año", 365.25))


def rows_per_period(dates: pd.Series) -> tuple[pd.Series, str]:
    """
    Rows per period over the whole span of `dates`, empty periods included as zero, and the period's
    Spanish name. The period is the finest one that keeps the series within TIME_SERIES_MAX_PERIODS.
    """
    span_days = (dates.max() - dates.min()).days
    for freq, label, days in _TIME_SERIES_PERIODS:
        if span_days / days + 1 <= TIME_SERIES_MAX_PERIODS or freq == "Y":
            break
    periods = dates.dt.to_period(freq)
    every_period = pd.period_range(periods.min(), periods.max(), freq=freq)
    return periods.value_counts().reindex(every_period, fill_value=0), label


@safe_plot
def plot_time_series(series: pd.Series, output_path: Path) -> bool:
    """
    Rows per day, week, month or year: when the records happen, with the empty periods at zero.

    It used to plot the row number against the date itself, which for dates stored as text drew
    one axis label per distinct day (366 of them on Sales_Receipts) and said nothing about time.
    """
    dates = series.dropna()
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = coerce_to_datetime(dates)
    dates = dates.dropna()
    if dates.nunique() < 2:
        return False
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    counts, period = rows_per_period(dates)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(counts.index.to_timestamp(), counts.values, linewidth=1.5)
    ax.set_title(f"Registros por {period}: {series.name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Filas")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_time_of_day(series: pd.Series, output_path: Path) -> bool:
    """One bar per hour of the day, so the empty hours show as clearly as the busy ones."""
    hours = parse_time_of_day(series.dropna()).dropna().dt.hour
    if hours.empty:
        return False

    counts = hours.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts.index, counts.values)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{hour:02d}" for hour in range(24)])
    ax.set_title(f"Hora del día: {series.name}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_target_distribution(series: pd.Series, output_path: Path) -> bool:
    """Plot target variable distribution."""
    valid = series.dropna()
    if len(valid) == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(series) and len(valid.unique()) > 20:
        # Continuous target: histogram
        sns.histplot(valid, kde=True, ax=ax, bins=30)
    else:
        # Categorical or discrete target: bar chart
        counts = valid.value_counts()
        counts.plot(kind="bar", ax=ax)
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(f"Distribución del Target: {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


@safe_plot
def plot_target_vs_feature(
    feature: pd.Series, target: pd.Series, output_path: Path, feature_type: str, target_type: str
) -> bool:
    """Plot relationship between feature and target."""
    valid = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if len(valid) < 2:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    if target_type == "classification":
        if feature_type == "numeric":
            # Boxplot: target vs feature
            with _suppress_seaborn_boxplot_warning():
                sns.boxplot(data=valid, x="target", y="feature", ax=ax)
        else:
            # Grouped bar chart
            ct = pd.crosstab(valid["feature"], valid["target"])
            ct.plot(kind="bar", ax=ax)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Target")
    else:
        # Regression target
        if feature_type == "numeric":
            # Scatter plot
            ax.scatter(valid["feature"], valid["target"], alpha=0.6, s=20)
        else:
            # Boxplot
            with _suppress_seaborn_boxplot_warning():
                sns.boxplot(data=valid, x="feature", y="target", ax=ax)
            ax.tick_params(axis="x", rotation=45)

    ax.set_title(f"{feature.name} vs {target.name}")
    ax.set_xlabel(feature.name)
    ax.set_ylabel(target.name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


# Nine categories plus «Otros» is one bar per colour of the default cycle; a tenth would
# repeat the first colour and make two different categories look like the same one.
TARGET_BAR_MAX_CATEGORIES = 9
TARGET_BAR_MAX_CLASSES = 12
TARGET_BAR_MAX_LABELLED = 24
_OTHERS_LABEL = "Otros"


def target_vs_categorical_counts(
    feature: pd.Series, target: pd.Series, max_categories: int = TARGET_BAR_MAX_CATEGORIES
) -> pd.DataFrame:
    """
    Counts per target class (rows) and feature category (columns), ready to plot as grouped bars.

    Rows missing either value are dropped, so the total equals what the chart shows and its
    percentages add up to 100. Categories beyond ``max_categories`` are folded into «Otros»
    instead of being dropped: a column such as `tipo_empleo` (21 values in the mortgage data)
    would otherwise turn the legend into a wall of colours.
    """
    valid = pd.DataFrame({"feature": feature, "target": target}).dropna()
    if valid.empty:
        return pd.DataFrame()

    valid["feature"] = valid["feature"].astype(str)
    valid["target"] = valid["target"].astype(str)

    ranking = valid["feature"].value_counts()
    if len(ranking) > max_categories:
        kept = list(ranking.index[:max_categories])
        valid["feature"] = valid["feature"].where(valid["feature"].isin(kept), _OTHERS_LABEL)
        order = kept + [_OTHERS_LABEL]
    else:
        order = list(ranking.index)

    counts = pd.crosstab(valid["target"], valid["feature"])
    return counts.reindex(index=_sorted_labels(counts.index), columns=order, fill_value=0)


def _sorted_labels(labels) -> list[str]:
    """Sort the target classes the way a reader expects: 2 before 10 when they are numbers."""
    values = list(labels)
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values)


@safe_plot
def plot_target_vs_categorical(
    feature: pd.Series, target: pd.Series, output_path: Path, max_categories: int = TARGET_BAR_MAX_CATEGORIES
) -> bool:
    """
    Grouped bars of one categorical variable against the target, labelled with percentages.

    The labels are shares of the plotted total, so the whole chart adds up to 100%: that is what
    makes two classes of an imbalanced target comparable at a glance.
    """
    counts = target_vs_categorical_counts(feature, target, max_categories)
    if counts.empty or counts.shape[0] < 2 or counts.shape[1] < 2:
        # Nothing to compare: a single target class, or a feature with one value.
        return False
    if counts.shape[0] > TARGET_BAR_MAX_CLASSES:
        logger.debug(f"Skipping grouped bars for {feature.name}: the target has {counts.shape[0]} classes")
        return False

    total = int(counts.to_numpy().sum())
    n_bars = counts.shape[0] * counts.shape[1]
    width = max(8, 1.1 * n_bars)
    fig, ax = plt.subplots(figsize=(min(20, width), 6))
    counts.plot(kind="bar", ax=ax, width=0.8)

    # Past a certain number of bars the percentages overlap each other and hide the chart,
    # the same reason the association heatmap only annotates while it stays readable.
    if n_bars <= TARGET_BAR_MAX_LABELLED:
        for container in ax.containers:
            labels = [f"{bar.get_height() / total * 100:.2f} %" for bar in container]
            ax.bar_label(container, labels=labels, fontsize=8, padding=2)

    ax.set_title(f"{feature.name} vs {target.name}")
    ax.set_xlabel(target.name)
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=feature.name)
    ax.margins(y=0.12)  # room for the labels above the tallest bar
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


# A pair plot draws every pair of the chosen columns, so its panels grow with the square of them: six
# columns are 36 panels and take about 8 s on penguins_lter; past that, a panel is too small to read.
# Its cost grows with the columns, not the rows (3 columns: 2.4 s for 1 000 rows, 3.4 s for 5 000), so
# a random sample of rows keeps large tables fast without losing the shape of the cloud.
PAIR_PLOT_MIN_COLUMNS = 3
PAIR_PLOT_MAX_COLUMNS = 6
PAIR_PLOT_MAX_ROWS = 2000

# The pair plot is coloured by a group only when that group actually separates the variables: a mean
# correlation ratio (eta) of at least 0.25, Cohen's medium effect (eta squared about 0.06). Measured
# over the 19 datasets in data/raw, that colours penguins by species (0.81), stroke by work_type (0.42)
# and leaves uncoloured the groupings that separate nothing, such as product_line in sales (0.07).
PAIR_PLOT_MIN_HUE_ETA = 0.25
PAIR_PLOT_MAX_GROUPS = PIE_MAX_CATEGORIES
PAIR_PLOT_MIN_GROUP_ROWS = 10  # a smaller group shows no pattern and cannot draw a distribution


@dataclass
class PairPlotSpec:
    """What the pair plot draws and why, reported next to the chart and in summary.json."""

    columns: list[str]
    hue: str | None
    hue_reason: str  # "target", "grupo" or "" when uncoloured
    hue_eta: float | None  # mean eta of the colouring group with the columns
    best_group: str | None  # the eligible grouping that separated the columns most, used or not
    best_eta: float | None
    rows_available: int  # complete rows for the columns (and the group)
    rows_plotted: int
    max_rows: int


def pair_plot_rows(df: pd.DataFrame, columns: list[str], hue: str | None, max_rows: int) -> pd.DataFrame:
    """The rows a pair plot draws: complete for its columns and group, and a fixed random sample above max_rows."""
    keep = columns + ([hue] if hue else [])
    rows = df[keep].dropna()
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=42)
    return rows


def _pair_plot_columns(df: pd.DataFrame, continuous: list[str]) -> list[str]:
    """Up to PAIR_PLOT_MAX_COLUMNS columns, those in the strongest correlations first, kept in dataset order."""
    if len(continuous) <= PAIR_PLOT_MAX_COLUMNS:
        return continuous
    chosen: list[str] = []
    corr = df[continuous].astype(float).corr()
    for first, second, _ in top_correlated_pairs(corr, limit=len(continuous) ** 2):
        for col in (first, second):
            if col not in chosen and len(chosen) < PAIR_PLOT_MAX_COLUMNS:
                chosen.append(col)
    for col in continuous:  # columns whose correlations are all undefined
        if col not in chosen and len(chosen) < PAIR_PLOT_MAX_COLUMNS:
            chosen.append(col)
    return [col for col in continuous if col in chosen]


def _mean_eta(df: pd.DataFrame, group: str, columns: list[str]) -> float | None:
    etas = [correlation_ratio(df[group], df[col].astype(float)) for col in columns]
    etas = [float(eta) for eta in etas if pd.notna(eta)]
    return sum(etas) / len(etas) if etas else None


def _usable_group(df: pd.DataFrame, columns: list[str], group: str, max_rows: int) -> bool:
    """2 to PAIR_PLOT_MAX_GROUPS groups, each with at least PAIR_PLOT_MIN_GROUP_ROWS rows in what is drawn."""
    sizes = pair_plot_rows(df, columns, group, max_rows)[group].value_counts()
    return 2 <= len(sizes) <= PAIR_PLOT_MAX_GROUPS and int(sizes.min()) >= PAIR_PLOT_MIN_GROUP_ROWS


def choose_pair_plot(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None = None,
    target_type: str | None = None,
    max_rows: int = PAIR_PLOT_MAX_ROWS,
) -> PairPlotSpec | None:
    """
    Decide the pair plot's columns and colouring group, or None when there are fewer than 3 continuous columns.

    The columns are the continuous numeric ones; discrete ones fall in stripes and work better as groups.
    A classification target colours the plot whenever it can; otherwise the categorical column that best
    separates the columns does, if it separates them at least by PAIR_PLOT_MIN_HUE_ETA.
    """
    continuous = [c for c, t in column_types.items() if t == "numeric_continuous" and c in df.columns]
    if len(continuous) < PAIR_PLOT_MIN_COLUMNS:
        return None
    columns = _pair_plot_columns(df, continuous)

    candidates = []
    for col, semantic_type in column_types.items():
        if col not in df.columns or col in columns:
            continue
        if semantic_type not in ("categorical", "boolean", "numeric_discrete"):
            continue
        if not _usable_group(df, columns, col, max_rows):
            continue
        eta = _mean_eta(df, col, columns)
        if eta is not None:
            candidates.append((eta, col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_eta, best_group = candidates[0] if candidates else (None, None)

    hue, reason, hue_eta = None, "", None
    if (
        target_type == "classification"
        and target_column in df.columns
        and target_column not in columns
        and _usable_group(df, columns, target_column, max_rows)
    ):
        hue, reason, hue_eta = target_column, "target", _mean_eta(df, target_column, columns)
    elif best_eta is not None and best_eta >= PAIR_PLOT_MIN_HUE_ETA:
        hue, reason, hue_eta = best_group, "grupo", best_eta

    return PairPlotSpec(
        columns=columns,
        hue=hue,
        hue_reason=reason,
        hue_eta=hue_eta,
        best_group=best_group,
        best_eta=best_eta,
        rows_available=len(df[columns + ([hue] if hue else [])].dropna()),
        rows_plotted=len(pair_plot_rows(df, columns, hue, max_rows)),
        max_rows=max_rows,
    )


@safe_plot
def plot_pair_plot(df: pd.DataFrame, spec: PairPlotSpec | None, output_path: Path) -> bool:
    """Every pair of the chosen columns, and each column's distribution on the diagonal, per group when coloured."""
    if spec is None:
        return False
    rows = pair_plot_rows(df, spec.columns, spec.hue, spec.max_rows)
    if len(rows) < 2:
        return False

    data = rows[spec.columns].astype(float)
    options = {"vars": spec.columns, "plot_kws": {"s": 12, "alpha": 0.6, "edgecolor": "none"}}
    if spec.hue:
        data[spec.hue] = rows[spec.hue].astype(str)
        options |= {
            "hue": spec.hue,
            "hue_order": list(data[spec.hue].value_counts().index),
            "diag_kind": "kde",
            # Each group's curve integrates to 1, so shapes compare even when one group is small: with a
            # shared scale the 5% of stroke cases draw a flat line under the other 95%.
            "diag_kws": {"warn_singular": False, "common_norm": False},
        }
    else:
        options["diag_kind"] = "hist"

    height = 2.2 if len(spec.columns) > 4 else 2.8
    grid = sns.pairplot(data, height=height, **options)
    title = "Pair Plot" + (f": por {spec.hue}" if spec.hue else "")
    grid.figure.suptitle(title, y=1.02)
    grid.savefig(output_path, dpi=100, bbox_inches="tight")
    return True


def generate_all_visualizations(
    df: pd.DataFrame,
    column_types: dict[str, str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    text_cols: list[str],
    corr_matrix: pd.DataFrame,
    output_dir: Path,
    config_viz,
    assoc_matrix: pd.DataFrame | None = None,
    target_column: str | None = None,
    target_type: str | None = None,
    time_cols: list[str] | None = None,
    pair_plot: PairPlotSpec | None = None,
) -> dict[str, list[str]]:
    """
    Generate all standard visualizations.

    Returns:
        Dictionary mapping plot type to list of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = {}

    # Numeric columns: histograms and boxplots
    logger.info(f"Generating {len(numeric_cols)} histograms...")
    hist_files = []
    for col in numeric_cols[: config_viz.max_histograms]:
        output_file = output_dir / f"histogram_{col.replace('/', '_')}.png"
        if plot_histogram(df[col], output_file):
            hist_files.append(str(output_file))
    plot_files["histograms"] = hist_files

    logger.info(f"Generating {len(numeric_cols)} boxplots...")
    box_files = []
    for col in numeric_cols[: config_viz.max_boxplots]:
        output_file = output_dir / f"boxplot_{col.replace('/', '_')}.png"
        if plot_boxplot(df[col], output_file):
            box_files.append(str(output_file))
    plot_files["boxplots"] = box_files

    # Categorical columns
    logger.info(f"Generating {len(categorical_cols)} bar charts...")
    cat_files = []
    for col in categorical_cols[: config_viz.max_histograms]:
        output_file = output_dir / f"categorical_{col.replace('/', '_')}.png"
        if plot_categorical(df[col], output_file):
            cat_files.append(str(output_file))
    plot_files["categorical"] = cat_files

    # The same columns as pies, in percentages, while they have few enough categories
    pie_files = []
    for col in categorical_cols[: config_viz.max_histograms]:
        output_file = output_dir / f"pie_{col.replace('/', '_')}.png"
        if plot_pie_chart(df[col], output_file):
            pie_files.append(str(output_file))
    plot_files["pie"] = pie_files

    # Target against each categorical variable (only makes sense for a classification target)
    target_bar_files = []
    if target_column and target_type == "classification" and target_column in df.columns:
        features = [col for col in categorical_cols if col != target_column]
        logger.info(f"Generating {len(features)} grouped bar charts against '{target_column}'...")
        for col in features[: config_viz.max_histograms]:
            output_file = output_dir / f"target_bars_{col.replace('/', '_')}.png"
            if plot_target_vs_categorical(df[col], df[target_column], output_file):
                target_bar_files.append(str(output_file))
    plot_files["target_categorical"] = target_bar_files

    # Correlation heatmap
    logger.info("Generating correlation heatmap...")
    corr_file = output_dir / "correlation_heatmap.png"
    if plot_correlation_heatmap(corr_matrix, corr_file, config_viz.max_correlation_heatmap_size):
        plot_files["correlation"] = [str(corr_file)]
    else:
        plot_files["correlation"] = []

    # The same picture, but including the categorical columns.
    assoc_file = output_dir / "association_heatmap.png"
    if assoc_matrix is not None and plot_association_heatmap(
        assoc_matrix, assoc_file, config_viz.max_correlation_heatmap_size
    ):
        plot_files["association"] = [str(assoc_file)]
    else:
        plot_files["association"] = []

    # Missing value matrix
    logger.info("Generating missing value matrix...")
    missing_file = output_dir / "missing_matrix.png"
    if plot_missing_matrix(df, missing_file):
        plot_files["missing"] = [str(missing_file)]
    else:
        plot_files["missing"] = []

    # Scatter plots for the most correlated pairs
    logger.info("Generating scatter plots...")
    scatter_files = []
    if corr_matrix.empty and len(numeric_cols) >= 2:
        # The relationships step failed or was skipped; rank the pairs here rather than draw none.
        corr_matrix = df[numeric_cols].astype(float).corr()
    for col1, col2, r in top_correlated_pairs(corr_matrix, config_viz.max_scatter_pairs):
        output_file = output_dir / f"scatter_{col1.replace('/', '_')}_vs_{col2.replace('/', '_')}.png"
        if plot_scatter(df[col1], df[col2], output_file, r=r):
            scatter_files.append(str(output_file))

    plot_files["scatter"] = scatter_files

    # Pair plot of the numeric variables, coloured by a group when one separates them (see choose_pair_plot)
    pair_file = output_dir / "pair_plot.png"
    plot_files["pair_plot"] = [str(pair_file)] if plot_pair_plot(df, pair_plot, pair_file) else []

    # Time series plots
    logger.info(f"Generating {len(datetime_cols)} time series plots...")
    ts_files = []
    for col in datetime_cols[:5]:  # Limit to 5
        output_file = output_dir / f"timeseries_{col.replace('/', '_')}.png"
        if plot_time_series(df[col], output_file):
            ts_files.append(str(output_file))
    plot_files["timeseries"] = ts_files

    # Times of day: how the rows spread over the 24 hours
    time_files = []
    for col in (time_cols or [])[: config_viz.max_histograms]:
        output_file = output_dir / f"time_of_day_{col.replace('/', '_')}.png"
        if plot_time_of_day(df[col], output_file):
            time_files.append(str(output_file))
    plot_files["time_of_day"] = time_files

    logger.info(f"Generated {sum(len(v) for v in plot_files.values())} visualizations")
    return plot_files
