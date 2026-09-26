"""
Which columns a capped chart shows, and what it leaves out.

The heatmaps stop at `max_correlation_heatmap_size` columns and the chart galleries at
`max_histograms` or `max_boxplots`. They used to take the first columns in the data's order and say
nothing about the rest: on Housing, SalePrice, the 37th of 37 numeric columns, was missing from both
matrices, and the categorical columns most related to it (ExterQual, KitchenQual) got no chart.

Now, when a list is longer than its cap:
- with a target, the target is always shown, and the other places go to the columns most related to
  it, by the effect of the target table (0 to 1, the same for numeric and categorical columns);
- without a target, the first columns in the data's order are kept, as before;
- either way the report names every column left out and the setting that raises the cap.

The shown columns keep the data's order, so each chart sits where it would without a cap.
"""

from dataclasses import dataclass
from typing import Optional

# Each capped chart: the setting that caps it and what its columns are called in the report.
CHARTS = {
    "correlation": ("max_correlation_heatmap_size", "numeric columns"),
    "association": ("max_correlation_heatmap_size", "numeric and categorical columns"),
    "histograms": ("max_histograms", "numeric columns"),
    "boxplots": ("max_boxplots", "numeric columns"),
    "qq": ("max_histograms", "skewed columns"),
    "categorical": ("max_histograms", "categorical columns"),
    "target_categorical": ("max_histograms", "categorical columns"),
    "time_of_day": ("max_histograms", "time-of-day columns"),
}


@dataclass
class ChartColumns:
    shown: list[str]
    left_out: list[str]
    by_target: bool = False  # the shown columns were chosen by their association with the target
    target: Optional[str] = None

    @property
    def capped(self) -> bool:
        return bool(self.left_out)

    @property
    def target_shown(self) -> bool:
        return self.target is not None and self.target in self.shown

    def note(self, noun: str, setting: str) -> str:
        """The sentence under a capped chart: how many, how they were chosen, which are missing, how to see them."""
        shown, total = len(self.shown), len(self.shown) + len(self.left_out)
        if not self.by_target and self.target_shown:
            how = f"{self.target} and the first {shown - 1} others in the data's order"
        elif not self.by_target:
            how = f"the first {shown} in the data's order"
        elif self.target_shown:
            how = f"{self.target} and the {shown - 1} most related to it"
        else:
            how = f"the {shown} most related to {self.target}"
        return (
            f"Showing {shown} of the {total} {noun}: {how}. "
            f"Left out ({len(self.left_out)}): {', '.join(str(name) for name in self.left_out)}. "
            f"To draw more, raise visualizations.{setting} in the configuration file (now {shown})."
        )

    def to_summary(self) -> dict:
        return {"shown": self.shown, "left_out": self.left_out, "chosen_by": "target" if self.by_target else "order"}


def choose_columns(
    columns: list[str],
    limit: int,
    target_column: Optional[str] = None,
    target_effects: Optional[dict[str, float]] = None,
) -> ChartColumns:
    """The columns a chart capped at ``limit`` shows; see the module docstring for the rule."""
    columns = list(columns)
    if len(columns) <= limit:
        return ChartColumns(columns, [], target=target_column)
    if target_column is None:
        return ChartColumns(columns[:limit], columns[limit:])

    effects = target_effects or {}
    pinned = [target_column] if target_column in columns else []
    others = [column for column in columns if column != target_column]
    # Strongest first; a column the target table did not measure goes last, and ties keep the data's
    # order. Without any effect (the target step failed), the target still goes in and the rest keep
    # the data's order.
    ranked = sorted(others, key=lambda column: (-effects.get(column, -1.0), columns.index(column)))
    chosen = set(pinned + ranked[: max(0, limit - len(pinned))])
    return ChartColumns(
        shown=[column for column in columns if column in chosen],
        left_out=[column for column in columns if column not in chosen],
        by_target=bool(effects),
        target=target_column,
    )


def choose_chart_columns(
    df,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config_viz,
    target_column: Optional[str] = None,
    target_type: Optional[str] = None,
    target_effects: Optional[dict[str, float]] = None,
    time_cols: Optional[list[str]] = None,
    transforms=None,
) -> dict[str, ChartColumns]:
    """The columns of every capped chart, keyed as in CHARTS."""
    numeric, categorical = set(numeric_cols), set(categorical_cols)
    lists = {
        "correlation": list(numeric_cols),
        "association": [column for column in df.columns if column in numeric or column in categorical],
        "histograms": list(numeric_cols),
        "boxplots": list(numeric_cols),
        "qq": [check.column for check in transforms.fixed] if transforms else [],
        "categorical": list(categorical_cols),
        "target_categorical": (
            [column for column in categorical_cols if column != target_column]
            if target_column and target_type == "classification"
            else []
        ),
        "time_of_day": list(time_cols or []),
    }
    chosen = {}
    for key, columns in lists.items():
        setting, _ = CHARTS[key]
        # Time columns are not measured against the target, so they keep the data's order.
        target = target_column if key != "time_of_day" else None
        chosen[key] = choose_columns(columns, getattr(config_viz, setting), target, target_effects)
    return chosen
