"""
HTML report generation: self-contained, offline-ready report.
"""

import base64
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Template
from PIL import Image

from .chart_columns import CHARTS
from .multicollinearity import CHANCE_R_SQUARED_WARNING, HIGH_VIF, SHOWN_VIF
from .recommendations import recommendations_by_step
from .transforms import FIXED_SKEW, MAX_FLOOR_PCT, MIN_DISTINCT, MIN_QQ_GAIN, MIN_SKEW, describe_attempts
from .type_inference import dtype_label, semantic_type_label
from .visualizations import (
    FLOOR_SPLIT_MIN_DISTINCT,
    FLOOR_SPLIT_MIN_SHARE,
    PAIR_PLOT_MIN_HUE_ETA,
    PIE_MAX_CATEGORIES,
)

logger = logging.getLogger(__name__)

SEVERITIES = ("critical", "high", "medium", "low")

# Alerts of these kinds name one column each and share their advice, so a report lists them as one line
# with a chip per column ("High cardinality · 10 columns"). Every other kind (duplicates, empty rows,
# a constant-columns list, leakage, class imbalance) keeps its own line and its full message.
ALERT_GROUP_TITLES = {
    "missing": "Missing values",
    "quasi_constant": "Almost constant",
    "high_cardinality": "High cardinality",
    "punctuation_missing": "Punctuation read as missing",
    "mixed_types": "Mixed types",
    "numbers_as_text": "Numbers kept as text",
}


def alert_counts(alerts) -> dict[str, int]:
    """How many alerts of each severity, every severity present even at 0."""
    return {severity: sum(1 for alert in alerts if alert.severity == severity) for severity in SEVERITIES}


def alert_groups(alerts) -> list[dict]:
    """
    The alerts as the report lists them: one group per severity, most severe first, in which alerts of
    one groupable kind and the same advice share a row that lists their columns and figures.
    """
    groups = []
    for severity in SEVERITIES:
        rows, by_key = [], {}
        for alert in (a for a in alerts if a.severity == severity):
            kind = getattr(alert, "kind", None)
            if kind in ALERT_GROUP_TITLES and alert.column is not None:
                key = (kind, alert.recommendation)
                if key not in by_key:
                    by_key[key] = {
                        "title": ALERT_GROUP_TITLES[kind],
                        "columns": [],
                        "recommendation": alert.recommendation,
                    }
                    rows.append(by_key[key])
                by_key[key]["columns"].append((alert.column, getattr(alert, "detail", None)))
            else:
                rows.append({"title": alert.message, "columns": [], "recommendation": alert.recommendation})
        if rows:
            groups.append(
                {"severity": severity, "count": sum(1 for a in alerts if a.severity == severity), "rows": rows}
            )
    return groups


def format_measure(value: object, decimals: int = 3) -> str:
    """
    A measured number for a report cell, or "N/A" when there is none.

    A p-value or an effect size of exactly 0 is a result: the template used to test the value for
    truth, so VIP's mutual information of 0.0 was shown as if it had not been measured.
    """
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number != number:  # NaN
        return "N/A"
    return f"{number:.{decimals}f}"


def format_p_value(value: object) -> str:
    """
    A p-value for a report cell.

    Anything under 0.0001 is shown as "<0.0001": rounding it to 0.0000 reads as certainty, and a
    p-value that reaches exactly 0 only underflowed. The template escapes the < like any other
    value, so it reaches the page as text.
    """
    text = format_measure(value, 4)
    if text == "N/A":
        return text
    return "<0.0001" if float(value) < 0.0001 else text


def format_vif(value: float) -> str:
    """A VIF for a report cell: ∞ for an exact combination, and no exponent however large it gets."""
    if math.isinf(value):
        return "∞"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def python_list_snippet(variable: str, columns) -> str:
    """
    ``num = ["Age","Fare"]``: column names as one line of Python, ready to paste into a notebook.

    json.dumps writes each name as a string literal Python reads back unchanged, so a name with
    quotes, backslashes or accents pastes as exactly that name. A label that is not text (an Excel
    header such as 2024) keeps its type, since that is what pandas indexes the column by.
    """
    literal = json.dumps(
        list(columns),
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda value: value.item() if hasattr(value, "item") else str(value),
    )
    return f"{variable} = {literal}"


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 for embedding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def plot_description(image_path) -> Optional[str]:
    """The description a chart carries in its PNG (see visualizations._save_chart), if any."""
    try:
        with Image.open(image_path) as image:
            return image.text.get("Description") or None
    except OSError:  # missing or unreadable: the template falls back to the chart's kind
        return None


def encode_plot(plot_path: str) -> str:
    """Encode a plot image to base64 if it exists."""
    p = Path(plot_path)
    if p.exists():
        return image_to_base64(p)
    return ""


# HTML Template. It is rendered with autoescaping on: every value it receives is text (column
# names, cell values, messages) and reaches the page escaped, so a name or a value that holds
# markup is shown, not run. All the report's markup is written here; nothing passed in is HTML.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA report: {{ dataset_name }}</title>
    <style>
        /* Instrument panel: cool neutral ground, white panels, a compact sans with figures in mono. The accent
           marks the target and the current section; severity has its own scale. Light and dark follow the
           system, and the toggle overrides it. Charts are PNGs, so in dark mode they sit on light plates. */
        :root {
            --font-sans: system-ui, -apple-system, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas, monospace;
            --ground: #eef1f4; --surface: #ffffff; --sunken: #f5f7f9; --ink: #121a22; --muted: #55616d; --rule: #d6dce2;
            --accent: #0a6a8f; --accent-ink: #ffffff; --accent-soft: #e2f0f6; --plate: #ffffff;
            --sev-low: #2f5f8a; --sev-low-bg: #e8f0f7; --sev-medium: #8a5a00; --sev-medium-bg: #faf0da;
            --sev-high: #a3440f; --sev-high-bg: #fbe8dc; --sev-critical: #a1172f; --sev-critical-bg: #fbe3e7;
            --ok: #2d6a3e; --ok-bg: #e2f0e5; --focus: #1a73c7;
            color-scheme: light;
        }
        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) {
                --ground: #0e1318; --surface: #161d24; --sunken: #121820; --ink: #e3e9ee; --muted: #9aa7b3; --rule: #2a343e;
                --accent: #5cc0e6; --accent-ink: #06222e; --accent-soft: #15303c; --plate: #eef1f4;
                --sev-low: #9cc3e6; --sev-low-bg: #18283a; --sev-medium: #e8c06a; --sev-medium-bg: #33291a;
                --sev-high: #f0a374; --sev-high-bg: #3a2418; --sev-critical: #f28b9c; --sev-critical-bg: #3d1b22;
                --ok: #8fd3a2; --ok-bg: #1c3324; --focus: #7cb8f2; color-scheme: dark;
            }
        }
        :root[data-theme="dark"] {
            --ground: #0e1318; --surface: #161d24; --sunken: #121820; --ink: #e3e9ee; --muted: #9aa7b3; --rule: #2a343e;
            --accent: #5cc0e6; --accent-ink: #06222e; --accent-soft: #15303c; --plate: #eef1f4;
            --sev-low: #9cc3e6; --sev-low-bg: #18283a; --sev-medium: #e8c06a; --sev-medium-bg: #33291a;
            --sev-high: #f0a374; --sev-high-bg: #3a2418; --sev-critical: #f28b9c; --sev-critical-bg: #3d1b22;
            --ok: #8fd3a2; --ok-bg: #1c3324; --focus: #7cb8f2; color-scheme: dark;
        }
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; scroll-padding-top: 1.25rem; }
        @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } *, *::before { transition: none !important; } }
        body { margin: 0; background: var(--ground); color: var(--ink); font: 0.9375rem/1.55 var(--font-sans); -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
        ::selection { background: var(--accent); color: var(--accent-ink); }
        a { color: var(--accent); text-underline-offset: 0.2em; }
        :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 4px; }
        code, pre { font-family: var(--font-mono); font-size: 0.9em; }
        p { margin: 0 0 0.85rem; max-width: 72ch; }
        ul, ol { margin: 0 0 0.85rem; padding-left: 1.4rem; }
        svg.i { width: 1rem; height: 1rem; flex: none; vertical-align: -0.15em; }

        .skip { position: absolute; left: 1rem; top: -3rem; z-index: 50; background: var(--surface); color: var(--ink); padding: 0.5rem 0.8rem;
            border-radius: 8px; border: 1px solid var(--rule); }
        .skip:focus { top: 1rem; }
        .shell { max-width: 80rem; margin: 0 auto; padding: 1.5rem max(16px, env(safe-area-inset-right, 0px)) 3rem max(16px, env(safe-area-inset-left, 0px)); }
        .masthead { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 1rem; margin-bottom: 1.5rem; }
        .masthead .kind { color: var(--muted); font-size: 0.82rem; margin: 0 0 0.25rem; }
        .masthead h1 { margin: 0; font-size: 1.6rem; line-height: 1.2; font-weight: 700; letter-spacing: -0.01em; overflow-wrap: anywhere; }
        .chips { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: var(--muted); }
        .chips > span { background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 0.25rem 0.6rem; }
        .chips strong { font-family: var(--font-mono); font-weight: 500; color: var(--ink); }
        .chips .status { border: 0; font-weight: 650; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.7rem; padding: 0.35rem 0.6rem; }
        .status--ok { background: var(--ok-bg) !important; color: var(--ok); }
        .status--failed { background: var(--sev-high-bg) !important; color: var(--sev-high); }
        .theme-toggle { font: inherit; font-size: 0.78rem; font-weight: 600; color: var(--ink); background: var(--surface); border: 1px solid var(--rule);
            border-radius: 999px; padding: 0.25rem 0.7rem; cursor: pointer; }

        .layout { display: grid; gap: 1.5rem; grid-template-columns: 13rem minmax(0, 1fr); align-items: start; }
        .toc { position: sticky; top: 1rem; background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 0.5rem; font-size: 0.86rem; }
        .toc ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.1rem; }
        .toc a { display: flex; justify-content: space-between; gap: 0.5rem; padding: 0.4rem 0.6rem; border-radius: 7px; color: var(--muted); text-decoration: none; }
        .toc a[aria-current="true"] { color: var(--ink); background: var(--accent-soft); font-weight: 600; }
        .toc .count { font-family: var(--font-mono); font-weight: 400; }
        .toc .sub a { padding-left: 1.2rem; font-size: 0.82rem; }
        main { display: grid; gap: 1.25rem; min-width: 0; }
        main > section { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 1.25rem 1.4rem 1.4rem; min-width: 0; }
        main > section > h2, .section-head h2 { font-size: 1.15rem; font-weight: 700; margin: 0 0 1rem; letter-spacing: -0.005em; }
        .section-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.6rem 1rem; margin-bottom: 1rem; }
        .section-head h2 { margin: 0; }
        h3 { font-size: 1rem; font-weight: 650; margin: 1.6rem 0 0.6rem; }
        h4 { font-size: 0.95rem; font-weight: 650; margin: 1.2rem 0 0.5rem; color: var(--ink); }
        .chart-note { color: var(--muted); font-size: 0.88rem; }
        .no-data { padding: 0.9rem 1rem; background: var(--sunken); border-radius: 8px; color: var(--muted); }

        .readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr)); border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }
        .stat-box { padding: 0.8rem 0.9rem; display: grid; gap: 0.3rem; align-content: start; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 0 -1px -1px 0; min-width: 0; }
        .stat-box .label { font-size: 0.74rem; color: var(--muted); letter-spacing: 0.03em; }
        .stat-box .value { font-family: var(--font-mono); font-size: 1.15rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
        .stat-box .more { font-size: 0.76rem; color: var(--muted); }
        .stat-box.is-target .value { color: var(--accent); }

        .table-wrap, .scroll-x { overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.25rem 0 1rem; }
        th { text-align: left; font-weight: 600; color: var(--muted); font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase; }
        th, td { padding: 0.5rem 0.75rem 0.5rem 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
        td { font-variant-numeric: tabular-nums; }

        .plot-container { margin: 0.75rem 0; }
        .plot-container img { display: block; width: 100%; height: auto; background: var(--plate); border: 1px solid var(--rule); border-radius: 10px; padding: 0.4rem; }
        details.fold { border-top: 1px solid var(--rule); margin-top: 1.25rem; }
        details.fold > summary { cursor: pointer; list-style: none; display: flex; align-items: center; gap: 0.6rem; }
        details.fold > summary::-webkit-details-marker { display: none; }
        details.fold > summary::before { content: ""; width: 0.45rem; height: 0.45rem; border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
            transform: rotate(-45deg); transition: transform 0.15s ease-out; flex: none; }
        details.fold[open] > summary::before { transform: rotate(45deg); }
        details.fold > summary h3 { margin: 1rem 0; }

        .heading-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin: 1.6rem 0 0.6rem; }
        .heading-row h3 { margin: 0; }
        .copy-list { font: inherit; font-size: 0.8em; font-weight: 600; padding: 0.25rem 0.65rem; color: var(--accent); background: var(--accent-soft);
            border: 1px solid transparent; border-radius: 999px; cursor: pointer; display: inline-flex; align-items: center; gap: 0.35rem; }
        .copy-list.copied { color: var(--ok); background: var(--ok-bg); }
        .copy-list-text { margin: 0 0 10px; padding: 8px 12px; background: var(--sunken); border-radius: 8px; font-size: 0.85em; white-space: pre-wrap; word-break: break-all; }
        .copy-list-text[hidden] { display: none; }
        .mc-list { padding-left: 1.4rem; }
        .mc-list li { margin-bottom: 0.3rem; }

        /* Alerts: counts by severity, then a group per severity in which alerts of one kind share a row. */
        .sevbar { display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .sevbar a, .sevbar span { display: inline-flex; align-items: center; gap: 0.4rem; text-decoration: none; padding: 0.3rem 0.6rem; border-radius: 999px; font-weight: 600; font-size: 0.8rem; }
        .sevbar strong { font-family: var(--font-mono); font-weight: 600; }
        .sevbar .is-zero { color: var(--muted); background: var(--sunken); }
        .sev--critical { color: var(--sev-critical); background: var(--sev-critical-bg); }
        .sev--high { color: var(--sev-high); background: var(--sev-high-bg); }
        .sev--medium { color: var(--sev-medium); background: var(--sev-medium-bg); }
        .sev--low { color: var(--sev-low); background: var(--sev-low-bg); }
        .sevgroup + .sevgroup { margin-top: 1.1rem; }
        .sevgroup > h3 { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; margin: 0 0 0.5rem; }
        .sevgroup > h3 strong { font-family: var(--font-mono); color: var(--muted); font-weight: 500; letter-spacing: 0; }
        .sev-text--critical { color: var(--sev-critical); } .sev-text--high { color: var(--sev-high); }
        .sev-text--medium { color: var(--sev-medium); } .sev-text--low { color: var(--sev-low); }
        .alert { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.25rem 0.75rem; padding: 0.8rem 0.9rem; border-radius: 10px; border: 1px solid var(--rule); }
        .alert + .alert { margin-top: 0.5rem; }
        .alert__icon { grid-row: span 3; padding-top: 0.1rem; }
        .alert--high { background: var(--sev-high-bg); border-color: transparent; }
        .alert--critical { background: var(--sev-critical-bg); border-color: transparent; }
        .alert__title { font-weight: 650; overflow-wrap: anywhere; }
        .alert__title span { font-weight: 500; color: var(--muted); }
        .alert__cols { display: flex; flex-wrap: wrap; gap: 0.35rem; }
        .colchip { display: inline-flex; flex-wrap: wrap; gap: 0.35rem; align-items: baseline; font-size: 0.8rem; padding: 0.15rem 0.5rem; border-radius: 6px;
            background: var(--sunken); border: 1px solid var(--rule); max-width: 100%; overflow-wrap: anywhere; }
        .alert--high .colchip, .alert--critical .colchip { background: var(--surface); }
        .colchip code { color: var(--ink); }
        .colchip span { font-family: var(--font-mono); color: var(--muted); }
        .alert__rec { color: var(--muted); font-size: 0.88rem; }
        .alert__rec strong { color: var(--ink); font-weight: 600; }

        footer { max-width: 80rem; margin: 0 auto; padding: 0 max(16px, env(safe-area-inset-right, 0px)) 2rem max(16px, env(safe-area-inset-left, 0px));
            display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; color: var(--muted); font-size: 0.82rem; }
        footer p { margin: 0; }

        @media (hover: hover) {
            .theme-toggle:hover { background: var(--sunken); }
            .toc a:hover { color: var(--ink); background: var(--sunken); }
            tbody tr:hover { background: var(--sunken); }
            .copy-list:hover { border-color: var(--accent); }
        }
        @media (max-width: 60rem) {
            .layout { grid-template-columns: minmax(0, 1fr); gap: 1rem; }
            .toc { position: sticky; top: 0; z-index: 5; overflow-x: auto; border-radius: 0 0 12px 12px; }
            .toc ol { display: flex; width: max-content; }
            .toc .sub a { padding-left: 0.6rem; }
            html { scroll-padding-top: 4rem; }
            main > section { padding: 1rem; }
        }
        @media print {
            body { background: #fff; color: #000; }
            .toc, .theme-toggle, .copy-list, .copy-list-text { display: none; }
            .layout { display: block; }
            main > section { border: 0; padding: 0; margin-bottom: 1.5rem; break-inside: auto; }
            .plot-container, .alert, tr { break-inside: avoid; }
        }
    </style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
    <symbol id="i-low" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8.25" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10 9v5" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/><circle cx="10" cy="6.2" r="1.05" fill="currentColor"/></symbol>
    <symbol id="i-medium" viewBox="0 0 20 20"><path d="M10 2.6 18 16.8H2L10 2.6Z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M10 7.8v4.4" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/><circle cx="10" cy="14.4" r="1.05" fill="currentColor"/></symbol>
    <symbol id="i-high" viewBox="0 0 20 20"><path d="M6.9 2h6.2L18 6.9v6.2L13.1 18H6.9L2 13.1V6.9L6.9 2Z" fill="currentColor"/><path d="M10 5.8v5" stroke="var(--surface)" stroke-width="1.9" stroke-linecap="round"/><circle cx="10" cy="13.9" r="1.15" fill="var(--surface)"/></symbol>
    <symbol id="i-critical" viewBox="0 0 20 20"><circle cx="10" cy="10" r="8.5" fill="currentColor"/><path d="m6.8 6.8 6.4 6.4m0-6.4-6.4 6.4" stroke="var(--surface)" stroke-width="1.9" stroke-linecap="round"/></symbol>
    <symbol id="i-copy" viewBox="0 0 20 20"><rect x="7" y="7" width="10" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M13 4.5V4a1.5 1.5 0 0 0-1.5-1.5h-7A1.5 1.5 0 0 0 3 4v7a1.5 1.5 0 0 0 1.5 1.5H5" fill="none" stroke="currentColor" stroke-width="1.5"/></symbol>
</svg>
<a class="skip" href="#resumen">Skip to the report</a>
<div class="shell">
    <header class="masthead">
        <div>
            <p class="kind">Exploratory data analysis</p>
            <h1>{{ dataset_name }}</h1>
        </div>
        <div class="chips">
            <span>Generated <strong>{{ timestamp }}</strong></span>
            <span>Rows <strong>{{ n_rows }}</strong></span>
            <span>Columns <strong>{{ n_cols }}</strong></span>
            <span>Correlation id <strong>{{ correlation_id }}</strong></span>
            {% if failed_steps %}<span class="status status--failed">{{ failed_steps | length }} step{% if failed_steps | length != 1 %}s{% endif %} failed</span>
            {% else %}<span class="status status--ok">All steps ran</span>{% endif %}
            <button type="button" class="theme-toggle" id="theme-toggle">Theme: system</button>
        </div>
    </header>

    <div class="layout">
    <nav class="toc" aria-label="Report sections">
        <ol>
            <li><a href="#resumen">Summary</a></li>
            {% if alerts %}<li><a href="#alertas">Alerts <span class="count">{{ alerts | length }}</span></a></li>{% endif %}
            {% if failed_steps %}<li><a href="#fallidos">Failed Steps <span class="count">{{ failed_steps | length }}</span></a></li>{% endif %}
            <li><a href="#calidad">Data Quality</a></li>
            {% if preview_rows %}<li><a href="#muestra">First Rows</a></li>{% endif %}
            <li><a href="#univariado">Column by Column</a></li>
            {% if correlations %}<li><a href="#relaciones">Relationships</a></li>{% endif %}
            {% if outliers %}<li><a href="#outliers">Outliers</a></li>{% endif %}
            {% if target_analysis %}<li><a href="#target">Target</a></li>{% endif %}
            <li><a href="#visualizaciones">Charts</a></li>
            {% if multicollinearity %}<li class="sub"><a href="#multicolinealidad">Multicollinearity</a></li>{% endif %}
            {% if transforms and transforms.columns %}<li class="sub"><a href="#transformaciones">Skewed Columns</a></li>{% endif %}
            {% if recommendation_steps %}<li><a href="#preprocesamiento">Preprocessing Plan <span class="count">{{ recommendations | length }}</span></a></li>{% endif %}
        </ol>
    </nav>
    <main>

    <section id="resumen">
        <h2>Summary</h2>
        <div class="readout">
            <div class="stat-box">
                <div class="label">Rows</div>
                <div class="value">{{ n_rows }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Columns</div>
                <div class="value">{{ n_cols }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Numeric columns</div>
                <div class="value">{{ numeric_cols_count }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Categorical columns</div>
                <div class="value">{{ categorical_cols_count }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Missing values</div>
                <div class="value">{{ missing_total }}%</div>
            </div>
            <div class="stat-box">
                <div class="label">Duplicate rows</div>
                <div class="value">{{ duplicates }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Alerts</div>
                <div class="value">{{ alerts | length }}</div>
                {% if alerts %}<div class="more">{% for severity, count in alert_counts.items() if count %}{{ count }} {{ severity }}{% if not loop.last %} · {% endif %}{% endfor %}</div>{% endif %}
            </div>
            {% if target_analysis %}
            <div class="stat-box is-target">
                <div class="label">Target</div>
                <div class="value">{{ target_analysis.target_column }}</div>
                <div class="more">{{ target_analysis.target_type }}</div>
            </div>
            {% endif %}
        </div>
    </section>

    {% if alerts %}
    <section id="alertas">
        <div class="section-head">
            <h2>Alerts</h2>
            <nav class="sevbar" aria-label="Alerts by severity">
                {% for severity, count in alert_counts.items() %}
                {% if count %}<a href="#alertas-{{ severity }}" class="sev--{{ severity }}"><svg class="i" aria-hidden="true"><use href="#i-{{ severity }}"/></svg>{{ severity | capitalize }} <strong>{{ count }}</strong></a>
                {% else %}<span class="is-zero"><svg class="i" aria-hidden="true"><use href="#i-{{ severity }}"/></svg>{{ severity | capitalize }} <strong>0</strong></span>{% endif %}
                {% endfor %}
            </nav>
        </div>
        {% for group in alert_groups %}
        <div class="sevgroup" id="alertas-{{ group.severity }}">
            <h3 class="sev-text--{{ group.severity }}"><svg class="i" aria-hidden="true"><use href="#i-{{ group.severity }}"/></svg>{{ group.severity }} <strong>{{ group.count }}</strong></h3>
            {% for row in group.rows %}
            <div class="alert alert--{{ group.severity }}">
                <span class="alert__icon sev-text--{{ group.severity }}"><svg class="i" aria-hidden="true"><use href="#i-{{ group.severity }}"/></svg></span>
                <div class="alert__title">{{ row.title }}{% if row.columns | length > 1 %} <span>· {{ row.columns | length }} columns</span>{% endif %}</div>
                {% if row.columns %}<div class="alert__cols">{% for column, detail in row.columns %}<span class="colchip"><code>{{ column }}</code>{% if detail %}<span>{{ detail }}</span>{% endif %}</span>{% endfor %}</div>{% endif %}
                {% if row.recommendation %}<div class="alert__rec"><strong>What to do:</strong> {{ row.recommendation }}</div>{% endif %}
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </section>
    {% endif %}

    {% if failed_steps %}
    <section id="fallidos">
        <h2>Failed Steps ({{ failed_steps | length }})</h2>
        <p>These analysis steps failed and were skipped; the rest of the report was produced anyway.
           The logs hold the traceback under correlation id <code>{{ correlation_id }}</code>.</p>
        <ul>
            {% for step in failed_steps %}
            <li><code>{{ step }}</code></li>
            {% endfor %}
        </ul>
    </section>
    {% endif %}

    <section id="calidad">
        <h2>Data Quality</h2>

        <h3>Type and Missing Values per Column</h3>
        <p>"Stored as" is how the column ended up in memory (hover for the exact pandas dtype);
           "Read as" is how the pipeline classified it, and that is what decides which analyses and
           which alerts it receives.</p>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>Stored as</th><th>Read as</th><th>Missing (%)</th></tr>
            </thead>
            <tbody>
                {% for row in column_quality %}
                <tr>
                    <td>{{ row.column }}</td>
                    <td><code title="{{ row.dtype_raw }}">{{ row.dtype }}</code></td>
                    <td>{{ row.type_label }}</td>
                    <td>{{ "%.2f" | format(row.missing_pct) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>

        {% if constant_columns %}
        <h3>Constant Columns</h3>
        <p>These columns hold a single value and can be dropped:</p>
        <ul>
            {% for col in constant_columns %}
            <li><code>{{ col }}</code></li>
            {% endfor %}
        </ul>
        {% endif %}

        {% if quasi_constant_columns %}
        <h3>Almost Constant Columns</h3>
        <p>These columns are dominated by one value:</p>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>Share of its most common value</th></tr>
            </thead>
            <tbody>
                {% for col, pct in quasi_constant_columns.items() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ "%.2f" | format(pct) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}

        {% if ignored_columns %}
        <h3>Ignored Columns (by configuration)</h3>
        <p>Left out of every analysis by <code>column_types.ignore</code>:</p>
        <ul>
            {% for col in ignored_columns %}
            <li><code>{{ col }}</code></li>
            {% endfor %}
        </ul>
        {% endif %}
    </section>

    {% if preview_rows %}
    <section id="muestra">
        <h2>First Rows</h2>
        <p>The first {{ preview_rows | length }} rows as they were read. A missing value shows as "—"
           and very long text is cut.</p>
        <div class="scroll-x">
            <table>
                <thead>
                    <tr>{% for col in preview_columns %}<th>{{ col }}</th>{% endfor %}</tr>
                </thead>
                <tbody>
                    {% for row in preview_rows %}
                    <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </section>
    {% endif %}

    <section id="univariado">
        <h2>Column by Column</h2>

        {% if numeric_stats %}
        <div class="heading-row">
            <h3>Numeric Columns</h3>
            <button type="button" class="copy-list" data-copy-from="copy-list-num"
                    title="Copy num = [...] with the {{ numeric_stats | length }} numeric column names"><svg class="i" aria-hidden="true"><use href="#i-copy"/></svg><span class="label">Copy list</span></button>
        </div>
        <pre class="copy-list-text" id="copy-list-num" hidden>{{ numeric_list }}</pre>
        <div class="table-wrap"><table>
            <thead>
                <tr>
                    <th>Column</th><th>Mean</th><th>Median</th><th>Std. dev.</th>
                    <th>Min</th><th>Max</th><th>Skewness</th><th>Missing</th>
                </tr>
            </thead>
            <tbody>
                {% for col, stats in numeric_stats.items() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ "%.3f" | format(stats.mean) }}</td>
                    <td>{{ "%.3f" | format(stats.median) }}</td>
                    <td>{{ "%.3f" | format(stats.std) }}</td>
                    <td>{{ "%.3f" | format(stats.min) }}</td>
                    <td>{{ "%.3f" | format(stats.max) }}</td>
                    <td>{{ "%.3f" | format(stats.skewness) if stats.skewness is not none else 'N/A' }}</td>
                    <td>{{ stats.missing }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}

        {% if categorical_stats %}
        <div class="heading-row">
            <h3>Categorical Columns</h3>
            <button type="button" class="copy-list" data-copy-from="copy-list-cat"
                    title="Copy cat = [...] with the {{ categorical_stats | length }} categorical column names"><svg class="i" aria-hidden="true"><use href="#i-copy"/></svg><span class="label">Copy list</span></button>
        </div>
        <pre class="copy-list-text" id="copy-list-cat" hidden>{{ categorical_list }}</pre>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>Distinct</th><th>Most common</th><th>Missing</th></tr>
            </thead>
            <tbody>
                {% for col, stats in categorical_stats.items() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ stats.nunique }}</td>
                    <td>{{ stats.mode }}</td>
                    <td>{{ stats.missing }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}

        {% if time_stats %}
        <h3>Time of Day Columns</h3>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>Distinct</th><th>Earliest</th><th>Latest</th><th>Peak hour</th><th>Missing</th></tr>
            </thead>
            <tbody>
                {% for col, stats in time_stats.items() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ stats.nunique }}</td>
                    <td>{{ stats.earliest }}</td>
                    <td>{{ stats.latest }}</td>
                    <td>{% if stats.peak_hour is not none %}{{ "%02d" | format(stats.peak_hour) }}:00 ({{ stats.hour_counts[stats.peak_hour] }} rows){% else %}—{% endif %}</td>
                    <td>{{ stats.missing }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}
    </section>

    {% if correlations %}
    <section id="relaciones">
        <h2>Relationships</h2>

        <h3>Significant Correlations (Pearson)</h3>
        {% if numeric_pairs %}
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>P-value</th></tr>
            </thead>
            <tbody>
                {% for pair in numeric_pairs[:20] %}
                <tr>
                    <td>{{ pair.var1 }}</td>
                    <td>{{ pair.var2 }}</td>
                    <td>{{ "%.3f" | format(pair.correlation) }}</td>
                    <td>{{ format_p_value(pair.p_value) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% else %}
        <p class="no-data">No significant correlation was found.</p>
        {% endif %}

        {% if categorical_pairs %}
        <h3>Categorical Associations (Cramér's V)</h3>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column 1</th><th>Column 2</th><th>Cramér's V</th></tr>
            </thead>
            <tbody>
                {% for pair in categorical_pairs[:20] %}
                <tr>
                    <td>{{ pair.var1 }}</td>
                    <td>{{ pair.var2 }}</td>
                    <td>{{ "%.3f" | format(pair.correlation) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}
    </section>
    {% endif %}

    {% if outliers %}
    <section id="outliers">
        <h2>Outliers</h2>
        <p>Found with three methods: IQR, MAD (a robust z-score) and a multivariate Isolation Forest.</p>
        <p><strong>Rows flagged by at least one of them:</strong> {{ outliers.outlier_indices_union | length }}</p>
        {% if outliers.multivariate_note %}
        <p><strong>Isolation Forest:</strong>
           {{ outliers.multivariate_outliers[0].n_outliers if outliers.multivariate_outliers else 0 }} rows.
           {{ outliers.multivariate_note }}</p>
        {% endif %}
        {% set no_spread = outliers.iqr_outliers.values() | selectattr("note") | map(attribute="column") | list %}
        {% if no_spread %}
        <p><strong>IQR measured differently:</strong> {{ no_spread | join(", ") }}. One repeated value takes over
           their quartiles, so the usual fence would measure that value instead of the spread. The note beside each
           column in <code>tables/outlier_summary.csv</code> says which value and how much of the column it is.</p>
        {% endif %}
    </section>
    {% endif %}

    {% if target_analysis %}
    <section id="target">
        <h2>Target</h2>
        <p><strong>Target:</strong> {{ target_analysis.target_column }} ({{ target_analysis.target_type }})</p>
        <p><strong>Rows:</strong> {{ target_analysis.n_samples }} | <strong>Missing:</strong> {{ target_analysis.n_missing }}</p>

        {% if target_analysis.class_balance %}
        <h3>Class Distribution</h3>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Class</th><th>Rows</th><th>Share</th></tr>
            </thead>
            <tbody>
                {% for cls, count in target_analysis.class_balance.class_counts.items() %}
                <tr>
                    <td>{{ cls }}</td>
                    <td>{{ count }}</td>
                    <td>{{ "%.2f" | format(target_analysis.class_balance.class_proportions[cls]) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% if target_analysis.class_balance.is_imbalanced %}
        <p><strong>⚠️ Imbalanced:</strong> ratio {{ "%.2f" | format(target_analysis.class_balance.imbalance_ratio) }}</p>
        {% endif %}
        {% endif %}

        {% if transforms and transforms.target %}
        {% set skewed_target = transforms.target %}
        <h3>Skew of the Target</h3>
        {% if skewed_target.fixed %}
        <p><strong>{{ skewed_target.column }}</strong> has a skew of {{ "%.2f" | format(skewed_target.skew_before) }}.
           Modelled as {{ skewed_target.label }}({{ skewed_target.column }}), its skew is {{ "%.2f" | format(skewed_target.skew_after) }}
           and its normal QQ plot straightens (r = {{ "%.3f" | format(skewed_target.qq_r_before) }} → {{ "%.3f" | format(skewed_target.qq_r_after) }}).
           On the raw scale its largest values would dominate the squared errors, whatever the model; the
           Transform step of the Preprocessing Plan says how to turn the predictions back.</p>
        {% if plots.qq_target %}
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.qq_target[0].data }}" alt="{{ plots.qq_target[0].alt or 'QQ plot of the target' }}">
        </div>
        {% endif %}
        {% else %}
        <p><strong>{{ skewed_target.column }}</strong> has a skew of {{ "%.2f" | format(skewed_target.skew_before) }},
           and neither transformation both brings it within ±{{ "%g" | format(fixed_skew) }} and straightens its QQ plot:
           {{ describe_attempts(skewed_target) }}.</p>
        {% endif %}
        {% endif %}

        {% if target_analysis.feature_relationships %}
        <h3>Columns Most Related to the Target</h3>
        <p>Ranked by the effect, which runs from 0 (unrelated) to 1 whatever the test.
        {% if target_analysis.target_type == "classification" %}For a class target: eta for numeric columns
           (Kruskal-Wallis p-value) and Cramér's V for categorical ones (chi²).
        {% else %}For a numeric target: |Spearman| for numeric columns and eta for categorical ones
           (Kruskal-Wallis p-value).{% endif %}
           Eta and Cramér's V are corrected for what many categories explain by chance alone.</p>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>Test</th><th>P-value</th><th>Effect</th></tr>
            </thead>
            <tbody>
                {% for rel in target_analysis.feature_relationships[:20] %}
                <tr>
                    <td>{{ rel.feature }}</td>
                    <td>{{ rel.test_name }}</td>
                    <td>{{ format_p_value(rel.p_value) }}</td>
                    <td>{{ format_measure(rel.effect_size, 3) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endif %}

        {% if target_analysis.leakage_alerts %}
        <h3>Posibles Fugas de Datos</h3>
        <ul>
            {% for alert in target_analysis.leakage_alerts %}
            <li>{{ alert }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </section>
    {% endif %}

    <section id="visualizaciones">
        <h2>Charts</h2>

        {% if plots.correlation %}
        <h3>Correlation Matrix</h3>
        {% if chart_notes.correlation %}
        <p class="chart-note">{{ chart_notes.correlation }}</p>
        {% endif %}
        <p>Pearson, pair by pair: each cell uses the rows where both columns have a value, the same rows as
           the Pearson table and the scatter plots, so the three read the same numbers.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.correlation[0].data }}" alt="{{ plots.correlation[0].alt or 'Correlation Matrix' }}">
        </div>
        {% endif %}

        {% if plots.association %}
        <h3>Association Between Columns (categorical ones included)</h3>
        <p>Each cell uses the measure that fits the pair: Pearson between two numbers (-1 to 1, signed),
           Cramér's V between two categories, and the correlation ratio (eta) between a category and a
           number. The last two run from 0 to 1, carry no sign, and are corrected for what many
           categories explain by chance.</p>
        {% if chart_notes.association %}
        <p class="chart-note">{{ chart_notes.association }}</p>
        {% endif %}
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.association[0].data }}" alt="{{ plots.association[0].alt or 'Association Matrix' }}">
        </div>
        {% endif %}

        {% if multicollinearity %}
        {% set mc = multicollinearity %}
        <h3 id="multicolinealidad">Multicollinearity</h3>
        <p>How much of each numeric column the other numeric columns already explain. The VIF of a
           column is 1 / (1 − R²), with R² from regressing it on all the others: 1 means it is
           independent of them, {{ "%g" | format(high_vif) }} or more that they explain 90% of it, and ∞ that it is an
           exact combination of them. 0/1 columns such as one-hot dummies count as numbers here, as they
           do for a model. The matrices above compare columns two at a time; a total and its parts can
           be an exact combination while no pair of them looks alike.</p>
        {% if mc.note %}
        <p class="no-data">{{ mc.note }}</p>
        {% else %}
        <p>Computed over the {{ mc.rows_used }} rows that have a value in all {{ mc.columns | length }} columns{% if mc.rows_used < mc.rows_total %}
           ({{ mc.rows_total - mc.rows_used }} rows with a gap in any of them are left out){% endif %}{% if mc.target_left_out %};
           the target, {{ mc.target_left_out }}, is not one of them{% endif %}.</p>
        {% if mc.chance_r_squared > chance_warning %}
        <p>With {{ mc.rows_used }} rows for {{ mc.columns | length }} columns, a column would reach an R² of
           about {{ "%.2f" | format(mc.chance_r_squared) }} against the others by chance alone, so part of every VIF below is chance.</p>
        {% endif %}

        {% if mc.exact_dependencies %}
        <h4>Exact combinations ({{ mc.exact_dependencies | length }})</h4>
        <ul class="mc-list">
            {% for dependency in mc.exact_dependencies %}
            {% set others = (dependency.columns | length) - 1 %}
            <li><code>{{ dependency.equation() }}</code> holds on
                {% if dependency.rows == dependency.rows_checked %}all {{ dependency.rows }}{% else %}{{ dependency.rows }} of the {{ dependency.rows_checked }}{% endif %}
                rows that have these columns. Dropping one of these {{ dependency.columns | length }} columns, any one, removes it,
                and nothing is lost: the dropped column can be recomputed from the other {% if others == 1 %}one{% else %}{{ others }}{% endif %}.</li>
            {% endfor %}
        </ul>
        <p>Which one you drop changes what is left. Dropping a column that is almost always 0 removes the
           exact combination but leaves the others nearly as tied as before. The drops suggested below, and
           the VIF they leave, are measured.</p>
        {% endif %}

        {% set ranked = mc.ranked(include_exact=False) %}
        {% set exact_count = mc.exact_columns | length %}
        {% if exact_count %}
        <p>The {{ exact_count }} columns of those combinations have an infinite VIF: each one is fully explained by the others.</p>
        {% endif %}
        {% if ranked %}
        <h4>Highest VIF{% if exact_count %} outside the exact combinations{% endif %}</h4>
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>VIF</th><th>Explained mostly by</th></tr>
            </thead>
            <tbody>
                {% for name, value in ranked %}
                <tr>
                    <td>{{ name }}</td>
                    <td>{{ format_vif(value) }}</td>
                    <td>{{ mc.partners.get(name, []) | join(", ") }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% set rest = (mc.columns | length) - exact_count - (ranked | length) %}
        {% if rest %}
        <p>The other {{ rest }} columns have a VIF under {{ "%g" | format(shown_vif) }}.</p>
        {% endif %}
        {% else %}
        <p class="no-data">{% if exact_count %}Outside those combinations, no{% else %}No{% endif %} column has a VIF of {{ "%g" | format(shown_vif) }} or more: none of them is largely explained by the others.</p>
        {% endif %}

        {% if mc.suggested_drops %}
        <h4>Columns to consider dropping ({{ mc.suggested_drops | length }})</h4>
        <p>Dropping these, in this order, leaves every remaining VIF under {{ "%g" | format(high_vif) }}: first one column
           per exact combination, then the column with the highest VIF, measured again after each drop.
           Which column of a related group to keep is a modelling choice the data cannot make, so each
           line names the columns that could go instead.</p>
        <ol class="mc-list">
            {% for drop in mc.suggested_drops %}
            <li><code>{{ drop.column }}</code>:
                {% if drop.exact %}an exact combination of {{ drop.partners | join(", ") }}{% else %}VIF {{ format_vif(drop.vif) }}, explained mostly by {{ drop.partners | join(", ") }}{% endif %}</li>
            {% endfor %}
        </ol>

        {% set after = mc.ranked_after_drops() %}
        {% set left = mc.vif_after_drops | length %}
        <h4>VIF after these drops</h4>
        <p>Measured again on the {{ left }} column{% if left != 1 %}s{% endif %} left. A column that read ∞ inside an
           exact combination shows here the VIF it really keeps.</p>
        {% if after %}
        <div class="table-wrap"><table>
            <thead>
                <tr><th>Column</th><th>VIF</th><th>Explained mostly by</th></tr>
            </thead>
            <tbody>
                {% for name, value in after %}
                <tr>
                    <td>{{ name }}</td>
                    <td>{{ format_vif(value) }}</td>
                    <td>{{ mc.partners_after_drops.get(name, []) | join(", ") }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% set rest_after = left - (after | length) %}
        {% if rest_after == 1 %}
        <p>The other column has a VIF under {{ "%g" | format(shown_vif) }}.</p>
        {% elif rest_after %}
        <p>The other {{ rest_after }} columns have a VIF under {{ "%g" | format(shown_vif) }}.</p>
        {% endif %}
        {% elif left == 1 %}
        <p class="no-data">After them a single column is left, so nothing is left to explain it.</p>
        {% else %}
        <p class="no-data">After them, every one of the {{ left }} columns left has a VIF under {{ "%g" | format(shown_vif) }}.</p>
        {% endif %}
        {% endif %}
        {% endif %}
        {% endif %}

        {% if plots.missing %}
        <h3>Missing Values</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.missing[0].data }}" alt="{{ plots.missing[0].alt or 'Missing Values Matrix' }}">
        </div>
        {% elif n_rows and missing_total == 0 %}
        <h3>Missing Values</h3>
        <p class="no-data">No cell is missing in any of the {{ n_cols }} columns, so there is no matrix to draw.</p>
        {% endif %}

        {% if plots.histograms %}
        <details class="fold" open><summary><h3>Numeric Distributions ({{ plots.histograms | length }})</h3></summary>
        {% if chart_notes.histograms %}
        <p class="chart-note">{{ chart_notes.histograms }}</p>
        {% endif %}
        {% if floor_split_columns %}
        <p class="chart-note">
            Drawn a second time without the value that fills them, beside the whole column:
            {% for col, check in floor_split_columns.items() %}<strong>{{ col }}</strong>
            ({{ "%g" | format(check.floor) }} on {{ "%.1f" | format(check.floor_pct) }}% of the rows){% if not loop.last %}, {% endif %}{% endfor %}.
            A value repeated at the bottom of a column flattens everything else, and when that value is 0 a
            log scale cannot fix it: it would leave most of the rows out. This is done when that value fills
            at least {{ "%g" | format(floor_split_min_share) }}% of the rows, what is left holds more than
            {{ floor_split_min_distinct }} distinct values, and the chart of the whole column is squeezed.
        </p>
        {% endif %}
        {% if log_scale_columns %}
        <p class="chart-note">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
            Used when the middle 90% of the rows takes less than 30% of the axis, the log at least doubles that
            span, and at most 5% of the values are 0 or negative (those are left out of the log panel).
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.histograms %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Histogram' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.boxplots %}
        <details class="fold" open><summary><h3>Boxplots ({{ plots.boxplots | length }})</h3></summary>
        {% if chart_notes.boxplots %}
        <p class="chart-note">{{ chart_notes.boxplots }}</p>
        {% endif %}
        {% if floor_split_columns %}
        <p class="chart-note">
            Without the value that fills them, beside the whole column:
            <strong>{{ floor_split_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        {% if log_scale_columns %}
        <p class="chart-note">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.boxplots %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Boxplot' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if transforms and transforms.columns %}
        <h3 id="transformaciones">Skewed Columns, Before and After a Transformation</h3>
        {% if chart_notes.qq %}
        <p class="chart-note">{{ chart_notes.qq }}</p>
        {% endif %}
        <p>A normal QQ plot sets a column's values against the values a normal distribution would have in the
           same positions: on the red line, the column is normal, and a tail that bends away from it is longer
           than a normal one. Checked here: the continuous columns with a skew beyond ±{{ "%g" | format(min_skew) }},
           more than {{ min_distinct }} distinct values, and their lowest value on less than {{ "%g" | format(max_floor_pct) }}%
           of the rows, since no transformation straightens a pile of identical values. A column is drawn when a
           transformation brings its skew within ±{{ "%g" | format(fixed_skew) }} and makes its QQ plot straighter (r up by
           at least {{ "%g" | format(min_qq_gain) }}): log1p, the log of 1 + x, when that is
           enough, since it is the easier one to read back, and Yeo-Johnson otherwise, which also takes negative
           values. Linear models, the ones that work with distances (kNN, SVM, k-means) and PCA gain from it;
           trees and boosting do not need it.</p>
        {% if transforms.unfixed %}
        <p>Skewed, but neither transformation both brings them within ±{{ "%g" | format(fixed_skew) }} and straightens
           their QQ plot:
            {% for check in transforms.unfixed %}<strong>{{ check.column }}</strong>
            (skew {{ "%.2f" | format(check.skew_before) }}: {{ describe_attempts(check) }}){% if not loop.last %}, {% endif %}{% endfor %}.</p>
        {% endif %}
        {% if plots.qq %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.qq %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'QQ plot before and after' }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}

        {% if plots.categorical %}
        <details class="fold" open><summary><h3>Categorical Distributions ({{ plots.categorical | length }})</h3></summary>
        {% if chart_notes.categorical %}
        <p class="chart-note">{{ chart_notes.categorical }}</p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.categorical %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Categorical' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.pie %}
        <details class="fold" open><summary><h3>Categorical Shares ({{ plots.pie | length }})</h3></summary>
        <p class="chart-note">
            Each category as a share of the rows that have a value. Only for columns with up to
            {{ pie_max_categories }} categories: above that the slices stop being distinguishable and the bar chart is enough.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.pie %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Pie chart' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.target_categorical %}
        <details class="fold" open><summary><h3>Target vs Categorical Columns ({{ plots.target_categorical | length }})</h3></summary>
        {% if chart_notes.target_categorical %}
        <p class="chart-note">{{ chart_notes.target_categorical }}</p>
        {% endif %}
        <p class="chart-note">
            Grouped bars of each categorical column against the target
            {% if target_analysis %}<strong>{{ target_analysis.target_column }}</strong>{% endif %}.
            Every label is a share of the chart total, so the bars add up to 100%.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.target_categorical %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Target vs categorical' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.scatter %}
        <details class="fold" open><summary><h3>Scatter Plots (Most Correlated Pairs)</h3></summary>
        {% if log_scale_columns %}
        <p class="chart-note">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.scatter %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Scatter' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.pair_plot and pair_plot %}
        <h3>Pair Plot: How the Numeric Columns Relate</h3>
        <p class="chart-note">
            Each panel crosses two of these columns: <strong>{{ pair_plot.columns | join(", ") }}</strong>.
            The diagonal holds each one's distribution{% if pair_plot.hue %}, by group{% endif %}.
            {% if pair_plot.hue_reason == "target" %}
            Coloured by the target <strong>{{ pair_plot.hue }}</strong>.
            {% elif pair_plot.hue %}
            Coloured by <strong>{{ pair_plot.hue }}</strong>, the categorical column that separates these the most
            (mean η {{ "%.2f" | format(pair_plot.hue_eta) }}).
            {% elif pair_plot.best_group %}
            Not coloured: the categorical column that separates these the most, <strong>{{ pair_plot.best_group }}</strong>,
            only reaches a mean η of {{ "%.2f" | format(pair_plot.best_eta) }} (colouring starts at {{ "%.2f" | format(pair_plot_min_eta) }}).
            {% else %}
            Not coloured: no categorical column has 2 to 6 groups with enough rows in each.
            {% endif %}
            {% if pair_plot.rows_plotted < pair_plot.rows_available %}
            A fixed random sample of {{ pair_plot.rows_plotted }} of the {{ pair_plot.rows_available }} complete rows.
            {% endif %}
        </p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.pair_plot[0].data }}" alt="{{ plots.pair_plot[0].alt or 'Pair plot' }}">
        </div>
        {% if plots.pair_plot_log %}
        <p class="chart-note">
            The same pair plot with a log scale (log10) on
            <strong>{% for col in pair_plot.columns if col in log_scale_columns %}{{ col }}{% if not loop.last %}, {% endif %}{% endfor %}</strong>,
            to compare against the one above. Rows with 0 or negative values in those columns are left out.
        </p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.pair_plot_log[0].data }}" alt="{{ plots.pair_plot_log[0].alt or 'Pair plot (log10)' }}">
        </div>
        {% endif %}
        {% endif %}

        {% if plots.timeseries %}
        <details class="fold" open><summary><h3>Rows Over Time ({{ plots.timeseries | length }})</h3></summary>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.timeseries %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Time series' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}

        {% if plots.time_of_day %}
        <details class="fold" open><summary><h3>By Hour of the Day ({{ plots.time_of_day | length }})</h3></summary>
        {% if chart_notes.time_of_day %}
        <p class="chart-note">{{ chart_notes.time_of_day }}</p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 30rem), 1fr)); gap: 20px;">
            {% for img in plots.time_of_day %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img.data }}" alt="{{ img.alt or 'Time of day' }}">
            </div>
            {% endfor %}
        </div>
        </details>
        {% endif %}
    </section>

    {% if recommendation_steps %}
    <section id="preprocesamiento">
        <h2>Preprocessing Plan ({{ recommendations | length }})</h2>
        <p>The steps in the order they are applied, and next to every line the measurement that produced
           it, so it can be checked instead of believed.</p>
        <p><strong>Only what the data settles on its own is here.</strong> Two things are left out, because
           they cannot be read from the table: whether a categorical column is <em>ordinal</em> and in what
           order its levels go, and whether your model needs scaling at all (a tree does not; a distance or
           a penalty does). The plan decides about the numeric and categorical columns; dates, times and free
           text only appear when they hold numbers as text.</p>
        {% for step, label, rows in recommendation_steps %}
        <h3>{{ label }} ({{ rows | length }})</h3>
        <div class="table-wrap"><table>
            <thead>
                <tr><th style="width: 18%;">Column</th><th style="width: 34%;">Action</th><th>Why (measured)</th></tr>
            </thead>
            <tbody>
                {% for rec in rows %}
                <tr>
                    <td>{% if rec.column %}<code>{{ rec.column }}</code>{% else %}<em>whole table</em>{% endif %}</td>
                    <td>{{ rec.action }}</td>
                    <td class="chart-note">{{ rec.evidence }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table></div>
        {% endfor %}
    </section>
    {% endif %}

    </main>
    </div>
</div>
<footer>
    <p>Generated by EDA Pipeline v0.1.0 · a self-contained report that works offline</p>
    <p><a href="#resumen">Back to the top</a></p>
</footer>
<script>
/* The report's only script: the theme toggle, the section the navigation marks, and the "Copy list" buttons. */
(function () {
    var root = document.documentElement;

    var themes = ["system", "light", "dark"];
    var toggle = document.getElementById("theme-toggle");
    var theme = "system";
    try { theme = localStorage.getItem("eda-report-theme") || "system"; } catch (err) {}
    function applyTheme(next) {
        theme = themes.indexOf(next) >= 0 ? next : "system";
        if (theme === "system") { root.removeAttribute("data-theme"); } else { root.setAttribute("data-theme", theme); }
        toggle.textContent = "Theme: " + theme;
        try { localStorage.setItem("eda-report-theme", theme); } catch (err) {}
    }
    applyTheme(theme);
    toggle.addEventListener("click", function () { applyTheme(themes[(themes.indexOf(theme) + 1) % themes.length]); });

    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
    var targets = links.map(function (link) { return document.getElementById(link.getAttribute("href").slice(1)); });
    var ticking = false;
    function markCurrent() {
        ticking = false;
        var current = 0;
        if (window.scrollY > 8) {
            targets.forEach(function (target, index) {
                if (target && target.getBoundingClientRect().top < window.innerHeight * 0.3) { current = index; }
            });
        }
        links.forEach(function (link, index) {
            if (index === current) { link.setAttribute("aria-current", "true"); } else { link.removeAttribute("aria-current"); }
        });
    }
    window.addEventListener("scroll", function () {
        if (!ticking) { ticking = true; window.requestAnimationFrame(markCurrent); }
    }, { passive: true });
    markCurrent();

    function legacyCopy(value) {
        var area = document.createElement("textarea");
        area.value = value;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try {
            ok = document.execCommand("copy");
        } catch (err) {
            ok = false;
        }
        document.body.removeChild(area);
        return ok;
    }

    function copy(value) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(value).then(
                function () { return true; },
                function () { return legacyCopy(value); }
            );
        }
        return Promise.resolve(legacyCopy(value));
    }

    document.querySelectorAll("button.copy-list").forEach(function (button) {
        var source = document.getElementById(button.dataset.copyFrom);
        var label = button.querySelector(".label");
        var text = label.textContent;
        button.addEventListener("click", function () {
            copy(source.textContent).then(function (ok) {
                button.focus();
                if (ok) {
                    source.hidden = true;
                    label.textContent = "Copied";
                    button.classList.add("copied");
                    setTimeout(function () {
                        label.textContent = text;
                        button.classList.remove("copied");
                    }, 1600);
                    return;
                }
                /* No clipboard here (a sandboxed viewer, say): show the list, selected, to copy by hand. */
                source.hidden = false;
                var range = document.createRange();
                range.selectNodeContents(source);
                var selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
            });
        });
    });
})();
</script>
</body>
</html>
"""


def generate_html_report(
    output_path: Path,
    dataset_name: str,
    correlation_id: str,
    df_shape: tuple,
    data_quality: dict,
    numeric_stats: dict,
    categorical_stats: dict,
    correlations: dict,
    outliers: dict,
    target_analysis: dict,
    plot_files: dict,
    alerts: list,
    ignored_columns: Optional[list] = None,
    failed_steps: Optional[list] = None,
    column_types: Optional[dict] = None,
    column_dtypes: Optional[dict] = None,
    data_preview: Optional[pd.DataFrame] = None,
    time_stats: Optional[dict] = None,
    pair_plot=None,
    log_scale_columns: Optional[list] = None,
    floor_split_columns: Optional[dict] = None,
    recommendations: Optional[list] = None,
    multicollinearity=None,
    transforms=None,
    chart_columns=None,
) -> str:
    """
    Generate HTML report with embedded base64 images.

    Args:
        ignored_columns: Columns excluded from analysis via `column_types.ignore`.
        failed_steps: Names of analysis steps that failed and were skipped (see pipeline._run_step).
        column_types: Inferred semantic type per column, shown in the data quality table.
        column_dtypes: Storage dtype per column (int64, float64, str, ...), shown next to it.
        data_preview: The analysed DataFrame; its first rows open the report body.
        time_stats: ``{column: TimeOfDayStats}`` for the time-of-day columns.
        pair_plot: The ``PairPlotSpec`` the pair plot was drawn from, described next to it.
        log_scale_columns: Columns drawn with a log scale next to the linear chart.
        floor_split_columns: ``{column: FloorSplitCheck as a dict}`` for the columns drawn again
            without the repeated lowest value that fills them.
        recommendations: The preprocessing plan (``Recommendation`` list), shown as the last section.
        multicollinearity: ``MulticollinearityReport``, shown under the correlation matrices.
        transforms: ``TransformReport``: skewed columns under the boxplots, a skewed target in its section.
        chart_columns: ``{chart: ChartColumns}``; each capped chart gets a note naming what it left out.

    Returns:
        Path to generated HTML file
    """
    logger.info(f"Generating HTML report: {output_path}")

    # Encode all plots to base64
    encoded_plots = {}
    for key, files in plot_files.items():
        encoded_plots[key] = [{"data": encode_plot(f), "alt": plot_description(f)} for f in files]

    # Prepare template data
    template_data = {
        "dataset_name": dataset_name,
        "correlation_id": correlation_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": df_shape[0],
        "n_cols": df_shape[1],
        "numeric_cols_count": len(numeric_stats),
        "categorical_cols_count": len(categorical_stats),
        "missing_total": overall_missing_pct(data_quality.get("missing_per_column", {})),
        "duplicates": data_quality.get("duplicates", 0),
        "column_quality": column_quality_rows(data_quality.get("missing_per_column", {}), column_types, column_dtypes),
        "preview_columns": [str(col) for col in data_preview.columns] if data_preview is not None else [],
        "preview_rows": preview_rows(data_preview) if data_preview is not None else [],
        "constant_columns": data_quality.get("constant_columns", []),
        "quasi_constant_columns": data_quality.get("quasi_constant_columns", {}),
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "multicollinearity": multicollinearity,
        "transforms": transforms,
        "chart_notes": {
            key: chosen.note(CHARTS[key][1], CHARTS[key][0])
            for key, chosen in (chart_columns or {}).items()
            if chosen.capped
        },
        "describe_attempts": describe_attempts,
        "min_skew": MIN_SKEW,
        "fixed_skew": FIXED_SKEW,
        "min_qq_gain": MIN_QQ_GAIN,
        "max_floor_pct": MAX_FLOOR_PCT,
        "min_distinct": MIN_DISTINCT,
        "format_vif": format_vif,
        "high_vif": HIGH_VIF,
        "shown_vif": SHOWN_VIF,
        "chance_warning": CHANCE_R_SQUARED_WARNING,
        "numeric_list": python_list_snippet("num", numeric_stats),
        "categorical_list": python_list_snippet("cat", categorical_stats),
        "time_stats": time_stats or {},
        "pair_plot": pair_plot,
        "log_scale_columns": log_scale_columns or [],
        "floor_split_columns": floor_split_columns or {},
        "recommendations": recommendations or [],
        "recommendation_steps": recommendations_by_step(recommendations),
        "pair_plot_min_eta": PAIR_PLOT_MIN_HUE_ETA,
        "correlations": correlations,
        "numeric_pairs": correlations.get("numeric_pairs", []),
        "categorical_pairs": correlations.get("categorical_pairs", []),
        "outliers": outliers,
        "target_analysis": target_analysis,
        "plots": encoded_plots,
        "pie_max_categories": PIE_MAX_CATEGORIES,
        "floor_split_min_share": FLOOR_SPLIT_MIN_SHARE,
        "floor_split_min_distinct": FLOOR_SPLIT_MIN_DISTINCT,
        "alerts": alerts,
        "alert_counts": alert_counts(alerts),
        "alert_groups": alert_groups(alerts),
        "ignored_columns": ignored_columns or [],
        "failed_steps": failed_steps or [],
        "format_measure": format_measure,
        "format_p_value": format_p_value,
    }

    # Render template
    template = Template(HTML_TEMPLATE, autoescape=True)
    html_content = template.render(**template_data)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved: {output_path}")
    return str(output_path)


PREVIEW_ROWS = 10
_PREVIEW_MAX_CHARS = 200


def preview_rows(df: pd.DataFrame, limit: int = PREVIEW_ROWS) -> list[list[str]]:
    """
    The first `limit` rows as display text: missing values become a dash, long values are cut.

    Cutting matters for free-text columns, where one cell can hold thousands of characters and
    would stretch the table well past the page.
    """
    rows: list[list[str]] = []
    for _, row in df.head(limit).iterrows():
        cells = []
        for value in row:
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):  # a list or dict cell is never "missing"
                missing = False
            if missing:
                cells.append("—")
                continue
            text = str(value)
            cells.append(text if len(text) <= _PREVIEW_MAX_CHARS else text[: _PREVIEW_MAX_CHARS - 1] + "…")
        rows.append(cells)
    return rows


def column_quality_rows(
    missing_per_column: dict[str, float],
    column_types: Optional[dict[str, str]] = None,
    column_dtypes: Optional[dict[str, str]] = None,
    limit: int = 20,
) -> list[dict]:
    """
    One row per column for the data quality table: how it was classified, and how much is missing.

    Ordered by missing percentage (descending) and capped at `limit` rows, as the table was before
    the type columns were added. A column with no known type shows a dash, so the report still
    renders when the maps are not provided.
    """
    column_types = column_types or {}
    column_dtypes = column_dtypes or {}
    ordered = sorted(missing_per_column.items(), key=lambda item: -item[1])[:limit]
    return [
        {
            "column": col,
            "dtype": dtype_label(column_dtypes.get(col)),
            "dtype_raw": str(column_dtypes.get(col, "—")),
            "type_label": semantic_type_label(column_types.get(col)),
            "missing_pct": pct,
        }
        for col, pct in ordered
    ]


def overall_missing_pct(missing_per_column: dict[str, float]) -> float:
    """
    Percentage (0-100) of missing cells across the analyzed columns, rounded to 2 decimals.

    `missing_per_column` already holds per-column percentages, so their mean equals the share
    of missing cells in the whole table.
    """
    if not missing_per_column:
        return 0.0
    return round(sum(missing_per_column.values()) / len(missing_per_column), 2)
