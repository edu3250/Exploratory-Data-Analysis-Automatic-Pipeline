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

from .multicollinearity import CHANCE_R_SQUARED_WARNING, HIGH_VIF, SHOWN_VIF
from .recommendations import recommendations_by_step
from .type_inference import dtype_label, semantic_type_label
from .visualizations import (
    FLOOR_SPLIT_MIN_DISTINCT,
    FLOOR_SPLIT_MIN_SHARE,
    PAIR_PLOT_MIN_HUE_ETA,
    PIE_MAX_CATEGORIES,
)

logger = logging.getLogger(__name__)


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
    p-value that reaches exactly 0 only underflowed. The template escapes what this returns, since
    it is rendered without autoescaping and a bare < opens a tag.
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


def encode_plot(plot_path: str) -> str:
    """Encode a plot image to base64 if it exists."""
    p = Path(plot_path)
    if p.exists():
        return image_to_base64(p)
    return ""


# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA report: {{ dataset_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        header .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .meta {
            background: #f9f9f9;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            font-size: 0.9em;
        }
        .toc {
            background: #f0f0f0;
            padding: 20px;
            margin: 20px;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }
        .toc h3 {
            margin-bottom: 10px;
            color: #667eea;
        }
        .toc ul {
            list-style: none;
        }
        .toc li {
            margin: 5px 0;
            padding-left: 20px;
        }
        .toc a {
            color: #667eea;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .alerts-section {
            background: #fff5e6;
            border-left: 4px solid #ff9800;
            margin: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .alerts-section h2 {
            color: #d84315;
            margin-bottom: 15px;
        }
        .alert {
            margin: 10px 0;
            padding: 10px;
            border-left: 3px solid #ff9800;
            background: white;
            border-radius: 2px;
        }
        .alert.critical {
            border-left-color: #d32f2f;
            background: #ffebee;
        }
        .alert.high {
            border-left-color: #f57c00;
            background: #fff3e0;
        }
        .alert.medium {
            border-left-color: #fbc02d;
            background: #fffde7;
        }
        .alert.low {
            border-left-color: #0288d1;
            background: #e1f5fe;
        }
        .alert-severity {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.85em;
            margin-right: 10px;
            color: white;
        }
        .alert.critical .alert-severity {
            background: #d32f2f;
        }
        .alert.high .alert-severity {
            background: #f57c00;
        }
        .alert.medium .alert-severity {
            background: #fbc02d;
            color: #333;
        }
        .alert.low .alert-severity {
            background: #0288d1;
        }
        section {
            margin: 30px 20px;
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 4px;
        }
        h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }
        th {
            background: #f0f0f0;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
            page-break-inside: avoid;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-box {
            display: inline-block;
            background: #f9f9f9;
            padding: 15px 20px;
            margin: 10px 5px;
            border-radius: 4px;
            border-left: 3px solid #667eea;
            min-width: 150px;
        }
        .stat-box .label {
            font-size: 0.85em;
            color: #999;
            text-transform: uppercase;
        }
        .stat-box .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        footer {
            background: #f0f0f0;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
            color: #999;
            border-top: 1px solid #ddd;
        }
        .no-data {
            padding: 20px;
            background: #f0f0f0;
            border-radius: 4px;
            color: #999;
            font-style: italic;
        }
        h4 {
            color: #555;
            margin: 16px 0 8px;
        }
        .mc-list {
            padding-left: 24px;
        }
        .mc-list li {
            margin-bottom: 4px;
        }
        .heading-row {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 12px;
            margin: 20px 0 10px;
        }
        .heading-row h3 {
            margin: 0;
        }
        .copy-list {
            font: inherit;
            font-size: 0.8em;
            padding: 3px 10px;
            color: #667eea;
            background: #f4f5ff;
            border: 1px solid #c9cdf5;
            border-radius: 4px;
            cursor: pointer;
        }
        .copy-list:hover {
            background: #e8eaff;
        }
        .copy-list:focus-visible {
            outline: 2px solid #667eea;
            outline-offset: 2px;
        }
        .copy-list.copied {
            color: #2e7d32;
            background: #f1f8f1;
            border-color: #a5d6a7;
        }
        .copy-list-text {
            margin: 0 0 10px;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .copy-list-text[hidden] {
            display: none;
        }
        @media print {
            body { background: white; }
            section { page-break-inside: avoid; }
            .container { box-shadow: none; }
            .copy-list, .copy-list-text { display: none; }
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>📊 Exploratory Data Analysis</h1>
        <p class="subtitle">{{ dataset_name }}</p>
    </header>

    <div class="meta">
        <strong>Generated:</strong> {{ timestamp }} |
        <strong>Rows:</strong> {{ n_rows }} |
        <strong>Columns:</strong> {{ n_cols }} |
        <strong>Correlation id:</strong> {{ correlation_id }}
    </div>

    <div class="toc">
        <h3>📑 Contents</h3>
        <ul>
            <li><a href="#resumen">Summary</a></li>
            <li><a href="#alertas">Alerts</a></li>
            <li><a href="#calidad">Data Quality</a></li>
            {% if preview_rows %}<li><a href="#muestra">First Rows</a></li>{% endif %}
            <li><a href="#univariado">Column by Column</a></li>
            <li><a href="#relaciones">Relationships</a></li>
            {% if outliers %}<li><a href="#outliers">Outliers</a></li>{% endif %}
            {% if target_analysis %}<li><a href="#target">Target</a></li>{% endif %}
            <li><a href="#visualizaciones">Charts</a></li>
            {% if multicollinearity %}<li><a href="#multicolinealidad">Multicollinearity</a></li>{% endif %}
            {% if recommendation_steps %}<li><a href="#preprocesamiento">Preprocessing Plan</a></li>{% endif %}
        </ul>
    </div>

    {% if alerts %}
    <div class="alerts-section">
        <h2 id="alertas">⚠️ Alerts ({{ alerts | length }})</h2>
        {% for alert in alerts %}
        <div class="alert {{ alert.severity }}">
            <span class="alert-severity">{{ alert.severity | upper }}</span>
            {% if alert.column %}<strong>{{ alert.column }}:</strong>{% endif %}
            {{ alert.message }}
            {% if alert.recommendation %}<br><em>→ {{ alert.recommendation }}</em>{% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if failed_steps %}
    <div class="alerts-section">
        <h2>❌ Failed Steps ({{ failed_steps | length }})</h2>
        <p>These analysis steps failed and were skipped; the rest of the report was produced anyway.
           The logs hold the traceback under correlation id <code>{{ correlation_id }}</code>.</p>
        <ul>
            {% for step in failed_steps %}
            <li><code>{{ step }}</code></li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <section id="resumen">
        <h2>📈 Summary</h2>
        <div>
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
        </div>
    </section>

    <section id="calidad">
        <h2>🔍 Data Quality</h2>

        <h3>Type and Missing Values per Column</h3>
        <p>"Stored as" is how the column ended up in memory (hover for the exact pandas dtype);
           "Read as" is how the pipeline classified it, and that is what decides which analyses and
           which alerts it receives.</p>
        <table>
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
        </table>

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
        <table>
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
        </table>
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
        <h2>🧾 First Rows</h2>
        <p>The first {{ preview_rows | length }} rows as they were read. A missing value shows as "—"
           and very long text is cut.</p>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>{% for col in preview_columns %}<th>{{ col | e }}</th>{% endfor %}</tr>
                </thead>
                <tbody>
                    {% for row in preview_rows %}
                    <tr>{% for cell in row %}<td>{{ cell | e }}</td>{% endfor %}</tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </section>
    {% endif %}

    <section id="univariado">
        <h2>📊 Column by Column</h2>

        {% if numeric_stats %}
        <div class="heading-row">
            <h3>Numeric Columns</h3>
            <button type="button" class="copy-list" data-copy-from="copy-list-num"
                    title="Copy num = [...] with the {{ numeric_stats | length }} numeric column names">📋 Copy list</button>
        </div>
        <pre class="copy-list-text" id="copy-list-num" hidden>{{ numeric_list | e }}</pre>
        <table>
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
        </table>
        {% endif %}

        {% if categorical_stats %}
        <div class="heading-row">
            <h3>Categorical Columns</h3>
            <button type="button" class="copy-list" data-copy-from="copy-list-cat"
                    title="Copy cat = [...] with the {{ categorical_stats | length }} categorical column names">📋 Copy list</button>
        </div>
        <pre class="copy-list-text" id="copy-list-cat" hidden>{{ categorical_list | e }}</pre>
        <table>
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
        </table>
        {% endif %}

        {% if time_stats %}
        <h3>Time of Day Columns</h3>
        <table>
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
        </table>
        {% endif %}
    </section>

    {% if correlations %}
    <section id="relaciones">
        <h2>🔗 Relationships</h2>

        <h3>Significant Correlations (Pearson)</h3>
        {% if numeric_pairs %}
        <table>
            <thead>
                <tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>P-value</th></tr>
            </thead>
            <tbody>
                {% for pair in numeric_pairs[:20] %}
                <tr>
                    <td>{{ pair.var1 }}</td>
                    <td>{{ pair.var2 }}</td>
                    <td>{{ "%.3f" | format(pair.correlation) }}</td>
                    <td>{{ format_p_value(pair.p_value) | e }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="no-data">No significant correlation was found.</p>
        {% endif %}

        {% if categorical_pairs %}
        <h3>Categorical Associations (Cramér's V)</h3>
        <table>
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
        </table>
        {% endif %}
    </section>
    {% endif %}

    {% if outliers %}
    <section id="outliers">
        <h2>🎯 Outliers</h2>
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
        <h2>🎯 Target</h2>
        <p><strong>Target:</strong> {{ target_analysis.target_column }} ({{ target_analysis.target_type }})</p>
        <p><strong>Rows:</strong> {{ target_analysis.n_samples }} | <strong>Missing:</strong> {{ target_analysis.n_missing }}</p>

        {% if target_analysis.class_balance %}
        <h3>Class Distribution</h3>
        <table>
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
        </table>
        {% if target_analysis.class_balance.is_imbalanced %}
        <p><strong>⚠️ Imbalanced:</strong> ratio {{ "%.2f" | format(target_analysis.class_balance.imbalance_ratio) }}</p>
        {% endif %}
        {% endif %}

        {% if target_analysis.feature_relationships %}
        <h3>Columns Most Related to the Target</h3>
        <table>
            <thead>
                <tr><th>Column</th><th>Test</th><th>P-value</th><th>Effect</th></tr>
            </thead>
            <tbody>
                {% for rel in target_analysis.feature_relationships[:20] %}
                <tr>
                    <td>{{ rel.feature }}</td>
                    <td>{{ rel.test_name }}</td>
                    <td>{{ format_p_value(rel.p_value) | e }}</td>
                    <td>{{ format_measure(rel.effect_size, 3) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if target_analysis.leakage_alerts %}
        <h3>⚠️ Posibles Fugas de Datos</h3>
        <ul>
            {% for alert in target_analysis.leakage_alerts %}
            <li>{{ alert }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </section>
    {% endif %}

    <section id="visualizaciones">
        <h2>📸 Charts</h2>

        {% if plots.correlation %}
        <h3>Correlation Matrix</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.correlation[0] }}" alt="Correlation Matrix">
        </div>
        {% endif %}

        {% if plots.association %}
        <h3>Association Between Columns (categorical ones included)</h3>
        <p>Each cell uses the measure that fits the pair: Pearson between two numbers (-1 to 1, signed),
           Cramér's V between two categories, and the correlation ratio (eta) between a category and a
           number. The last two run from 0 to 1 and carry no sign.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.association[0] }}" alt="Association Matrix">
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
        <p class="no-data">{{ mc.note | e }}</p>
        {% else %}
        <p>Computed over the {{ mc.rows_used }} rows that have a value in all {{ mc.columns | length }} columns{% if mc.rows_used < mc.rows_total %}
           ({{ mc.rows_total - mc.rows_used }} rows with a gap in any of them are left out){% endif %}{% if mc.target_left_out %};
           the target, {{ mc.target_left_out | e }}, is not one of them{% endif %}.</p>
        {% if mc.chance_r_squared > chance_warning %}
        <p>With {{ mc.rows_used }} rows for {{ mc.columns | length }} columns, a column would reach an R² of
           about {{ "%.2f" | format(mc.chance_r_squared) }} against the others by chance alone, so part of every VIF below is chance.</p>
        {% endif %}

        {% if mc.exact_dependencies %}
        <h4>Exact combinations ({{ mc.exact_dependencies | length }})</h4>
        <ul class="mc-list">
            {% for dependency in mc.exact_dependencies %}
            <li><code>{{ dependency.equation() | e }}</code> holds on
                {% if dependency.rows == dependency.rows_checked %}all {{ dependency.rows }}{% else %}{{ dependency.rows }} of the {{ dependency.rows_checked }}{% endif %}
                rows that have these columns. Dropping any one of these {{ dependency.columns | length }} columns breaks it.</li>
            {% endfor %}
        </ul>
        {% endif %}

        {% set ranked = mc.ranked(include_exact=False) %}
        {% set exact_count = mc.exact_columns | length %}
        {% if exact_count %}
        <p>The {{ exact_count }} columns of those combinations have an infinite VIF: each one is fully explained by the others.</p>
        {% endif %}
        {% if ranked %}
        <h4>Highest VIF{% if exact_count %} outside the exact combinations{% endif %}</h4>
        <table>
            <thead>
                <tr><th>Column</th><th>VIF</th><th>Explained mostly by</th></tr>
            </thead>
            <tbody>
                {% for name, value in ranked %}
                <tr>
                    <td>{{ name | e }}</td>
                    <td>{{ format_vif(value) }}</td>
                    <td>{{ mc.partners.get(name, []) | join(", ") | e }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
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
            <li><code>{{ drop.column | e }}</code>:
                {% if drop.exact %}an exact combination of {{ drop.partners | join(", ") | e }}{% else %}VIF {{ format_vif(drop.vif) }}, explained mostly by {{ drop.partners | join(", ") | e }}{% endif %}</li>
            {% endfor %}
        </ol>
        {% endif %}
        {% endif %}
        {% endif %}

        {% if plots.missing %}
        <h3>Missing Values</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.missing[0] }}" alt="Missing Values Matrix">
        </div>
        {% endif %}

        {% if plots.histograms %}
        <h3>Numeric Distributions ({{ plots.histograms | length }})</h3>
        {% if floor_split_columns %}
        <p style="color: #555; margin-top: -8px;">
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
        <p style="color: #555; margin-top: -8px;">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
            Used when the middle 90% of the rows takes less than 30% of the axis, the log at least doubles that
            span, and at most 5% of the values are 0 or negative (those are left out of the log panel).
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.histograms %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Histogram">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.boxplots %}
        <h3>Boxplots ({{ plots.boxplots | length }})</h3>
        {% if floor_split_columns %}
        <p style="color: #555; margin-top: -8px;">
            Without the value that fills them, beside the whole column:
            <strong>{{ floor_split_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        {% if log_scale_columns %}
        <p style="color: #555; margin-top: -8px;">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.boxplots %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Boxplot">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.categorical %}
        <h3>Categorical Distributions ({{ plots.categorical | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.categorical %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Categorical">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.pie %}
        <h3>Categorical Shares ({{ plots.pie | length }})</h3>
        <p style="color: #555; margin-top: -8px;">
            Each category as a share of the rows that have a value. Only for columns with up to
            {{ pie_max_categories }} categories: above that the slices stop being distinguishable and the bar chart is enough.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.pie %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Pie chart">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.target_categorical %}
        <h3>Target vs Categorical Columns ({{ plots.target_categorical | length }})</h3>
        <p style="color: #555; margin-top: -8px;">
            Grouped bars of each categorical column against the target
            {% if target_analysis %}<strong>{{ target_analysis.target_column }}</strong>{% endif %}.
            Every label is a share of the chart total, so the bars add up to 100%.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.target_categorical %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Target vs categorical">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.scatter %}
        <h3>Scatter Plots (Most Correlated Pairs)</h3>
        {% if log_scale_columns %}
        <p style="color: #555; margin-top: -8px;">
            With a log scale to the right of the linear one: <strong>{{ log_scale_columns | join(", ") }}</strong>.
        </p>
        {% endif %}
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.scatter %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Scatter">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.pair_plot and pair_plot %}
        <h3>Pair Plot: How the Numeric Columns Relate</h3>
        <p style="color: #555; margin-top: -8px;">
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
            <img src="data:image/png;base64,{{ plots.pair_plot[0] }}" alt="Pair plot">
        </div>
        {% if plots.pair_plot_log %}
        <p style="color: #555;">
            The same pair plot with a log scale (log10) on
            <strong>{% for col in pair_plot.columns if col in log_scale_columns %}{{ col }}{% if not loop.last %}, {% endif %}{% endfor %}</strong>,
            to compare against the one above. Rows with 0 or negative values in those columns are left out.
        </p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.pair_plot_log[0] }}" alt="Pair plot (log10)">
        </div>
        {% endif %}
        {% endif %}

        {% if plots.timeseries %}
        <h3>Rows Over Time ({{ plots.timeseries | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.timeseries %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Time series">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.time_of_day %}
        <h3>By Hour of the Day ({{ plots.time_of_day | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.time_of_day %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Time of day">
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </section>

    {% if recommendation_steps %}
    <section id="preprocesamiento">
        <h2>🧭 Preprocessing Plan ({{ recommendations | length }})</h2>
        <p>The steps in the order they are applied, and next to every line the measurement that produced
           it, so it can be checked instead of believed.</p>
        <p><strong>Only what the data settles on its own is here.</strong> Two things are left out, because
           they cannot be read from the table: whether a categorical column is <em>ordinal</em> and in what
           order its levels go, and whether your model needs scaling at all (a tree does not; a distance or
           a penalty does). The plan decides about the numeric and categorical columns; dates, times and free
           text only appear when they hold numbers as text.</p>
        {% for step, label, rows in recommendation_steps %}
        <h3>{{ label }} ({{ rows | length }})</h3>
        <table>
            <thead>
                <tr><th style="width: 18%;">Column</th><th style="width: 34%;">Action</th><th>Why (measured)</th></tr>
            </thead>
            <tbody>
                {% for rec in rows %}
                <tr>
                    <td>{% if rec.column %}<code>{{ rec.column }}</code>{% else %}<em>whole table</em>{% endif %}</td>
                    <td>{{ rec.action }}</td>
                    <td style="color: #555;">{{ rec.evidence }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endfor %}
    </section>
    {% endif %}

    <footer>
        <p>Generated by EDA Pipeline v0.1.0 | self-contained report, works offline</p>
    </footer>
</div>
<script>
/* The report's only script: the "Copy list" buttons beside Numeric and Categorical Columns. */
(function () {
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
        var label = button.textContent;
        button.addEventListener("click", function () {
            copy(source.textContent).then(function (ok) {
                button.focus();
                if (ok) {
                    source.hidden = true;
                    button.textContent = "✓ Copied";
                    button.classList.add("copied");
                    setTimeout(function () {
                        button.textContent = label;
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

    Returns:
        Path to generated HTML file
    """
    logger.info(f"Generating HTML report: {output_path}")

    # Encode all plots to base64
    encoded_plots = {}
    for key, files in plot_files.items():
        encoded_plots[key] = [encode_plot(f) for f in files]

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
        "ignored_columns": ignored_columns or [],
        "failed_steps": failed_steps or [],
        "format_measure": format_measure,
        "format_p_value": format_p_value,
    }

    # Render template
    template = Template(HTML_TEMPLATE)
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
