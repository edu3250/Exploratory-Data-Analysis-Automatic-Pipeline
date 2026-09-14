"""
HTML report generation: self-contained, offline-ready report.
"""

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Template

from .type_inference import dtype_label, semantic_type_label
from .visualizations import PAIR_PLOT_MIN_HUE_ETA, PIE_MAX_CATEGORIES

logger = logging.getLogger(__name__)


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
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte EDA: {{ dataset_name }}</title>
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
        @media print {
            body { background: white; }
            section { page-break-inside: avoid; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>📊 Análisis Exploratorio de Datos</h1>
        <p class="subtitle">{{ dataset_name }}</p>
    </header>

    <div class="meta">
        <strong>Generado:</strong> {{ timestamp }} |
        <strong>Filas:</strong> {{ n_rows }} |
        <strong>Columnas:</strong> {{ n_cols }} |
        <strong>ID Correlación:</strong> {{ correlation_id }}
    </div>

    <div class="toc">
        <h3>📑 Tabla de Contenidos</h3>
        <ul>
            <li><a href="#resumen">Resumen Ejecutivo</a></li>
            <li><a href="#alertas">Alertas y Recomendaciones</a></li>
            <li><a href="#calidad">Calidad de Datos</a></li>
            {% if preview_rows %}<li><a href="#muestra">Primeras Filas</a></li>{% endif %}
            <li><a href="#univariado">Análisis Univariado</a></li>
            <li><a href="#relaciones">Relaciones entre Variables</a></li>
            {% if outliers %}<li><a href="#outliers">Análisis de Outliers</a></li>{% endif %}
            {% if target_analysis %}<li><a href="#target">Análisis de Target</a></li>{% endif %}
            <li><a href="#visualizaciones">Visualizaciones</a></li>
        </ul>
    </div>

    {% if alerts %}
    <div class="alerts-section">
        <h2 id="alertas">⚠️ Alertas y Recomendaciones ({{ alerts | length }})</h2>
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
        <h2>❌ Pasos con Error ({{ failed_steps | length }})</h2>
        <p>Los siguientes pasos del análisis fallaron y se omitieron; el resto del reporte se generó igualmente.
           Revise los logs con el ID de correlación <code>{{ correlation_id }}</code> para más detalle.</p>
        <ul>
            {% for step in failed_steps %}
            <li><code>{{ step }}</code></li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <section id="resumen">
        <h2>📈 Resumen Ejecutivo</h2>
        <div>
            <div class="stat-box">
                <div class="label">Total de Filas</div>
                <div class="value">{{ n_rows }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Total de Columnas</div>
                <div class="value">{{ n_cols }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Columnas Numéricas</div>
                <div class="value">{{ numeric_cols_count }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Columnas Categóricas</div>
                <div class="value">{{ categorical_cols_count }}</div>
            </div>
            <div class="stat-box">
                <div class="label">Valores Faltantes</div>
                <div class="value">{{ missing_total }}%</div>
            </div>
            <div class="stat-box">
                <div class="label">Filas Duplicadas</div>
                <div class="value">{{ duplicates }}</div>
            </div>
        </div>
    </section>

    <section id="calidad">
        <h2>🔍 Calidad de Datos</h2>

        <h3>Tipo y Valores Faltantes por Columna</h3>
        <p>«Tipo de dato» es cómo quedó almacenada la columna (el tipo exacto de pandas aparece al
           pasar el cursor); «Categoría inferida» es cómo la clasificó el pipeline, que es lo que
           decide qué análisis y qué alertas recibe.</p>
        <table>
            <thead>
                <tr><th>Columna</th><th>Tipo de dato</th><th>Categoría inferida</th><th>Faltantes (%)</th></tr>
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
        <h3>Columnas Constantes</h3>
        <p>Las siguientes columnas tienen un único valor y pueden eliminarse:</p>
        <ul>
            {% for col in constant_columns %}
            <li><code>{{ col }}</code></li>
            {% endfor %}
        </ul>
        {% endif %}

        {% if quasi_constant_columns %}
        <h3>Columnas Quasi-Constantes</h3>
        <p>Las siguientes columnas están dominadas por un único valor:</p>
        <table>
            <thead>
                <tr><th>Columna</th><th>% Valor Más Frecuente</th></tr>
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
        <h3>Columnas Ignoradas (por configuración)</h3>
        <p>Excluidas de todo el análisis por <code>column_types.ignore</code>:</p>
        <ul>
            {% for col in ignored_columns %}
            <li><code>{{ col }}</code></li>
            {% endfor %}
        </ul>
        {% endif %}
    </section>

    {% if preview_rows %}
    <section id="muestra">
        <h2>🧾 Primeras Filas</h2>
        <p>Las primeras {{ preview_rows | length }} filas del dataset, tal como se leyeron. Los valores
           faltantes aparecen como «—» y los textos muy largos se recortan.</p>
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
        <h2>📊 Análisis Univariado</h2>

        {% if numeric_stats %}
        <h3>Columnas Numéricas</h3>
        <table>
            <thead>
                <tr>
                    <th>Columna</th><th>Media</th><th>Mediana</th><th>Desv.Est.</th>
                    <th>Min</th><th>Max</th><th>Skewness</th><th>Nulos</th>
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
        <h3>Columnas Categóricas</h3>
        <table>
            <thead>
                <tr><th>Columna</th><th>Únicos</th><th>Modo</th><th>Nulos</th></tr>
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
        <h3>Columnas de Hora del Día</h3>
        <table>
            <thead>
                <tr><th>Columna</th><th>Únicos</th><th>Primera</th><th>Última</th><th>Hora pico</th><th>Nulos</th></tr>
            </thead>
            <tbody>
                {% for col, stats in time_stats.items() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ stats.nunique }}</td>
                    <td>{{ stats.earliest }}</td>
                    <td>{{ stats.latest }}</td>
                    <td>{% if stats.peak_hour is not none %}{{ "%02d" | format(stats.peak_hour) }}:00 ({{ stats.hour_counts[stats.peak_hour] }} filas){% else %}—{% endif %}</td>
                    <td>{{ stats.missing }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </section>

    {% if correlations %}
    <section id="relaciones">
        <h2>🔗 Relaciones entre Variables</h2>

        <h3>Correlaciones Significativas (Pearson)</h3>
        {% if numeric_pairs %}
        <table>
            <thead>
                <tr><th>Variable 1</th><th>Variable 2</th><th>Correlación</th><th>P-Value</th></tr>
            </thead>
            <tbody>
                {% for pair in numeric_pairs[:20] %}
                <tr>
                    <td>{{ pair.var1 }}</td>
                    <td>{{ pair.var2 }}</td>
                    <td>{{ "%.3f" | format(pair.correlation) }}</td>
                    <td>{{ "%.4f" | format(pair.p_value) if pair.p_value else 'N/A' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="no-data">No se encontraron correlaciones significativas.</p>
        {% endif %}

        {% if categorical_pairs %}
        <h3>Asociaciones Categóricas (Cramér's V)</h3>
        <table>
            <thead>
                <tr><th>Variable 1</th><th>Variable 2</th><th>Cramér's V</th></tr>
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
        <h2>🎯 Análisis de Outliers</h2>
        <p>Se detectaron outliers usando IQR, MAD (z-score robusto) e Isolation Forest.</p>
        <p><strong>Total de filas con al menos un outlier detectado:</strong> {{ outliers.outlier_indices_union | length }}</p>
        {% if outliers.multivariate_note %}
        <p><strong>Isolation Forest:</strong>
           {{ outliers.multivariate_outliers[0].n_outliers if outliers.multivariate_outliers else 0 }} filas.
           {{ outliers.multivariate_note }}</p>
        {% endif %}
        {% set no_spread = outliers.iqr_outliers.values() | selectattr("note") | map(attribute="column") | list %}
        {% if no_spread %}
        <p><strong>IQR no aplicado:</strong> {{ no_spread | join(", ") }}. Al menos la mitad de sus valores son iguales,
           así que el IQR vale 0 y cualquier otro valor saldría como atípico.</p>
        {% endif %}
    </section>
    {% endif %}

    {% if target_analysis %}
    <section id="target">
        <h2>🎯 Análisis de Variable Target</h2>
        <p><strong>Target:</strong> {{ target_analysis.target_column }} ({{ target_analysis.target_type }})</p>
        <p><strong>Muestras:</strong> {{ target_analysis.n_samples }} | <strong>Faltantes:</strong> {{ target_analysis.n_missing }}</p>

        {% if target_analysis.class_balance %}
        <h3>Distribución de Clases</h3>
        <table>
            <thead>
                <tr><th>Clase</th><th>Frecuencia</th><th>Porcentaje</th></tr>
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
        <p><strong>⚠️ Desbalance detectado:</strong> Ratio {{ "%.2f" | format(target_analysis.class_balance.imbalance_ratio) }}</p>
        {% endif %}
        {% endif %}

        {% if target_analysis.feature_relationships %}
        <h3>Features más relacionadas con Target</h3>
        <table>
            <thead>
                <tr><th>Feature</th><th>Test</th><th>P-Value</th><th>Efecto</th></tr>
            </thead>
            <tbody>
                {% for rel in target_analysis.feature_relationships[:20] %}
                <tr>
                    <td>{{ rel.feature }}</td>
                    <td>{{ rel.test_name }}</td>
                    <td>{{ "%.4f" | format(rel.p_value) if rel.p_value and rel.p_value == rel.p_value else 'N/A' }}</td>
                    <td>{{ "%.3f" | format(rel.effect_size) if rel.effect_size and rel.effect_size == rel.effect_size else 'N/A' }}</td>
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
        <h2>📸 Visualizaciones</h2>

        {% if plots.correlation %}
        <h3>Matriz de Correlación</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.correlation[0] }}" alt="Correlation Matrix">
        </div>
        {% endif %}

        {% if plots.association %}
        <h3>Asociación entre Variables (incluye categóricas)</h3>
        <p>Cada celda usa la medida que corresponde al par: Pearson entre numéricas (de -1 a 1, con
           signo), Cramér's V entre categóricas y razón de correlación (eta) entre una categórica y
           una numérica; estas dos van de 0 a 1 y no tienen signo.</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.association[0] }}" alt="Association Matrix">
        </div>
        {% endif %}

        {% if plots.missing %}
        <h3>Mapa de Valores Faltantes</h3>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.missing[0] }}" alt="Missing Values Matrix">
        </div>
        {% endif %}

        {% if plots.histograms %}
        <h3>Distribuciones Numéricas ({{ plots.histograms | length }})</h3>
        {% if log_scale_columns %}
        <p style="color: #555; margin-top: -8px;">
            Con escala logarítmica a la derecha de la lineal: <strong>{{ log_scale_columns | join(", ") }}</strong>.
            Se usa cuando el 90 % central de las filas ocupa menos del 30 % del eje, el logaritmo al menos duplica
            ese espacio y como mucho el 5 % de los valores son 0 o negativos (esos quedan fuera del panel logarítmico).
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
        {% if log_scale_columns %}
        <p style="color: #555; margin-top: -8px;">
            Con escala logarítmica a la derecha de la lineal: <strong>{{ log_scale_columns | join(", ") }}</strong>.
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
        <h3>Distribuciones Categóricas ({{ plots.categorical | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.categorical %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Categorical">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.pie %}
        <h3>Proporciones Categóricas ({{ plots.pie | length }})</h3>
        <p style="color: #555; margin-top: -8px;">
            Porcentaje de cada categoría sobre las filas que tienen valor. Solo para columnas con hasta
            {{ pie_max_categories }} categorías: con más, las porciones ya no se distinguen y queda el gráfico de barras.
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
        <h3>Target vs Variables Categóricas ({{ plots.target_categorical | length }})</h3>
        <p style="color: #555; margin-top: -8px;">
            Barras agrupadas de cada variable categórica frente al target
            {% if target_analysis %}<strong>{{ target_analysis.target_column }}</strong>{% endif %}.
            Cada etiqueta es el porcentaje sobre el total de filas del gráfico, así que las barras suman 100%.
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
        <h3>Scatter Plots (Pares con Mayor Correlación)</h3>
        {% if log_scale_columns %}
        <p style="color: #555; margin-top: -8px;">
            Con escala logarítmica a la derecha de la lineal: <strong>{{ log_scale_columns | join(", ") }}</strong>.
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
        <h3>Pair Plot: Relaciones entre Variables Numéricas</h3>
        <p style="color: #555; margin-top: -8px;">
            Cada panel cruza dos de estas variables: <strong>{{ pair_plot.columns | join(", ") }}</strong>.
            En la diagonal, la distribución de cada una{% if pair_plot.hue %} por grupo{% endif %}.
            {% if pair_plot.hue_reason == "target" %}
            Coloreado por el target <strong>{{ pair_plot.hue }}</strong>.
            {% elif pair_plot.hue %}
            Coloreado por <strong>{{ pair_plot.hue }}</strong>, la variable categórica que más separa estas columnas
            (η medio {{ "%.2f" | format(pair_plot.hue_eta) }}).
            {% elif pair_plot.best_group %}
            Sin colorear: la variable categórica que más separa estas columnas, <strong>{{ pair_plot.best_group }}</strong>,
            solo llega a un η medio de {{ "%.2f" | format(pair_plot.best_eta) }} (se colorea desde {{ "%.2f" | format(pair_plot_min_eta) }}).
            {% else %}
            Sin colorear: no hay una variable categórica de 2 a 6 grupos con suficientes filas en cada uno.
            {% endif %}
            {% if pair_plot.rows_plotted < pair_plot.rows_available %}
            Muestra aleatoria de {{ pair_plot.rows_plotted }} de las {{ pair_plot.rows_available }} filas completas.
            {% endif %}
        </p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.pair_plot[0] }}" alt="Pair plot">
        </div>
        {% if plots.pair_plot_log %}
        <p style="color: #555;">
            El mismo pair plot con escala logarítmica (log10) en
            <strong>{% for col in pair_plot.columns if col in log_scale_columns %}{{ col }}{% if not loop.last %}, {% endif %}{% endfor %}</strong>,
            para compararlo con el de arriba. Se omiten las filas con valores 0 o negativos en esas columnas.
        </p>
        <div class="plot-container">
            <img src="data:image/png;base64,{{ plots.pair_plot_log[0] }}" alt="Pair plot (log10)">
        </div>
        {% endif %}
        {% endif %}

        {% if plots.timeseries %}
        <h3>Series Temporales ({{ plots.timeseries | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.timeseries %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Time series">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if plots.time_of_day %}
        <h3>Distribución por Hora del Día ({{ plots.time_of_day | length }})</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px;">
            {% for img in plots.time_of_day %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ img }}" alt="Time of day">
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </section>

    <footer>
        <p>Generado por EDA Pipeline v0.1.0 | Reporte Auto-Contenido (Offline-Ready)</p>
    </footer>
</div>
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
        "time_stats": time_stats or {},
        "pair_plot": pair_plot,
        "log_scale_columns": log_scale_columns or [],
        "pair_plot_min_eta": PAIR_PLOT_MIN_HUE_ETA,
        "correlations": correlations,
        "numeric_pairs": correlations.get("numeric_pairs", []),
        "categorical_pairs": correlations.get("categorical_pairs", []),
        "outliers": outliers,
        "target_analysis": target_analysis,
        "plots": encoded_plots,
        "pie_max_categories": PIE_MAX_CATEGORIES,
        "alerts": alerts,
        "ignored_columns": ignored_columns or [],
        "failed_steps": failed_steps or [],
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
