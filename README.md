# Pipeline de Análisis Exploratorio de Datos (EDA)

Un pipeline completo y automatizado para realizar análisis exploratorio de datos en cualquier dataset tabular. Genera reportes HTML interactivos, detección de calidad de datos, análisis estadísticos, detección de outliers, y visualizaciones en una única ejecución.

## Características

- **Carga de datos flexible**: CSV, TSV, Excel, Parquet, JSON/JSONL con detección automática de delimitador, codificación y separador decimal
- **Inferencia de tipos semánticos**: Automático reconocimiento de numéricos, categóricos, datetime, texto, identificadores, constantes
- **Análisis de calidad de datos**: Valores faltantes, duplicados, columnas constantes, cardinalidad alta, tipos mixtos
- **Análisis univariado**: Estadísticas descriptivas (media, mediana, desv.est., skewness, kurtosis), pruebas de normalidad, frecuencias
- **Detección de outliers**: IQR, MAD (z-score robusto), Isolation Forest multivariado
- **Análisis de relaciones**: Correlación Pearson, Cramér's V (categóricas), correlation ratio (mixtas), detección de multicolinealidad (VIF)
- **Análisis de target**: Auto-detección clasificación/regresión, balance de clases, relaciones feature-target, detección de fugas
- **Visualizaciones**: Histogramas, boxplots, gráficos categóricos, matriz de correlaciones, mapa de nulos, scatter plots, series temporales — todos los gráficos generados se incluyen en el reporte
- **Reportes**: HTML auto-contenido (offline), JSON estructurado (`summary.json`), tablas CSV de resultados (`tables/`)
- **Batch processing**: Procesa todos los archivos soportados de una carpeta, uno a la vez; un archivo que falla no aborta el resto
- **Configuración flexible**: YAML + overrides de CLI, con validación y mensajes de error claros
- **CLI en español**: Comandos y ayuda completamente en español

## Instalación

### Requisitos previos

- Python 3.10+
- pip

### Setup

```powershell
# Clonar o descargar el proyecto
cd D:\DataScience\Pipeline

# Instalar en modo desarrollo (usa el entorno virtual del proyecto)
.\.venv\Scripts\python.exe -m pip install -e .

# O instalar los requisitos directamente
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Para desarrollo y testing

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Uso

### 1. Analizar un archivo único

```powershell
# CSV simple (delimitador, codificación y separador decimal se auto-detectan)
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv

# Con target (clasificación/regresión, auto-detectado si no se indica --target-type)
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --target categoria

# Con encoding, delimitador y separador decimal explícitos
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --encoding latin-1 --delimiter ";" --decimal ","

# Excel: por defecto se usa la primera hoja; --sheet acepta nombre o índice
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file datos.xlsx --sheet 1
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file datos.xlsx --sheet "Hoja2"

# Con directorio de salida personalizado
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --output-dir mis_reportes
```

Si `--target` no existe en el dataset, el comando termina con código de salida distinto de cero
y un mensaje en español que lista las columnas disponibles (y una sugerencia si el nombre es
parecido a una columna real).

### 2. Procesar múltiples archivos

```powershell
# Todos los archivos con extensión soportada en una carpeta (los ocultos, ej. .gitkeep,
# y las extensiones no soportadas se omiten con un aviso; no abortan el lote)
.\.venv\Scripts\python.exe -m eda_pipeline analyze-batch ./data

# Solo CSVs
.\.venv\Scripts\python.exe -m eda_pipeline analyze-batch ./data --pattern "*.csv"

# Con target y las mismas opciones de carga que analyze-file
.\.venv\Scripts\python.exe -m eda_pipeline analyze-batch ./data --target target_col --target-type classification
.\.venv\Scripts\python.exe -m eda_pipeline analyze-batch ./data --delimiter ";" --encoding cp1252 --decimal "," --sheet 0
```

Cada archivo se carga y analiza de a uno (no se cargan todos en memoria antes de empezar). Un
archivo que falla al cargar o analizar se reporta con ❌ y el resto del lote continúa, salvo que
se use `--strict` (en cuyo caso se detiene en el primer error). Si dos archivos comparten el mismo
nombre base pero distinta extensión (p. ej. `sales.csv` y `sales.parquet`), sus resultados se
nombran `sales_csv` y `sales_parquet` para no pisarse — tanto en las claves del resultado como en
el nombre de la carpeta de salida. Si el target configurado no existe en un dataset del lote, se
registra una advertencia y se agrega una alerta a ese reporte, mientras el resto del análisis
continúa sin target (a diferencia de `analyze-file`, donde falta el target aborta ese archivo).

### 3. Crear archivo de configuración

```powershell
.\.venv\Scripts\python.exe -m eda_pipeline init-config

# O con ruta personalizada
.\.venv\Scripts\python.exe -m eda_pipeline init-config --output mi_config.yaml
```

### 4. Usar configuración YAML

Editar `config/default.yaml` (creado con `init-config`):

```yaml
# Tipo de archivo (auto-detecta por extensión si es None)
file_format: null

# Para CSVs (todo se auto-detecta si se deja en null)
delimiter: null
encoding: null
decimal: null          # null = auto-detecta '.' o ',' según el delimitador y el contenido

# Para Excel
excel_sheet: null       # null = primera hoja; nombre o índice si se especifica

sample_size: null
batch_pattern: null     # patrón glob para analyze-batch; null = todas las extensiones soportadas

# Tipos de columnas (overrides sobre la inferencia automática)
column_types:
  numeric:              # fuerza conversión con pd.to_numeric (errors="coerce")
    - edad
    - ingreso
  categorical:
    - region
    - categoria
  datetime: []          # fuerza conversión con pd.to_datetime (errors="coerce")
  text: []
  ignore:               # excluye estas columnas de TODO el análisis
    - id_interno

# Configuración de análisis
data_quality:
  missing_threshold: 0.95      # (0, 1]
  cardinality_threshold: 100   # entero positivo
  duplicate_threshold: 0.1     # (0, 1]; controla la severidad de la alerta de duplicados
  constant_threshold: 0.99     # (0, 1]

outliers:
  iqr_multiplier: 1.5
  z_score_threshold: 3.0
  isolation_forest_enabled: true
  isolation_forest_contamination: 0.1

visualizations:
  max_histograms: 20
  max_boxplots: 20
  max_correlation_heatmap_size: 30
  max_scatter_pairs: 10
  correlation_threshold: 0.05

target:
  target_column: null
  target_type: null              # null = auto-detecta; "classification" o "regression"
  class_imbalance_threshold: 0.8 # se marca desbalanceado si la clase mayoritaria supera este %

output_dir: "reports"
strict_mode: false   # true = detener todo el análisis en el primer error (dataset o paso)
verbose: false        # true = logging a nivel DEBUG
```

Todas las claves anteriores son las únicas reconocidas: una clave desconocida (mal escrita, o de
una versión anterior) hace que el comando falle con un mensaje en español que lista las claves
válidas, y con código de salida 2. La clave `language`, presente en configuraciones antiguas, ya
no tiene efecto: si aparece se ignora con una advertencia de depreciación, en vez de fallar.

El orden de precedencia es **valores por defecto < archivo YAML < opciones de CLI**: una opción de
CLI solo se aplica si el usuario la pasó explícitamente (de lo contrario no pisa lo que diga el
YAML), y las secciones anidadas (`target`, `data_quality`, etc.) se combinan clave por clave en
vez de reemplazarse por completo (p. ej. `--target` no borra un `class_imbalance_threshold` que
venga del YAML).

Luego usar:

```powershell
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --config config/default.yaml
```

## Ejemplos Completos

### Ejemplo 1: Dataset de clasificación desbalanceado

```powershell
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file transactions.csv `
  --target is_fraud `
  --output-dir ./fraud_analysis `
  --target-type classification
```

Genera:
- `fraud_analysis/<dataset>_<timestamp>/report.html` – Reporte interactivo con análisis de balance, features importantes, alertas de desbalance
- `fraud_analysis/<dataset>_<timestamp>/summary.json` – Resultados estructurados
- `fraud_analysis/<dataset>_<timestamp>/plots/` – Gráficos PNG
- `fraud_analysis/<dataset>_<timestamp>/tables/` – Tablas CSV de resultados

### Ejemplo 2: Dataset con encoding y decimal latinos

```powershell
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file datos_argentina.csv `
  --encoding cp1252 `
  --delimiter ";" `
  --decimal ","
```

Estas tres opciones son opcionales: si el archivo usa `;` como delimitador y valores como `"4,0"`
en columnas numéricas, el separador decimal `,` se detecta automáticamente sin pasar `--decimal`.

### Ejemplo 3: Procesamiento batch

```powershell
.\.venv\Scripts\python.exe -m eda_pipeline analyze-batch ./raw_data `
  --pattern "*.xlsx" `
  --target "categoria" `
  --output-dir ./eda_reports
```

Procesa todos los `.xlsx` en `./raw_data`. Cada archivo tiene su propio reporte y todos quedan
juntos en una única carpeta de la ejecución: `eda_reports/raw_data_batch_<timestamp>/<archivo>/`.

## Estructura de Salida

Un solo archivo (`analyze-file`) deja una carpeta con su marca de tiempo:

```
reports/
├── dataset_20250910_143025/
│   ├── report.html           # Reporte principal (auto-contenido, abre en navegador)
│   ├── summary.json          # Resultados en JSON (máquina-legible), incluye failed_steps
│   ├── plots/
│   │   ├── histogram_age.png
│   │   ├── boxplot_age.png
│   │   ├── categorical_region.png
│   │   ├── correlation_heatmap.png
│   │   ├── association_heatmap.png
│   │   ├── missing_matrix.png
│   │   ├── scatter_age_vs_income.png
│   │   ├── timeseries_date.png
│   │   └── ... (todos los gráficos generados aparecen también en report.html)
│   └── tables/
│       ├── numeric_stats.csv
│       ├── categorical_stats.csv
│       ├── missing_per_column.csv
│       ├── correlations.csv
│       ├── outlier_summary.csv
│       └── alerts.csv
│
└── dataset2_20250910_144000/
    ├── report.html
    ├── summary.json
    ├── plots/
    └── tables/
```

Un lote (`analyze-batch`) agrupa **todos** los reportes de esa ejecución en una sola carpeta,
`<carpeta_de_entrada>_batch_<timestamp>`, con una subcarpeta por archivo. Las subcarpetas no llevan
marca de tiempo propia, porque ya la lleva la carpeta del lote:

```
reports/
└── Vistara_batch_20250910_143025/
    ├── Customers/
    │   ├── report.html
    │   ├── summary.json
    │   ├── plots/
    │   └── tables/
    ├── Order_Details/
    │   └── ...
    └── Sales_Receipts/
        └── ...
```

Así, ocho archivos de entrada dejan **una** carpeta en `reports/` en vez de ocho, y dos ejecuciones
del mismo lote no se mezclan. La ruta de la carpeta se muestra al terminar y queda en el log.

En la sección «Calidad de Datos» del HTML, la tabla «Tipo y Valores Faltantes por Columna» indica
para cada columna el tipo de dato con el que quedó almacenada (`int`, `float`, `texto`, `fecha`…,
con el tipo exacto de pandas —`Int64`, `string`…— al pasar el cursor) y la categoría que le asignó
el pipeline (Numérica continua, Categórica, Identificador, Fecha/hora…). Esa categoría es la que
decide qué análisis y qué alertas recibe la
columna, así que es el primer sitio donde mirar si algo quedó mal clasificado. Las mismas dos
columnas aparecen en `tables/missing_per_column.csv`.

Los logs de cada ejecución NO se guardan dentro de la carpeta del reporte, sino en una carpeta
`logs/` hermana del directorio de salida (`<output_dir>/../logs/eda_<timestamp>_<correlation_id>.log`).
Por ejemplo, con `--output-dir reports` los logs quedan en `logs/`, junto a `reports/`.

## Configuración de Análisis

### Tipos de Datos Detectados Automáticamente

- **numeric_continuous**: Float, muchos valores únicos (ej: edad, salario)
- **numeric_discrete**: Int, pocos valores únicos (ej: num_hijos, rating 1-5)
- **categorical**: Texto cuyos valores se repiten (ej: región, género, causa de incumplimiento), sin
  importar lo largas que sean las etiquetas. Con hasta 20 valores distintos siempre es categórica;
  por encima de eso lo sigue siendo mientras la mayoría de las filas repitan valores.
- **boolean**: True/False, sí/no
- **datetime**: Fecha/hora; para columnas de texto, solo se reconocen formatos explícitos
  (ISO 8601, `dd/mm/aaaa`, `mm/dd/aaaa`, `dd-mm-aaaa`, con o sin hora) con una proporción alta
  de aciertos, priorizando día-primero en casos ambiguos. Los valores numéricos con separador
  decimal no se confunden con fechas.
- **text**: Texto libre: más de 20 valores distintos y más de la mitad de las filas con un valor
  diferente (ej: comentarios, descripciones, direcciones)
- **identifier**: Códigos que nombran una entidad (IDs): alfanuméricos sin espacios, con letras y
  dígitos (ej: `CUST-00001`, `ORD-2024-000001`). También los **números enteros cuyo nombre de
  columna los marca como clave** (`id`, `customer_id`, `id_cliente`, `orderId`, `row_key`), con
  más valores distintos que el umbral de discretas; así `id` deja de recibir histograma,
  correlaciones y gráficos de dispersión. Un importe con valores casi únicos, como
  `monto_siniestro`, sigue siendo numérico: manda el nombre, no la unicidad. Se reconocen tanto los únicos por fila (clave
  primaria) como los que se repiten (clave foránea) en cuanto superan `cardinality_threshold`
  valores distintos. Por debajo de ese umbral un código sigue siendo una categoría útil para
  agrupar (ej: 40 `product_id`). Los identificadores nunca generan alertas de cardinalidad.
- **constant**: Un solo valor único

Estos tipos pueden sobreescribirse por columna mediante `column_types` en la configuración (ver
la sección de YAML más arriba); las columnas en `ignore` se excluyen de absolutamente todo el
análisis (calidad, univariado, relaciones, outliers, target, visualizaciones, tablas).

### Formatos de Entrada Soportados

| Extensión | Notas |
|---|---|
| `.csv`, `.tsv`, `.txt` | Delimitador, codificación y separador decimal auto-detectados si no se especifican |
| `.xlsx`, `.xls` | Primera hoja por defecto; `--sheet` acepta nombre o índice (un valor numérico se interpreta como índice) |
| `.parquet` | Sin opciones adicionales |
| `.json` | Un objeto, o un arreglo de registros (incluso anidados: se aplanan con notación `campo.subcampo`); si no es JSON válido se reintenta como JSON Lines |
| `.jsonl` | Siempre JSON Lines (un registro por línea) |

Cualquier celda que contenga una lista o diccionario (común en JSON anidado) se serializa como
texto JSON en vez de hacer fallar los pasos posteriores del análisis.

### Alertas de Calidad

El sistema genera alertas (alta, media, baja) para:

- >95% valores faltantes por columna (`missing_threshold`)
- Duplicados exactos: la severidad depende de `duplicate_threshold` (por encima del umbral → alta;
  por encima de la mitad del umbral → media; el resto → baja)
- Columnas constantes/quasi-constantes (`constant_threshold`)
- Alta cardinalidad (`cardinality_threshold`), solo en columnas categóricas, de texto o
  identificadores: los números y las fechas tienen muchos valores distintos por naturaleza
- Tipos mixtos en columnas
- Números almacenados como texto
- Desbalance de clases en el target (ver más abajo)
- Posibles fugas de datos (target)
- Columna target inexistente (solo en modo batch; en `analyze-file` este caso aborta el archivo)

Las alertas de target (desbalance, fugas) aparecen tanto en la sección «Alertas y
Recomendaciones» del HTML como en `summary.json → alerts`, junto con las de calidad de datos.

### Outliers

Detectados por:

- **IQR**: límites `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
- **MAD**: z-score robusto `|x - mediana| / MAD > 3.0`
- **Isolation Forest**: Anomalías multivariadas

### Relaciones

- **Numeric ↔ Numeric**: Correlación Pearson (con p-value)
- **Categorical ↔ Categorical**: Cramér's V (sesgada-corregida)
- **Categorical ↔ Numeric**: Correlation ratio (eta)
- **Multicolinealidad**: VIF (Variance Inflation Factor)

El reporte incluye dos mapas de calor:

- **Matriz de Correlación**: Pearson, solo entre columnas numéricas (de -1 a 1, con signo).
- **Asociación entre Variables**: cubre también las categóricas. Cada celda usa la medida que
  corresponde al par (Pearson, Cramér's V o eta) en vez de convertir las categorías a números.
  Codificarlas como 0, 1, 2… inventa un orden que no existe: en el dataset de stroke ese atajo
  convierte la asociación entre `work_type` y `age` (eta 0.68) en un engañoso -0.36.

### Target Analysis

Para columnas target:

- **Auto-detección**: Clasificación si ≤20 valores únicos; regresión si >20 numéricos
- **Balance de clases**: se marca **desbalanceado** cuando la clase mayoritaria representa más de
  `class_imbalance_threshold` del total (por defecto 0.8, es decir, más del 80%) —
  independientemente de cuántas clases haya
- **Feature Importance**: Pruebas estadísticas (chi-square, Kruskal-Wallis, mutual information)
- **Leakage**: Correlaciones sospechosamente altas (>0.99) o asociaciones categóricas casi perfectas
- **Columna inexistente**: en `analyze-file` el comando falla con un mensaje en español (columnas
  disponibles + sugerencia); en `analyze-batch` se registra una advertencia y el análisis de ese
  archivo continúa sin target

### Manejo de errores por paso

Cada paso del análisis (calidad, univariado, relaciones, outliers, target, visualizaciones,
tablas CSV, reporte HTML) se ejecuta de forma aislada: si uno falla, se registra el traceback
junto con el ID de correlación de la ejecución, el paso se agrega a `failed_steps`, se muestra en
una sección «Pasos con Error» del HTML, y el resto del análisis continúa con valores por defecto
seguros. Con `--strict`, cualquier error (de un dataset o de un paso) detiene la ejecución de
inmediato.

### Códigos de salida

| Código | Significado |
|---|---|
| 0 | Todo se procesó correctamente (sin errores de dataset ni de pasos) |
| 1 | Al menos un dataset, o al menos un paso de un dataset, falló |
| 2 | Error de uso o de configuración (ruta inexistente, clave de configuración desconocida, valor fuera de rango) |

## Testing

```powershell
# Todos los tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Con coverage
.\.venv\Scripts\python.exe -m pytest tests/ --cov=eda_pipeline --cov-report=term-missing

# Tests específicos
.\.venv\Scripts\python.exe -m pytest tests/test_modules.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_integration.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -v
```

## Linting y Formateo

```powershell
# Verificar
.\.venv\Scripts\python.exe -m ruff check .

# Formatear
.\.venv\Scripts\python.exe -m ruff format .

# Verificar formato sin modificar
.\.venv\Scripts\python.exe -m ruff format --check .
```

## Notebook de ejemplo

`notebooks/01_eda_template.ipynb` ejecuta el pipeline a través de su API de Python (sin CLI) sobre
`data/raw/ecommerce.csv` y muestra un resumen de los resultados (tipos de columnas, alertas,
balance de clases). Requiere `jupyter`/`nbconvert` además de las dependencias del proyecto.

## Generación de Datos de Prueba

`scripts/generate_sample_data.py` genera datasets sintéticos (e-commerce, salud, ventas) en
varios formatos dentro de `data/raw/`:

```powershell
.\.venv\Scripts\python.exe scripts/generate_sample_data.py
```

## Estructura del Proyecto

```
D:\DataScience\Pipeline/
├── src/eda_pipeline/              # Paquete principal
│   ├── __init__.py
│   ├── __main__.py                # Punto de entrada CLI
│   ├── cli.py                     # Comandos CLI
│   ├── config.py                  # Sistema de configuración
│   ├── logging_util.py            # Logging estructurado
│   ├── data_loader.py             # Carga de datos
│   ├── type_inference.py          # Inferencia de tipos
│   ├── data_quality.py            # Análisis de calidad
│   ├── univariate_analysis.py     # Estadísticas univariadas
│   ├── outlier_detection.py       # Detección de outliers
│   ├── relationships.py           # Análisis de relaciones
│   ├── target_analysis.py         # Análisis de target
│   ├── visualizations.py          # Generación de gráficos
│   ├── html_report.py             # Generación de reportes HTML
│   ├── tables.py                  # Exportación de tablas CSV
│   └── pipeline.py                # Orquestador principal
│
├── tests/                         # Suite de tests
│   ├── conftest.py                # Fixtures pytest
│   ├── test_modules.py            # Tests unitarios
│   ├── test_integration.py        # Tests de integración (pipeline completo)
│   └── test_cli.py                # Tests de CLI (click.testing.CliRunner)
│
├── config/                        # Configuración
│   └── default.yaml               # Config por defecto (generado con `init-config`)
│
├── notebooks/                     # Notebooks Jupyter
│   └── 01_eda_template.ipynb      # Ejemplo de uso vía la API de Python
│
├── data/                          # Datos
│   ├── raw/                       # Datos crudos
│   ├── interim/                   # Datos intermedios
│   ├── processed/                 # Datos procesados
│   ├── external/                  # Datos externos
│   └── .gitkeep
│
├── reports/                       # Reportes generados
│   └── .gitkeep
│
├── logs/                          # Logs de ejecución
│   └── .gitkeep
│
├── scripts/                       # Scripts auxiliares
│   └── generate_sample_data.py
│
├── pyproject.toml                 # Configuración del proyecto (build, ruff, pytest)
├── requirements.txt               # Dependencias (versiones compatibles)
├── requirements-dev.txt           # Dependencias de desarrollo
├── .gitignore                     # Git ignore
├── README.md                      # Este archivo
└── IMPLEMENTATION_PLAN.md         # Plan de implementación (fases)
```

### Datos y control de versiones

Git no versiona ningún archivo dentro de `data/`, solo la estructura de carpetas (con `.gitkeep`),
porque los datasets pueden contener información privada.

**Guarda siempre tus datasets dentro de `data/`.** Fuera de esa carpeta los CSV, Excel, Parquet o
JSON ya no se ignoran, para que los tests puedan incluir archivos de ejemplo.

Los reportes (`reports/`) y los logs (`logs/`) tampoco se versionan.

## Extensión del Pipeline

### Agregar un nuevo análisis

1. Crear módulo `src/eda_pipeline/mi_analisis.py`:

```python
from dataclasses import dataclass


@dataclass
class MiResultado:
    metrica1: float
    metrica2: int


def analizar_mi_cosa(df: pd.DataFrame) -> MiResultado:
    # Tu lógica
    return MiResultado(...)
```

2. Integrar en `pipeline.py`:

```python
from .mi_analisis import analizar_mi_cosa

# En EDAPipeline._analyze_dataset(), envuelto con self._run_step(...) para aislar errores:
mi_resultado = self._run_step("mi_analisis", failed_steps, analizar_mi_cosa, df, default=None)
```

3. Agregar tests en `tests/test_modules.py`
4. Actualizar reporte HTML si es necesario

### Agregar un nuevo formato de entrada

Editar `data_loader.py`:

```python
def load_mi_formato(file_path: Path, **kwargs) -> pd.DataFrame:
    # Cargar datos
    return df

# En detect_file_format() agregar:
elif suffix == ".miext":
    return "mi_formato"

# En load_data() agregar:
elif file_format == "mi_formato":
    df = load_mi_formato(file_path, **kwargs)
```

## Troubleshooting

### Error: "No module named 'eda_pipeline'"

```powershell
# Instalar en modo editable
.\.venv\Scripts\python.exe -m pip install -e .
```

### Error: "Encoding mismatch"

```powershell
# Especificar encoding manualmente si la auto-detección falla
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file archivo.csv --encoding cp1252
```

### Error: "La columna target '...' no existe en el dataset"

```powershell
# El mensaje ya lista las columnas disponibles y una sugerencia si el nombre es parecido.
# Verifique el nombre exacto de la columna (sensible a mayúsculas/minúsculas).
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --target nombre_exacto_columna
```

### Reportes vacíos o incompletos

```powershell
# Ejecutar en modo verbose para ver detalles (nivel DEBUG)
.\.venv\Scripts\python.exe -m eda_pipeline analyze-file data.csv --verbose
# Ver logs en logs/ (hermana de --output-dir, no dentro de la carpeta del reporte)
# El summary.json de cada reporte incluye "failed_steps" si algún paso falló
```

### Acentos o símbolos raros al redirigir la salida (`> salida.txt`, `| Select-String ...`)

En Windows con página de códigos cp1252, al redirigir la salida los acentos pueden verse como `�`
y ✅/❌ como `?`. El comando no falla y los reportes no se ven afectados. Para que se vean bien,
activa el modo UTF-8 de Python en esa sesión de PowerShell antes de ejecutar:

```powershell
$env:PYTHONUTF8 = "1"
```

## Limitaciones conocidas y trabajo futuro

### Limitaciones Actuales

- Los reportes HTML se generan sin caché (grandes para datasets muy amplios >100 columnas, o con
  muchas visualizaciones habilitadas)
- Visualizaciones limitadas a 20-30 gráficos por tipo (configurable vía `visualizations`)
- Sin análisis de texto avanzado (TF-IDF, topic modeling)
- Sin validación de datos (schema/constraints)

### Mejoras Planificadas

- [ ] Exportar a Power BI / Tableau templates
- [ ] Validación de datos con Great Expectations
- [ ] Análisis de series temporales (decomposition, autocorrelation)
- [ ] Análisis de texto (word clouds, TF-IDF, NLP)
- [ ] Recomendaciones automáticas de preprocesamiento
- [ ] Integración con MLflow para tracking
- [ ] API REST para uso remoto
- [ ] Dashboard interactivo (Dash/Streamlit)
- [ ] Soporte para bases de datos (SQL)

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Clonar el repo
2. Crear rama feature (`git checkout -b feature/nueva_feature`)
3. Hacer cambios + tests
4. Verificar: tests, lint, coverage
5. Commit (`git commit -m "feat: descripción"`)
6. Push y pull request

## Licencia

MIT

## Autor

Generado con Claude Code – Análisis Exploratorio de Datos automatizado.
