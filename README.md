# Exploratory Data Analysis Automatic Pipeline

**One command turns any tabular file into a self-contained HTML report** — data quality, statistics,
outliers, relationships, target analysis, 40+ charts, and a preprocessing plan where every
recommendation carries the measurement that produced it.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![CI](https://github.com/edu3250/Exploratory-Data-Analysis-Automatic-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/edu3250/Exploratory-Data-Analysis-Automatic-Pipeline/actions/workflows/ci.yml)
![coverage 93%](https://img.shields.io/badge/coverage-93%25-brightgreen)
![ruff](https://img.shields.io/badge/lint-ruff%20clean-purple)
![license MIT](https://img.shields.io/badge/license-MIT-lightgrey)

![The report the pipeline generates](docs/report-preview.png)

<sub>A real report from `data/raw/spaceship_titanic/train.csv` (8 693 × 14). Below the summary, the chart
the pipeline drew for `RoomService`: 65.5% of its rows are 0, so the column is drawn a second time
without them — that is the only way its distribution is visible at all.</sub>

---
## Target analysis
<img width="1196" height="813" alt="Target section: the class balance of Transported, and every column ranked by how much it separates the target" src="https://github.com/user-attachments/assets/622eeeb9-efcc-4952-9061-ae1b61df3d10" />

<sub>The Target section of the same report, run with <code>--target Transported</code>. The two classes are
balanced (50.4% / 49.6%), so no imbalance is flagged. Below, every column ranked by how much it
separates the target, on one 0-to-1 scale whatever the test: CryoSleep leads at 0.47, ahead of the
spending columns (0.21–0.24). The effects are corrected for chance, so a column cannot rank high just
by having many categories.</sub>

---

## Missing values
<img width="1200" height="786" alt="Missing values matrix: one white line per missing cell, scattered across twelve columns" src="https://github.com/user-attachments/assets/12db49df-4a25-495f-b787-2248cb7bccd9" />

<sub>Each white line is a missing cell, in a random sample of 500 of the 8 693 rows. Twelve of the 14
columns are missing on about 2% of the rows each (2.1–2.5%), and the gaps rarely line up: 24% of the
rows have at least one, but only 2.5% have two or more. Dropping incomplete rows would throw away a
quarter of the data, so the preprocessing plan fills the columns one by one.</sub>

---

## Relationships between columns and target
<img width="1197" height="529" alt="Grouped bars of HomePlanet and CryoSleep against Transported" src="https://github.com/user-attachments/assets/927f193e-70a4-4410-aff9-20da667247be" />

<sub>Each categorical column against the target, with every label a share of the chart's total. Read
the bars within each group: 82% of the passengers in cryosleep were transported, against 33% of
those awake; by home planet, 66% of Europa's passengers against 42% of Earth's.</sub>

---

## How to use

```bash
python -m venv .venv && .venv/Scripts/activate   # source .venv/bin/activate on Linux/macOS
python -m pip install -e .

# one file — CSV, TSV, Excel, Parquet or JSON; --target is optional
eda analyze-file your_data.csv --target your_label_column

# every supported file in a folder, in one run; a file that fails does not abort the rest
eda analyze-batch data/raw/

# a YAML you can edit: delimiters, encodings, column types, thresholds, which charts to draw
eda init-config
```

Delimiter, encoding, decimal separator and column types are detected on their own; `--delimiter`,
`--encoding`, `--decimal`, `--sheet`, `--sample-size` and `--output-dir` override that when you need
to. `eda --help` has the rest.

Each run writes one folder under `reports/`:

```
reports/your_data_20260917_112601/
├── report.html      # the whole report, offline, images embedded — just open it
├── summary.json     # every measurement, machine-readable
├── plots/           # one PNG per chart (40 for the report above)
└── tables/          # 7 CSV tables (stats, correlations, outliers, alerts, the plan)
```

`data/` is not tracked, so bring your own file — the one in the screenshot is Kaggle's Spaceship
Titanic training set.

## What the report contains

| Section | What you get |
|---|---|
| Data quality | Missing values, duplicates, constant and quasi-constant columns, high cardinality, mixed types, numbers stored as text |
| Type inference | Each column classified as continuous, discrete, categorical, boolean, datetime, time of day, text, identifier or constant — this is what decides the analysis it receives |
| Univariate | Descriptive statistics, normality tests, frequency tables, histograms and boxplots, and, for skewed columns, QQ plots before and after the transformation that straightens them |
| Outliers | IQR, MAD z-score and a multivariate Isolation Forest whose cut is drawn from each dataset's own anomaly scores |
| Relationships | Pearson, Cramér's V and the correlation ratio (the last two corrected for chance), two heatmaps, scatter plots of the strongest pairs and a pair plot coloured by group |
| Multicollinearity | Exact linear combinations written as equations (a total and its parts, the dummy trap), VIF, the columns to consider dropping and the VIF left after dropping them |
| Target analysis | Class balance, leakage checks, and the features most related to the target on one 0-to-1 scale |
| Preprocessing plan | What to convert, drop, impute, encode, transform and scale — every line with the measurement behind it |


## Notes and roadmap

- Reports are self-contained, so they get large on very wide datasets (>100 columns). Chart counts
  are configurable.
- Next: generated scikit-learn preprocessing code, ordinal levels declared in the config, missingness
  that carries signal, text analysis (TF-IDF, topic modelling), and schema validation.

## License

MIT — see [LICENSE](LICENSE).
