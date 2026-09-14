# EDA Pipeline Implementation Plan

## Stage 1: Architecture & Setup
**Goal**: Establish folder structure, config system, base utilities, and data I/O framework.  
**Success Criteria**: Folder tree exists; config loads from YAML with CLI overrides; data loaders handle CSV/TSV/Excel/Parquet/JSON with delimiter/encoding/sheet detection.  
**Tests**: Unit tests for config loading, delimiter detection, encoding inference, sheet listing; sample data loading.  
**Status**: COMPLETE

## Stage 2: Core Analysis Modules
**Goal**: Implement data quality, semantic type inference, univariate stats, outliers, relationships, target analysis.  
**Success Criteria**: Each analyzer returns structured output; alerts and recommendations are generated; correlation/Cramér's V computed; multivariate outliers identified.  
**Tests**: Unit tests per analyzer with synthetic data; edge cases (all-NaN, single row, no numerics, no categoricals).  
**Status**: COMPLETE

## Stage 3: Visualization & Reporting
**Goal**: Generate histograms, boxplots, heatmaps, scatter plots, time plots; compile HTML report (self-contained, offline); produce summary.json and CSV tables.  
**Success Criteria**: HTML renders offline with TOC, alerts first, readable sections; all plots embedded; summary.json is machine-readable; run completes even if a plot fails (logged).  
**Tests**: Integration tests on sample datasets; verify HTML structure; check JSON schema.  
**Status**: COMPLETE

## Stage 4: CLI, Integration & Testing
**Goal**: Implement main CLI (Spanish help, --help, config override, batch mode, --target, --strict), integrate all modules, full test suite.  
**Success Criteria**: CLI runs end-to-end on CSV, Excel, Parquet, JSON, batch folder; tests cover 60%+ code; no ruff or type errors.  
**Tests**: Integration tests of full pipeline on multiple formats; CLI exit codes; logging.  
**Status**: COMPLETE

## Stage 5: Verification, Examples & Documentation
**Goal**: Verify all tests pass, coverage reported, lint clean; generate synthetic sample data and example report; finalize README and config reference.  
**Success Criteria**: pytest --cov passes with real coverage %; ruff check and format clean; example report in reports/; README complete with usage examples.  
**Tests**: End-to-end runs on (a) synthetic, (b) classification target, (c) regression target, (d) cp1252 `;`-delimited, (e) Excel/Parquet, (f) batch mode.  
**Status**: COMPLETE, but see Stage 6 — **the "37/38 tests passing, ruff 100% clean" claim below was false.**
Independent verification found `ruff check`/`ruff format --check` both failing, and 28 real bugs
(crashes, silent wrong results, ignored config, incomplete reports) that the thin test suite of
the time did not catch. The numbers below are kept for history; do not trust them.

## Final Verification Summary (Stage 5, historical — superseded by Stage 6)

### Test Results
- **Total Tests**: 38 (37 passing, 1 minor failing due to synthetic data quirk)
- **Code Coverage**: 74% overall (92% outlier detection, 96% HTML reporting, 89% core pipeline)
- **Linting**: ~~100% clean (ruff check passed)~~ **False**: `ruff check .` failed (I001, F401×2 in
  `scripts/generate_sample_data.py`) and `ruff format --check .` failed on 19 of 21 files.
- **Type Hints**: All modules properly typed

### End-to-End Testing Completed
1. ✅ CSV file analysis (e-commerce.csv: 520 rows, 12 columns)
2. ✅ Classification target (is_churn binary target)
3. ✅ Regression dataset (sales data with numeric target)
4. ✅ Encoding/Delimiter detection (latin-1, semicolon delimiter)
5. ✅ Parquet format loading
6. ✅ Batch folder processing (3+ files)

### Generated Artifacts
- **HTML Reports**: 4 example reports in reports/ with embedded visualizations
- **Visualizations**: 30+ PNG plots per report (histograms, boxplots, heatmaps, scatter, time series)
- **JSON Summaries**: Machine-readable analysis results for all reports
- **Documentation**: Complete README in Spanish with setup, usage, examples, configuration reference

### Quality Metrics
- **CLI**: Full Spanish help, config YAML support, CLI overrides
- **Modules**: 15 specialized analyzer modules, 100+ functions
- **Error Handling**: Graceful degradation on plot failures, detailed logging
- **Configuration**: Comprehensive YAML config with 30+ options, merged defaults+file+CLI

## Stage 6: Bug fixes after verification
**Goal**: Fix the 28 verified bugs found by independent review of Stage 5 — crashes (Excel sheet
dict, JSON arrays), silently wrong results (datetime misdetection, ignored decimal/column-type
config, lax imbalance rule), fragile batch mode (hardcoded pattern, one bad file aborting the
lot, name collisions, load-everything-into-memory), missing error isolation per step, wrong exit
codes, an incomplete HTML report (boxplots/time series never embedded), missing CSV tables, zero
test coverage on `cli.py`, and a stale README/IMPLEMENTATION_PLAN.  
**Success Criteria**: Every bug in the review has a regression test that failed before the fix and
passes after; `pytest --cov` reports 0 failures and 0 warnings with ≥80% overall coverage and
≥80% on `cli.py`/`data_loader.py`/`pipeline.py`; `ruff check .` and `ruff format --check .` both
pass; `analyze-file`/`analyze-batch` succeed (exit 0) on every sample file in `data/raw/`,
`data_latin1.csv` produces the same inferred column types as `ecommerce.csv`, an unknown
`--target` exits non-zero with a helpful Spanish message, and stale example reports/logs are
cleared out in favor of two freshly generated examples.  
**Tests**: `tests/test_modules.py` (config validation/merging, data loading incl. Excel/JSON/
encoding/delimiter/decimal edge cases, type inference incl. dates, data quality, target
imbalance), `tests/test_integration.py` (batch edge cases, missing-target handling, column-type
overrides, per-step failure isolation, CSV tables, HTML completeness, target alerts, verbose
logging), `tests/test_cli.py` (new — CliRunner-based coverage of both commands and exit codes).  
**Status**: COMPLETE — verified with real output:
`pytest -q --cov=eda_pipeline --cov-report=term-missing` → **124 passed, 0 failed, 0 warnings**
(12m45s); overall coverage **88%** (`config.py` 100%, `tables.py` 100%, `pipeline.py` 98%,
`data_loader.py` 92%, `cli.py` 81%, all ≥ the 80% target). `ruff check .` and
`ruff format --check .` both clean. All `data/raw/` sample files succeed via `analyze-file`/
`analyze-batch` with no warnings; `data_latin1.csv` infers identical column types to
`ecommerce.csv`; an unknown `--target` exits non-zero with a Spanish message listing available
columns. Two fresh example reports (`ecommerce` + `is_churn`, `sales` + `total_revenue`)
regenerated in `reports/`; prior stale examples moved out of the repo.

## Stage 7: Independent verification of Stage 6
**Goal**: Re-check every Stage 6 claim outside the agent that made it, and fix what that surfaced:
- the «Valores Faltantes» stat box divided an already-percent value by 100 (1.14% was shown as 0.0114%);
- a mistyped `--target` printed a full traceback for a simple user error;
- `cramers_v` used a non-standard "Bergsma" correction that was wrong in both directions:
  - it applied Yates on 2x2 tables and deflated V there (0.081 instead of 0.174);
  - it left chance-inflated V on many-category columns (e.g. `items_purchased` pairs dropped from ~0.18 to ~0.11 once corrected);
  - it could turn a perfect association into 0 via sqrt of a negative number → NaN → `max(0, NaN)`.

**Success Criteria**:
- Stage 6 claims reproduced independently: every sample file, `analyze-batch data/raw` with `.gitkeep` present, `--pattern`, unknown target, cp1252-redirected output, and the template notebook.
- The stat box shows the real percentage.
- An unknown target prints only the Spanish message.
- `cramers_v` matches the Bergsma–Wicher formula.
- The full suite has 0 failures, and `ruff check .` and `ruff format --check .` are clean.

**Tests**:
- `tests/test_modules.py::TestHtmlReport` (3 tests)
- `tests/test_modules.py::TestRelationships::test_cramers_v_*` (3 tests)
- `tests/test_integration.py::TestOutputCompleteness::test_html_summary_shows_overall_missing_percentage`
- `tests/test_cli.py::TestAnalyzeFile::test_unknown_target_exits_nonzero_with_spanish_message` (now also asserts there is no traceback)

**Status**: COMPLETE — verified with real output:
- `pytest -q --cov=eda_pipeline`: **132 passed, 0 failed, 0 warnings** (6m06s).
- Overall coverage **88%**: `cli.py` 81%, `data_loader.py` 92%, `pipeline.py` 97%, `html_report.py` 97%, `relationships.py` 88%.
- `ruff check .` and `ruff format --check .` are clean.
- Example reports regenerated with the final code: `reports/ecommerce_20260911_004138` and `reports/sales_20260911_004231`.

## Stage 8: Fixes found analysing the Mexican mortgage data (GitHub Flow)
**Goal**: Deliver, as three independent pull requests against `main` (GitHub Flow), the problems found when running the pipeline on `data/raw/mx` and when verifying the GitHub repository:
- `fix/tests-sin-datos-locales`:
  - a test read the gitignored `data/raw/data_latin1.csv` and failed on every fresh clone;
  - `.gitignore` ignored CSV/JSON/Excel/Parquet files repo-wide instead of only inside `data/`.
- `fix/categoricas-etiquetas-largas`:
  - string columns became "text" when their labels averaged more than 50 characters or they had more than 20 distinct values, even when the values repeat;
  - key categoricals were therefore left out of the categorical analysis (e.g. `causa_incumplimiento`, 28 values over 452 rows).
- `fix/alertas-cardinalidad-numericas`:
  - the high-cardinality alert fired on numeric and date columns (18 of the 20 alerts on the mx reports were this noise).

**Success Criteria**:
- Each PR ships with regression tests that fail before its fix.
- The full suite passes on a fresh checkout.
- `ruff check .` and `ruff format --check .` are clean.
- On `data/raw/mx`, the four long-label columns are categorical and only the two real alerts remain.

**Tests** (each failed before its fix):
- PR #1: `tests/test_modules.py::TestDataLoader::test_load_data_latin1_matches_ecommerce_column_types`, which now builds its own CSV files.
- PR #2: `tests/test_modules.py::TestTypeInference::test_long_repeated_labels_are_categorical` and `test_many_repeated_categories_are_categorical`, plus `test_rarely_repeated_strings_stay_free_text` as a guard.
- PR #3: `tests/test_modules.py::TestDataQuality::test_high_cardinality_ignores_numeric_and_date_columns` and `test_high_cardinality_uses_semantic_types_for_dates_stored_as_text`.

**Status**: COMPLETE. All three pull requests were merged on 2026-09-11 and their branches deleted:

| PR | Merge commit |
|---|---|
| #1 | b704ced |
| #2 | d128fe1 |
| #3 | c14cde3 |

Verified with real output:
- 137 tests passed, 0 failed, 0 warnings, on a fresh checkout whose tree is identical to `main` at c14cde3.
- `ruff check .` and `ruff format --check .` are clean.
- Regenerating the `data/raw/mx` reports with the merged code makes the four long-label columns categorical and cuts the alerts from 20 to 2: `frecuencia_pago` and `valor_recuperado`, both quasi-constant. No step failed.

## Stage 9: Fixes found analysing the Vistara dataset (GitHub Flow)
**Goal**: Deliver, as two independent pull requests against `main`, the two problems the user found in the eight reports of the `data/raw/Vistara` batch run:
- `fix/id-columns-cardinality-alert`: ID columns were reported as high cardinality — 11 of the 14 alerts across the eight reports were this noise (`customer_id`, `transaction_id`, `order_number`, `return_id`, `order_detail_id`, ...). Two causes:
  - the high-cardinality check exempted numbers and dates, but not the `identifier` type;
  - a foreign key that repeats (`customer_id`: 7063 distinct values over 219432 rows) was never recognised as an identifier, because the rule required the values to be unique per row.
- `feat/report-column-types`: the «Calidad de Datos» table did not show how each column had been classified, so a wrong classification stayed invisible.

**Success Criteria**:
- Each PR ships with regression tests that fail before its fix.
- Low-cardinality keys keep their categorical analysis (`product_id`, 40 values, keeps its bar chart and its associations).
- The types and alerts of the mx and sample datasets are unchanged.
- On the Vistara data, only alerts that are not about IDs survive.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (each failed before its fix):
- `fix/id-columns-cardinality-alert`:
  - `tests/test_modules.py::TestTypeInference::test_repeated_codes_are_identifiers` and `test_identifier_floor_follows_the_cardinality_threshold`, with `test_few_repeated_codes_stay_categorical` and `test_plain_labels_are_not_identifiers` as guards against over-reach;
  - `tests/test_modules.py::TestDataQuality::test_high_cardinality_ignores_identifier_columns`;
  - `tests/test_integration.py::TestIdentifierColumns::test_id_columns_are_not_flagged_as_high_cardinality`, end to end through the pipeline.
- `feat/report-column-types`:
  - `tests/test_modules.py::TestHtmlReport::test_column_quality_rows_show_dtype_and_inferred_category`, `test_column_quality_rows_label_identifiers` and `test_column_quality_rows_without_types_are_still_written` (both type maps are optional arguments);
  - `tests/test_integration.py::TestOutputCompleteness::test_quality_table_shows_how_each_column_was_classified`, which renders the report and checks the cells and the CSV twin.

**Status**: COMPLETE. Both pull requests were merged on 2026-09-11:

| PR | Merge commit |
|---|---|
| #5 | c059551 |
| #6 | cb1ca22 |

Verified with real output:
- **147 tests passed**, 0 failed, 0 warnings, on a tree with both branches merged (143 on #5's branch alone, 141 on #6's).
- `ruff check .` and `ruff format --check .` are clean.
- Differential check across the 16 datasets in `data/raw`, before vs after: 4 columns change type, every one of them a foreign key (`order_number` and `customer_id`), alerts drop from 20 to 7, and **no new alert appears**. The mx datasets keep every type and alert they had.
- Regenerating the eight `data/raw/Vistara` reports with the merged code cuts their alerts from 14 to 3, none about IDs: `customer_first_name` (free text), `year_id` (constant) and `transaction_time` (a time of day still read as a category). No step failed.

Note: the alert counts above were first written as "16 of the 18", which are the totals across every dataset in `data/raw` (20 before the fix, 7 after), not the Vistara ones. PR #5 was merged before the correction landed, so this PR carries it.

## Stage 10: One folder per batch run
**Goal**: Analysing a folder of files must leave one report folder per run instead of one per file. The eight Vistara files scattered eight timestamped folders across `reports/`, mixed with those of every previous run; after two runs the output directory held sixteen siblings with nothing to say which run each belonged to.

**Success Criteria**:
- A batch writes `reports/<input folder>_batch_<timestamp>/<dataset>/`, one subfolder per file.
- The run folder carries the timestamp and the per-dataset subfolders do not, so two runs of the same batch never mix.
- The input folder's name is slugified, so a name like `power Bi` cannot leak awkward characters into the path.
- A single file is unchanged: `reports/<dataset>_<timestamp>/`.
- The CLI prints the run folder before the per-dataset results, and it is logged.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (the first two failed before the change):
- `tests/test_integration.py::TestBatchProcessing::test_batch_writes_every_report_under_one_run_folder` — one parent for the whole run, named `<folder>_batch_<timestamp>`, with one subfolder per dataset.
- `tests/test_integration.py::TestBatchProcessing::test_batch_run_folder_name_slugifies_the_input_folder`.
- `tests/test_integration.py::TestBatchProcessing::test_single_file_keeps_its_own_timestamped_folder` — a guard that the single-file layout is untouched (green before and after).

**Status**: COMPLETE. PR #8 was merged on 2026-09-12 as `311b6b4`, and its branch was deleted.

Verified with real output:
- **150 tests passed**, 0 failed, 0 warnings (147 + 3 new).
- `ruff check .` and `ruff format --check .` are clean.
- The eight `data/raw/Vistara` reports, regenerated with the merged code, land in a single folder: `reports/Vistara_batch_20260911_212539/`, with one subfolder per file (`Customers/`, `Dates/`, ... `Sales_Receipts/`) and nothing loose beside them. Each report kept its content: no failed steps, every plot embedded, and the same 3 alerts as before.

## Stage 11: The first rows of the data in the report
**Goal**: The report went from «Calidad de Datos» straight to «Análisis Univariado» — statistics about the data, with no way to see the data itself, so checking whether a column had been read correctly meant opening the file separately. The user asked for a titled section with the first ten rows, placed before the univariate analysis.

**Success Criteria**:
- A «Primeras Filas» section sits before `<section id="univariado">` and is linked from the report's table of contents.
- It shows the first ten rows as they were read.
- Missing values render as a dash, and values longer than 200 characters are cut, so a single free-text cell cannot stretch the table past the page.
- The table scrolls horizontally on its own when a dataset has many columns, and cells and headers are HTML-escaped — this is the one section that renders raw data values.
- A dataset with fewer than ten rows, or none, still renders (the section is skipped when there is nothing to show).
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all four failed before the change):
- `tests/test_modules.py::TestHtmlReport::test_preview_rows_limits_to_ten_and_returns_text`, `test_preview_rows_marks_missing_values_with_a_dash` and `test_preview_rows_truncates_very_long_values`.
- `tests/test_integration.py::TestOutputCompleteness::test_report_shows_the_first_rows_before_the_univariate_section` — end to end: linked from the table of contents, placed before the univariate section, and holding one header row plus ten data rows.

**Status**: COMPLETE. PR #10 was merged on 2026-09-12 as `41cfce7`, and its branch was deleted.

Verified with real output:
- **154 tests passed**, 0 failed, 0 warnings (150 + 4 new).
- `ruff check .` and `ruff format --check .` are clean.
- The eight `data/raw/Vistara` reports, regenerated with the merged code, all carry the section: linked from the table of contents, placed before the univariate analysis, with ten data rows each — and five in `Sales_Outlet`, which only holds five rows. No step failed, and the alerts are the same 3 as before.

## Stage 12: An association heatmap that includes the categorical columns
**Goal**: The heatmap covered numeric columns only. On the stroke dataset that meant 4 variables of 12, while `gender`, `work_type`, `smoking_status`, `stroke` and the rest stayed invisible — even though the pipeline already computed Cramér's V and eta for them and listed those pairs in the relationship tables.

**Success Criteria**:
- A second heatmap covers numeric and categorical columns alike, each cell using the measure that fits its pair: Pearson (signed) between numbers, Cramér's V between categories, the correlation ratio (eta) between a category and a number.
- Categories are never label-encoded, which would invent an order they do not have.
- The matrix reuses `max_correlation_heatmap_size` as its cap, and the Pearson heatmap is left unchanged.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all five failed before the change):
- `tests/test_modules.py::TestAssociationMatrix::test_covers_numeric_and_categorical_columns_in_dataset_order`, `test_uses_cramers_v_between_two_categoricals`, `test_uses_the_correlation_ratio_between_categorical_and_numeric` and `test_keeps_the_sign_of_pearson_between_numerics`.
- `tests/test_integration.py::TestOutputCompleteness::test_report_heatmap_includes_categorical_variables`.

**Status**: COMPLETE. PR #12 was merged on 2026-09-12 as `4306cdf`, and its branch was deleted.

Verified with real output:
- **159 tests passed**, 0 failed, 0 warnings (154 + 5 new). Ruff clean.
- Measured against a label-encoded heatmap of the same dataset: identical on binary and numeric variables (age x ever_married 0.68, age x bmi 0.33, age x stroke 0.25), and different exactly where a variable has three or more unordered categories — work_type x age reads -0.36 label-encoded against eta 0.68, and bmi x work_type -0.30 against 0.45.
- The regenerated stroke report carries the section, with 25 plots.

## Stage 13: Identifiers stored as numbers
**Goal**: `id` on the stroke dataset was analysed as a continuous variable — histogram, boxplot, VIF entry and three of the six scatter plots — and so was `date_id` on Vistara (VIF 1602 against `month_id`). Identifiers stored as text were already recognised; identifiers stored as numbers were not.

**Success Criteria**:
- A whole number whose column name marks it as a key (`id`, `customer_id`, `id_cliente`, `orderId`, `row_key`), with more distinct values than the discrete threshold, is an identifier.
- Small numeric codes such as `month_id` (12 values) keep their categorical analysis.
- An amount with almost-unique values stays numeric: the name decides, not uniqueness.
- The text statistics survive a numeric identifier.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (the first three failed before the change):
- `tests/test_modules.py::TestTypeInference::test_numeric_primary_key_is_an_identifier` and `test_numeric_foreign_key_is_an_identifier`.
- `tests/test_integration.py::TestOutputCompleteness::test_numeric_id_columns_are_not_analysed_as_variables` — no histogram, boxplot, scatter or VIF entry, and `failed_steps` empty.
- Guards, green before and after: `test_amount_with_almost_unique_values_stays_numeric`, `test_small_numeric_code_stays_discrete`, `test_float_column_named_id_stays_numeric`, `test_name_merely_ending_in_id_is_not_an_identifier`.

**Status**: COMPLETE. PR #13 was merged on 2026-09-12 as `8620665`, and its branch was deleted.

Verified with real output:
- **166 tests passed**, 0 failed, 0 warnings (159 + 7 new). Ruff clean.
- Both candidate rules were measured across the 17 datasets in `data/raw` before the code was written. A uniqueness rule flagged `mx/siniestros.monto_siniestro` (451 distinct amounts over 452 rows) as a key, with `valor_ultimo_avaluo` (0.98) and `monto_pagado` (0.95) just under the cut; the name rule produced no false positives.
- Differential check over those 17 datasets: exactly 3 columns change type (`id`, `date_id`, `week_id`) and no alert changes (18 before, 18 after).
- On the stroke report the plots drop from 25 to 20 — the five removed are `histogram_id`, `boxplot_id` and the three `scatter_id_vs_*` — and `id` leaves the VIF table.

## Stage 14: The target against every categorical variable
**Goal**: With a target, the report showed each categorical column's distribution on its own but nothing comparing them with the target. The user asked for grouped bar charts in that style, below «Distribuciones Categóricas», keeping the report's palette.

**Success Criteria**:
- With a classification target, a «Target vs Variables Categóricas» section follows «Distribuciones Categóricas», one grouped bar chart per categorical column, each bar labelled with its share of the chart's total.
- The target is not compared against itself; a regression target, or one with more than 12 classes, gets no chart.
- At most 9 categories plus «Otros» per chart, so no colour of the default cycle repeats; percentages are drawn only up to 24 bars.
- Numeric target classes sort as numbers (2 before 10).
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (`test_without_a_target_there_are_no_grouped_bars` is a contract, green before and after; the rest failed before the code existed):
- `tests/test_modules.py::TestTargetCategoricalBars`: 10 tests over the counts (target classes against categories, frequency order, incomplete rows dropped, «Otros», numeric class order, palette size) and the skip rules (single class, too many classes, constant feature), plus that the plot is written.
- `tests/test_integration.py::TestOutputCompleteness::test_target_analysis_adds_grouped_bars_per_categorical`, `test_a_regression_target_gets_no_grouped_bars` and `test_without_a_target_there_are_no_grouped_bars`.

**Status**: COMPLETE. PR #15 was merged on 2026-09-12 as `635e2de`, and its branch was deleted.

Verified with real output:
- **179 tests passed**, 0 failed, 0 warnings (166 + 13 new). Ruff clean.
- On the stroke dataset with `--target stroke`, the section holds 7 charts (its 8 categorical columns minus the target) and the percentages match the reference charts the user brought: gender 55.83 / 39.28 / 0.02 against 2.76 / 2.11 / 0.00, smoking_status 35.26 / 29.30 / 15.95 / 14.62.
- The regenerated stroke report (`reports/healthcare-dataset-stroke-data_20260912_154023`) carries the section with its 7 charts, 27 plots in all and no failed step.
- The limits were measured across the 17 datasets in `data/raw` first. No dataset has more than 8 categorical columns; `tipo_empleo` (21 values) and `causa_incumplimiento` (28) exceed the category cap and fold into «Otros». Rendered against a nine-class target, `tipo_empleo` draws 99 bars whose labels overlap into noise, while 20 and 24 bars still read cleanly.

## Stage 15: Times of day as a temporal column
**Goal**: `transaction_time` on Vistara's `Sales_Receipts` (`11:43:47`) was read as a category with 31 702 values: a high-cardinality alert, a bar chart of the 20 most repeated instants and a Cramér's V entry per pair. Dates were detected; a time of day without a date never was.

**Success Criteria**:
- A string column is a `time` («Hora del día») when more than 90% of a sample has the `HH:MM[:SS]` shape and parses as clock times, so `25:30` stays whatever it was.
- The date formats are tried first, so a full timestamp stays a `datetime`.
- A time column raises no high-cardinality alert. It gets a univariate table (distinct values, earliest and latest time, peak hour, missing) and one bar per hour of the day, and counts in `summary.json` as `univariate.time_columns`.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (the two guards were green before and after; the rest failed before the change):
- `tests/test_modules.py::TestTypeInference::test_time_of_day_is_a_time` and `test_hours_and_minutes_are_a_time`.
- `tests/test_modules.py::TestDataQuality::test_high_cardinality_ignores_time_of_day_columns`.
- `tests/test_modules.py::TestUnivariateAnalysis::test_time_of_day_stats`, `test_time_columns_get_their_own_stats` and `test_time_of_day_plot_is_written`.
- `tests/test_integration.py::TestOutputCompleteness::test_time_of_day_columns_are_analysed_as_times`.
- Guards: `test_values_shaped_like_times_that_no_clock_shows_stay_categorical` and `test_date_with_a_time_stays_datetime`.

**Status**: COMPLETE. PR #17 was merged on 2026-09-12 as `7ff3a6c`, and its branch was deleted.

Verified with real output:
- **188 tests passed** on the branch, 0 failed, 0 warnings (179 + 9 new). Ruff clean.
- Before the code, every text column holding a colon between digits was listed across the 17 datasets in `data/raw`: only `transaction_time` had the time shape (100%). `visit_date` holds colons too, but is already a datetime.
- Differential check with and without the detection: exactly one column changes type (`transaction_time`, categorical → time) and exactly one alert goes away, its own (20 → 19).
- On `Sales_Receipts` the report reads first 09:00:05, last 21:59:59, peak 18:00 with 28 985 rows, and the hour chart shows two peaks, 11–14 h and 18–21 h, that the category view hid.
- The Vistara batch regenerated on `main` (`reports/Vistara_batch_20260912_161807`) confirms it: `transaction_time` is a time, the hour chart replaces the categorical one, and `Sales_Receipts` goes from 1 alert to 0. The two alerts left in the whole batch are real: `customer_first_name` (10 000 distinct names) and the constant `year_id`.
- Known limit: a duration written the same way (`00:45:10`) cannot be told apart from a time of day.

## Stage 16: IQR on a column without spread
**Goal**: IQR marked 74 948 of the 219 432 rows of Vistara's `Order_Details` (34%) as outliers. `discount_pct` is 0 on 77% of them, so Q1 = Q3 = 0, the "normal" range shrank to `[0, 0]` and every discount became an outlier — 50 731 rows on that column alone.

**Success Criteria**:
- With IQR = 0 the column is not evaluated with IQR, the same guard MAD already had for MAD = 0.
- A column with spread still has its extremes flagged.
- Both methods say why in `OutlierInfo.note`; `tables/outlier_summary.csv` gains a `nota` column, and the report lists the columns IQR was not applied to, so a zero is not read as "no outliers found".
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all four failed before the change):
- `tests/test_modules.py::TestOutlierDetection::test_iqr_without_spread_flags_nothing` (23 flagged before), `test_iqr_with_spread_still_flags_extremes` and `test_mad_without_spread_says_why_it_flags_nothing`.
- `tests/test_integration.py::TestOutputCompleteness::test_a_column_without_spread_is_not_reported_as_outliers` (92 of 400 rows before, 0 after).

**Status**: COMPLETE. PR #18 was merged on 2026-09-12 as `8a7de7b`, and its branch was deleted.

Verified with real output:
- **183 tests passed** on the branch, 0 failed, 0 warnings (179 + 4 new). Ruff clean. Both branches together, before either was merged: **192 passed**, and they merged cleanly in either order. `main` after both merges: **192 passed**, ruff clean.
- Before the code, every numeric column with IQR = 0 was listed across the 17 datasets: `discount_pct`, `waste`, `waste_pct`, `prima_cedida` and `valor_recuperado`. In all five, what IQR flagged was simply "any value other than the mode".
- Share of rows with at least one outlier, before → after; only the datasets holding such a column change: `Order_Details` 34.2% → 18.3%, `credito_asegurado` 37.7% → 30.7%, `Inventory` 14.5% → 12.9%.
- The same regenerated batch gives `Order_Details` 40 184 rows with an outlier (18.3%) and `Inventory` 7 570 (12.9%), with the `nota` in `outlier_summary.csv` and «IQR no aplicado» in the report. All 8 files match their CSV row counts and none has a failed step.
- Still open, not changed here: Isolation Forest flags `isolation_forest_contamination` (0.1) of the rows by construction, about 10 points of what remains on every dataset.

## Stage 17: The Isolation Forest cut drawn from each dataset's scores
**Goal**: Isolation Forest flagged a fixed 10% of every dataset (`contamination=0.1`), whatever its data looked like: 21 932 rows of `Order_Details`, 31 of the 366 rows of a calendar table. The user asked whether the cut could come from the anomaly scores the algorithm already computes for every row.

**Success Criteria**:
- By default, a row is flagged when its anomaly score exceeds `Q3 + iqr_multiplier·IQR` of that dataset's scores, the Tukey fence the report already applies to single columns.
- A number in `isolation_forest_contamination` still flags that fixed share; the setting now defaults to `null`.
- Scores with no spread flag nothing, and fewer than 30 complete rows are not scored.
- The cut used, or why the method did not run, reaches `summary.json` (`outliers.isolation_forest`), `tables/outlier_summary.csv` and the report.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (the `1.5` case of `test_invalid_values_raise_config_error` is a guard that passed before too; the rest failed before the code existed):
- `tests/test_modules.py::TestIsolationForest::test_cutoff_is_tukeys_upper_fence_over_the_scores`, `test_cutoff_without_spread_is_none`, `test_flags_the_isolated_rows_and_not_a_fixed_share`, `test_explicit_contamination_keeps_the_fixed_share` and `test_too_few_rows_are_not_scored`.
- `tests/test_modules.py::TestConfig::test_isolation_forest_contamination_is_optional`, plus the `1.5` guard.
- `tests/test_integration.py::TestOutputCompleteness::test_isolation_forest_cut_is_explained_in_every_output` (failed on the old code with `KeyError: 'isolation_forest'`).

**Status**: COMPLETE. PR #20 was merged on 2026-09-13 as `e18ee73`, and its branch was deleted.

Verified with real output:
- **200 tests passed** on the branch, 0 failed, 0 warnings (192 + 8 new). Ruff clean. `main` after the merge: **200 passed**, ruff clean.
- Three cuts were compared before the code was written. scikit-learn's own `"auto"` cut (score > 0.5) flagged 17.2% of clean normal data and 33.8% of `Order_Details`, worse than the fixed share, and was dropped. The Tukey fence flagged 1.6% of clean data, and all 100 anomalies planted in a synthetic set plus 58 normal rows.
- 300 trees, from timing and 5-seed stability runs on every dataset: with 100 trees the seed alone moved `Dates` between 0 and 31 flagged rows and `siniestros` between 37 and 63; with 300 they held at 0 and at 50–59; 500 steadied nothing further and took 9.5–11.5 s against 4.6 s on the 219 432 rows of `Order_Details`. Parallel scoring did not help at 300 trees.
- Differential run over the 17 datasets, `main` against the branch, Isolation Forest share (and rows with any outlier): `Order_Details` 10.0% → 0.6% (18.3% → 14.2%), `stroke` 9.6% → 1.5% (15.5% → 14.0%), `ecommerce` 9.0% → 1.5% (12.7% → 6.7%), `Dates` 8.5% → 0% (8.5% → 0%), but `Inventory` 10.0% → 12.1% (12.9% → 14.4%) and `siniestros` 10.2% → 12.8%. `Sales_Outlet` (5 rows) is no longer scored. The outlier step takes about twice as long (`Order_Details` 3.9 s → 6.0 s).
- The Vistara batch regenerated on `main` (`reports/Vistara_batch_20260913_121354`) matches the differential run: `Order_Details` 40 184 → 31 122 rows with an outlier (14.2%), of which Isolation Forest flags 1 272 above a score of 0.707; `Dates` 31 → 0; `Inventory` 7 570 → 8 461; `Sales_Outlet` reads «No aplicado: 5 filas completas». All 8 files match their CSV row counts, none has a failed step, and the alerts are unchanged.
- Known limit: the fence is relative. On a dataset with no anomaly it still flags the most isolated 1–2% of rows, so the report reads as "most isolated rows", not as errors.

## Stage 18: Pie charts in percentages for categorical columns
**Goal**: The categorical distributions were bar charts of raw counts only. The user asked for pie charts explained in percentages, except for columns with too many categories to read as a pie, and left that criterion to Claude.

**Success Criteria**:
- Each categorical column with 2 to 6 categories gets a pie in a «Proporciones Categóricas» block right below «Distribuciones Categóricas»; columns with more keep their bars only.
- Shares are computed over the rows that have a value, most frequent first.
- Only slices of at least 5% carry their share inside the pie; the legend names every category with its share and row count, and a share that would round to zero reads `<0.1 %`.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all failed before the code existed):
- `tests/test_modules.py::TestCategoricalPieChart`: 8 tests over the shares (sum and order, missing values left out), the category limit (6 still gets a pie; 7, or a single category, gets none), the inside label of a thin slice, `<0.1 %`, and that the file is or is not written.
- `tests/test_integration.py::TestOutputCompleteness::test_categorical_columns_with_few_categories_get_a_pie`.

**Status**: COMPLETE. PR #22 was merged on 2026-09-13 as `7001ca0`, and its branch was deleted.

Verified with real output:
- **209 tests passed** on the branch, 0 failed, 0 warnings (200 + 9 new). Ruff clean. `main` after the merge: **209 passed**, ruff clean.
- The limit was measured before the code: over the 19 datasets in `data/raw`, 66 of the 101 categorical columns have 5 categories or fewer, none has 6, and the next ones hold 7 to 10 slices of similar size (`año`, ten slices of about 10%; `estado_civil`, nine, one under 0.1%). A cut at 5 or 6 selects the same columns; 6 is also where a pie stops reading as part-to-whole.
- Rendered and checked by eye on `discount_pct`, `quantity`, `product_category`, and stroke's `gender` (`Other: <0.1 % (1)`), `work_type` and `ever_married`. The palette validator of the dataviz guidance needs Node, which is not installed; the report's palette was kept, and no slice depends on colour alone.
- On the new penguin datasets (`reports/penguins_batch_20260913_181715`), `penguins_size` gets 3 pies (species, island, sex) and `penguins_lter` 5 (studyName, Species, Island, Sex, Clutch Completion); `Comments` (7 categories) and `Date Egg` (50) keep bars only. The species shares match the CSV: 44.2% / 36.0% / 19.8%.
- The Vistara batch regenerated on `main` (`reports/Vistara_batch_20260913_181954`) holds 15 pies across its 8 files, among them `Products.product_category` and `Order_Details` `discount_pct` and `quantity`; `product_id` (40), `month_name` (12), `product_type` (19) and `color` (14) keep bars only. Outlier counts and alerts are identical to the previous batch, and no file has a failed step.

## Stage 19: Scatter plots of the most correlated pairs
**Goal**: The section is titled «Scatter Plots (Pares con Mayor Correlación)», but the code drew the first `max_scatter_pairs` numeric pairs in column order and never looked at the correlation. Found reviewing the penguin datasets.

**Success Criteria**:
- Pairs are ranked by |r| from the Pearson matrix the relationships step computes (recomputed if that step failed), strongest first, sign kept, undefined correlations left out.
- Each plot shows its r in the title.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all failed before the change):
- `tests/test_modules.py::TestScatterPairs`: `test_pairs_come_strongest_first_whatever_their_sign`, `test_limit_is_respected_and_undefined_correlations_are_left_out`, `test_no_matrix_no_pairs`.
- `tests/test_integration.py::TestOutputCompleteness::test_scatter_plots_show_the_most_correlated_pairs` (drew `scatter_a_vs_b.png`, the first pair, on the old code).

**Status**: COMPLETE. PR #24 was merged on 2026-09-14 as `c2e272e`, and its branch was deleted.

Verified with real output:
- **213 tests passed** on the branch (209 + 4 new). Ruff clean.
- Measured before the code: every dataset with more than 10 numeric pairs was affected, 7 of the 19 in `data/raw`. The strongest pair was not drawn on `credito_asegurado` (|r| = 1.00, 8 of 10 plots from the weaker half of the ranking), `penguins_lter` (|r| = 0.87, 7 of 10) and `siniestros` (|r| = 0.98).
- On `penguins_lter` the 10 plots are now exactly the top 10 of the ranking, including `Flipper Length (mm) vs Body Mass (g) (r = 0.87)`.

## Stage 20: Row counters are identifiers
**Goal**: `Sample Number` in `penguins_lter` (1 to 152) numbers the samples, but was analysed as a continuous measure: histogram, boxplot, VIF entry and 6 of the 10 scatter plots. The identifier-by-name rule knows `id` and `key`; widening it to "number" would catch real counts such as `numero_creditos`.

**Success Criteria**:
- An integer column with more distinct values than the discrete threshold whose row-to-row step is exactly +1 on at least 90% of its rows is an identifier.
- A count covering every value from 1 up in no particular order, and a sorted measure with repeats, stay numeric.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (the first two failed before the change; the two guards were green before and after):
- `tests/test_modules.py::TestTypeInference::test_sequence_number_is_an_identifier`.
- `tests/test_integration.py::TestOutputCompleteness::test_sequence_number_columns_are_not_analysed_as_variables`.
- Guards: `test_shuffled_count_covering_every_value_stays_numeric`, `test_sorted_measure_with_repeats_stays_numeric`.
- Three existing tests built their data as counters (`range()`, `20 + i % 50`) and failed under the new rule: `test_amount_with_almost_unique_values_stays_numeric`, `test_name_merely_ending_in_id_is_not_an_identifier` and `test_numeric_id_columns_are_not_analysed_as_variables`. Their data is now shuffled, random or stepped by 7, and each still checks what it was written for.

**Status**: COMPLETE. PR #25 was merged on 2026-09-14 as `467e3ef`, and its branch was deleted.

Verified with real output:
- **213 tests passed** on the branch (209 + 4 new). Ruff clean.
- Measured before the code, over every integer column with more than 20 values in the 19 datasets: `Sample Number` steps by +1 on 99.4% of its rows; no measure exceeds 18.3% (`numero_mensualidades_no_pagadas`). Coverage of 1..max is no signal: `units_sold` and `days_to_close` reach 100% with 0.9% of +1 steps.
- Differential run over the 19 datasets: only `Sample Number` changes type, and the alert count stays at 21.

## Stage 21: Two-digit-year dates, and dates charted as dates
**Goal**: `Date Egg` in `penguins_lter` (`11/11/07`) was read as a category with 50 values, because the date formats only accepted four-digit years. Two things downstream did not treat dates as dates either: the date statistics parsed text with a bare `pd.to_datetime` (reading `05/03/07` as May 3rd, with a `UserWarning`), and the time series plotted the row number against the date value, one axis label per distinct day (366 on `Sales_Receipts`), for every date column.

**Success Criteria**:
- `%d/%m/%y`, `%d-%m-%y`, `%m/%d/%y` and `%m-%d-%y` are accepted, after every four-digit format and day-first before month-first.
- Date statistics read text with the same explicit formats (`coerce_to_datetime`, now public).
- The time series counts rows per day, week, month or year — the finest period that keeps it under 120 points — with empty periods at zero.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (`test_the_time_series_plot_is_written_for_text_dates` is a contract, green before and after; the other 8 failed before the change):
- `tests/test_modules.py::TestTypeInference::test_dates_with_a_two_digit_year_are_dates`, `test_two_digit_year_dates_parse_to_the_right_day`, `test_ambiguous_two_digit_year_dates_read_day_first`.
- `tests/test_modules.py::TestDatesOverTime`: stats read two-digit years without warnings; ambiguous dates day-first; rows per day over a few weeks; per week or month over longer spans; the plot is written for text dates.
- `tests/test_integration.py::TestOutputCompleteness::test_two_digit_year_dates_are_analysed_as_dates`.

**Status**: COMPLETE. PR #26 was merged on 2026-09-14 as `d6506ee`, and its branch was deleted.

Verified with real output:
- **218 tests passed** on the branch (209 + 9 new). Ruff clean.
- Type differential over the 19 datasets: only `Date Egg` changes type.
- `Date Egg` spans 2007-11-09 to 2009-12-01 and its weekly chart shows the three November laying seasons; `Sales_Receipts.transaction_date` is weekly, about 3 900 sales a week with peaks in January, May and November–December; `Customers.customer_since` is monthly over five years. No `UserWarning` reading any of them.

## Stage 22: Punctuation-only cells are missing values
**Goal**: `sex` in both penguin datasets holds one `.` among `MALE` and `FEMALE`. pandas leaves such a cell as text, so the report counted a third sex: a bar, a 0.3% slice of the pie, and an understated missing share.

**Success Criteria**:
- Before any analysis, a cell holding only `.`, `-`, `?`, `_`, `*` or `/` becomes a missing value; text that merely contains punctuation is kept.
- A text column whose remaining values are all numbers becomes numeric.
- Each affected column gets a low-severity alert naming the markers and counts, and `summary.json` lists them under `data_quality.missing_placeholders`.
- `ruff check .` and `ruff format --check .` are clean.

**Tests** (all failed before the change):
- `tests/test_modules.py::TestMissingValuePlaceholders`: 4 tests (replaced and counted; text with punctuation kept; numbers recovered; nothing to replace leaves the frame as it was).
- `tests/test_integration.py::TestOutputCompleteness::test_missing_value_placeholders_are_reported_and_not_analysed`.

**Status**: COMPLETE. PR #27 was merged on 2026-09-14 as `818a6f8`, and its branch was deleted.

Verified with real output:
- **214 tests passed** on the branch (209 + 5 new). Ruff clean.
- Across the 19 datasets only the two penguin `sex` columns hold such cells, one `.` each.
- `penguins_size`: `sex` has 2 categories, 3.20% missing (11 of 344, matching the CSV), a pie of MALE 50.5% / FEMALE 49.5% (n = 333), and one alert.

## Stages 19 to 22 together
- The four branches were checked pairwise and merge cleanly in any order. The tree with all four merged, before any was on `main`, passed **231 tests**; ruff clean. `main` after the four merges: **231 passed**, ruff clean.
- The penguin batch regenerated on `main` (`reports/penguins_batch_20260913_193213`) confirms all four findings resolved: on `penguins_lter` the 10 scatter plots are the top 10 pairs by |r|, `Date Egg` is a date with a time series, `Sample Number` is an identifier with no plot and no VIF entry, and `Sex` has 2 categories with 3.20% missing and an alert naming the `.`; `penguins_size` shows the same for its 6 pairs and `sex`. Shapes match the CSVs and no step failed.
- The Vistara batch (`reports/Vistara_batch_20260913_193243`) keeps every type, alert and outlier count of the previous one: none of its files has more than 5 numeric columns, so every pair was already drawn, and it holds no two-digit years, counters or placeholders. Its date columns now get the per-period time series. The mortgage batch (`reports/mx_batch_20260913_193455`), the one most affected by stage 19, now draws the strongest pair on each file, including `monto_credito` vs `saldo_principal` (|r| = 1.00) on `credito_asegurado`; all four files match their CSV row counts with no failed step.

## Stage 23: A pair plot of the numeric variables, coloured by group
**Goal**: The user asked for a pair plot section below «Scatter Plots (Pares con Mayor Correlación)», to explore several numeric relationships at once and see whether groups differ.

**Success Criteria**:
- A «Pair Plot: Relaciones entre Variables Numéricas» block follows the scatter plots whenever a dataset has at least 3 continuous numeric columns; with more than 6, the columns in the strongest correlations are kept, in dataset order.
- It is coloured by a classification target when there is one; otherwise by the categorical column of 2 to 6 groups with the highest mean correlation ratio against the columns, only if that reaches 0.25. Every group needs at least 10 rows in what is drawn.
- On the diagonal each group's KDE is normalised on its own; above 2 000 complete rows a fixed random sample is drawn.
- The report and `summary.json → pair_plot` state the columns, the group and why, or why the plot is uncoloured.
- `ruff check .` and `ruff format --check .` are clean, with no warnings.

**Tests** (all failed before the code existed):
- `tests/test_modules.py::TestPairPlot`: 8 tests (the best-separating group colours the plot and one that separates nothing does not; a classification target wins even when weak; groups too many or too small are skipped; fewer than 3 continuous columns give no plot; the most related columns are kept above 6; large tables are sampled; the file is written).
- `tests/test_integration.py::TestOutputCompleteness::test_pair_plot_follows_the_scatter_plots_and_names_its_group` and `test_no_pair_plot_with_fewer_than_three_continuous_columns`.

**Status**: COMPLETE. PR #29 was merged on 2026-09-14 as `a7415a5`, and its branch was deleted.

Verified with real output:
- **241 tests passed** on the branch, 0 failed, and no warning under `-W error::UserWarning` (231 + 10 new). Ruff clean. `main` after the merge: **241 passed**, ruff clean.
- Measured before the code, over the 19 datasets in `data/raw`: drawing cost follows the columns, not the rows (3 columns: 2.4 s for 1 000 rows, 3.4 s for 5 000; 6 columns on `penguins_lter`: 7.7 s), which set the 6-column cap and the 2 000-row sample. 13 of the 23 files have at least 3 continuous columns; no Vistara file does.
- The mean η of the best grouping per file set the 0.25 threshold (Cohen's medium effect): both penguin files are coloured by species (0.84, 0.81) while `healthcare` (0.14), `ecommerce` (0.11), `sales` (0.07) and `credito_asegurado` (0.04) stay uncoloured.
- Reviewing the rendered stroke plot showed the shared diagonal scale flattening the 5% of stroke cases; normalising each group on its own shows them concentrated at older ages and high glucose.
- Reports regenerated on `main`, all without a failed step and with the block right after the scatter plots: `reports/penguins_batch_20260913_202350` (`penguins_lter` 6 columns and `penguins_size` 4, coloured by species), `reports/mx_batch_20260913_202434` (`cobranza` 3 columns, `siniestros` 5, `credito_asegurado` 6 sampled to 2 000 of 76 526 rows, all uncoloured; `clientes` has a single continuous column and no pair plot) and `reports/healthcare-dataset-stroke-data_20260913_202558` (with `--target stroke`, coloured by the target, 2 000 of 4 909 complete rows).
- Known limit: heavily skewed money columns (`credito_asegurado`) crowd almost every point near zero, as in the scatter plots; a log scale is not part of this stage.
