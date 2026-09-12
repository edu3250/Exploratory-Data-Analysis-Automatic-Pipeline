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
