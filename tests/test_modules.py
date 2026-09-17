"""
Unit tests for EDA Pipeline modules.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda_pipeline.config import (
    ColumnTypeConfig,
    Config,
    ConfigError,
    create_default_config_file,
    load_config_file,
    merge_configs,
)
from eda_pipeline.data_loader import (
    detect_decimal_separator,
    detect_delimiter,
    detect_encoding,
    detect_file_format,
    discover_batch_files,
    list_excel_sheets,
    load_batch,
    load_data,
)
from eda_pipeline.data_quality import analyze_data_quality, analyze_duplicates
from eda_pipeline.html_report import column_quality_rows, overall_missing_pct, preview_rows
from eda_pipeline.outlier_detection import (
    ISOLATION_FOREST_MIN_ROWS,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_mad,
    isolation_forest_cutoff,
)
from eda_pipeline.relationships import association_matrix, correlation_ratio, cramers_v, pearson_correlation
from eda_pipeline.target_analysis import analyze_class_balance
from eda_pipeline.type_inference import (
    apply_column_type_overrides,
    infer_all_types,
    infer_semantic_type,
)
from eda_pipeline.univariate_analysis import (
    analyze_categorical,
    analyze_numeric,
    analyze_time_of_day,
    analyze_univariate,
)
from eda_pipeline.visualizations import (
    PAIR_PLOT_MAX_COLUMNS,
    PAIR_PLOT_MIN_GROUP_ROWS,
    PAIR_PLOT_MIN_HUE_ETA,
    PIE_MAX_CATEGORIES,
    TARGET_BAR_MAX_CLASSES,
    choose_pair_plot,
    format_share,
    pie_chart_shares,
    pie_slice_label,
    plot_pair_plot,
    plot_pie_chart,
    plot_target_vs_categorical,
    plot_time_of_day,
    target_vs_categorical_counts,
    top_correlated_pairs,
)


class TestConfig:
    """Test configuration system."""

    def test_default_config(self):
        config = Config()
        assert config.output_dir == "reports"
        assert config.strict_mode is False
        assert config.verbose is False
        assert config.decimal is None  # None means auto-detect

    def test_language_key_deprecated_but_accepted(self):
        """'language' is no longer a real setting, but old YAML files must not crash (with a warning)."""
        with pytest.warns(DeprecationWarning):
            config = Config.from_dict({"language": "es", "output_dir": "reports"})
        assert not hasattr(config, "language")
        assert config.output_dir == "reports"

    def test_config_from_dict(self):
        data = {"input_file": "test.csv", "target": {"target_column": "target_col"}, "output_dir": "my_reports"}
        config = Config.from_dict(data)
        assert config.input_file == "test.csv"
        assert config.target.target_column == "target_col"
        assert config.output_dir == "my_reports"

    def test_create_config_file(self, tmp_output_dir):
        config_path = tmp_output_dir / "config.yaml"
        create_default_config_file(config_path)
        assert config_path.exists()
        loaded = load_config_file(config_path)
        assert "column_types" in loaded
        assert "data_quality" in loaded

    def test_default_config_file_round_trips(self, tmp_output_dir):
        """The generated default YAML must itself pass Config.from_dict validation."""
        config_path = tmp_output_dir / "config.yaml"
        create_default_config_file(config_path)
        loaded = load_config_file(config_path)
        config = Config.from_dict(loaded)
        assert config.data_quality.missing_threshold == 0.95
        assert config.target.class_imbalance_threshold == 0.8

    def test_unknown_top_level_key_raises_config_error(self):
        with pytest.raises(ConfigError, match="no_existe"):
            Config.from_dict({"no_existe": 1})

    def test_unknown_nested_key_raises_config_error(self):
        with pytest.raises(ConfigError, match="no_existe"):
            Config.from_dict({"data_quality": {"no_existe": 1}})

    @pytest.mark.parametrize(
        "overrides",
        [
            {"data_quality": {"missing_threshold": 1.5}},
            {"data_quality": {"missing_threshold": 0}},
            {"outliers": {"iqr_multiplier": -1}},
            {"outliers": {"isolation_forest_contamination": 1.5}},
            {"visualizations": {"max_histograms": 0}},
            {"target": {"class_imbalance_threshold": 1.5}},
            {"target": {"target_type": "bogus"}},
            {"decimal": ";"},
            {"sample_size": -5},
        ],
    )
    def test_invalid_values_raise_config_error(self, overrides):
        with pytest.raises(ConfigError):
            Config.from_dict(overrides)

    def test_merge_configs_precedence_defaults_lt_file_lt_cli(self):
        """CLI overrides win over YAML, which wins over defaults; None CLI values never clobber YAML."""
        base = Config()
        file_config = {"output_dir": "from_yaml", "strict_mode": True}
        cli_overrides = {"output_dir": None, "strict_mode": None}
        merged = merge_configs(base, file_config, cli_overrides)
        assert merged.output_dir == "from_yaml"
        assert merged.strict_mode is True

    def test_merge_configs_deep_merges_nested_target_section(self):
        """A CLI --target must not wipe out a YAML-provided class_imbalance_threshold."""
        base = Config()
        file_config = {"target": {"class_imbalance_threshold": 0.65}}
        cli_overrides = {"target": {"target_column": "y"}}
        merged = merge_configs(base, file_config, cli_overrides)
        assert merged.target.target_column == "y"
        assert merged.target.class_imbalance_threshold == 0.65

    def test_isolation_forest_contamination_is_optional(self):
        # No fixed share by default: the cut comes from each dataset's own anomaly scores.
        assert Config().outliers.isolation_forest_contamination is None
        config = Config.from_dict({"outliers": {"isolation_forest_contamination": 0.05}})
        assert config.outliers.isolation_forest_contamination == 0.05

    def test_duplicate_threshold_and_batch_pattern_are_configurable(self):
        config = Config.from_dict({"data_quality": {"duplicate_threshold": 0.2}, "batch_pattern": "*.csv"})
        assert config.data_quality.duplicate_threshold == 0.2
        assert config.batch_pattern == "*.csv"


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_csv(self, csv_file):
        df = load_data(csv_file)
        assert df is not None
        assert len(df) > 0

    def test_load_with_encoding(self, latin1_csv_file):
        df = load_data(latin1_csv_file, encoding="latin-1", delimiter=";", decimal=",")
        assert df is not None
        assert "región" in df.columns

    def test_sample_size_log_reports_original_length_not_sampled_length(self, csv_file, caplog):
        """Bug #12: the log used to report len(df) AFTER sampling, i.e. the sample size itself."""
        original_len = len(pd.read_csv(csv_file))
        sample_size = 5
        assert sample_size < original_len

        with caplog.at_level("INFO", logger="eda_pipeline.data_loader"):
            df = load_data(csv_file, sample_size=sample_size)

        assert len(df) == sample_size
        sample_logs = [r.message for r in caplog.records if "Sampled" in r.message]
        assert len(sample_logs) == 1
        assert f"Sampled {sample_size} rows from {original_len}" in sample_logs[0]

    def test_detect_delimiter(self, latin1_csv_file):
        delim = detect_delimiter(latin1_csv_file, "latin-1")
        assert delim == ";"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent.csv"))

    # --- Bug #1: Excel without an explicit sheet must not crash -----------------------------

    def test_load_excel_defaults_to_first_sheet(self, multi_sheet_excel_file):
        """pd.read_excel(sheet_name=None) returns a dict of all sheets; we must default to the first."""
        df = load_data(multi_sheet_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a"]
        assert df["a"].tolist() == [1, 2, 3]

    def test_load_excel_with_explicit_sheet_name(self, multi_sheet_excel_file):
        df = load_data(multi_sheet_excel_file, excel_sheet="Segunda")
        assert list(df.columns) == ["b"]

    def test_load_excel_with_explicit_sheet_index(self, multi_sheet_excel_file):
        df = load_data(multi_sheet_excel_file, excel_sheet=1)
        assert list(df.columns) == ["b"]

    # --- Bug #2: JSON arrays / nested cells must not crash -----------------------------------

    def test_load_json_array_flattens_nested_records(self, json_array_file):
        df = load_data(json_array_file)
        assert "address.city" in df.columns
        assert "address.zip" in df.columns
        assert df.loc[0, "address.city"] == "X"

    def test_load_json_array_stringifies_list_cells(self, json_array_file):
        """A nested list cell must become a hashable JSON string, not crash value_counts/duplicated."""
        df = load_data(json_array_file)
        assert isinstance(df.loc[0, "tags"], str)
        assert json.loads(df.loc[0, "tags"]) == ["a", "b"]
        # These would raise TypeError: unhashable type: 'dict'/'list' before the fix.
        df["tags"].value_counts()
        df.duplicated().sum()

    def test_load_jsonl_file(self, jsonl_file):
        df = load_data(jsonl_file)
        assert len(df) == 5
        assert list(df.columns) == ["id", "value"]

    # --- Bug #8: encoding/delimiter detection robustness -------------------------------------

    def test_detect_encoding_handles_late_accent(self, late_accent_cp1252_file):
        """A short 10 KB sample would see only ASCII and wrongly guess 'ascii'."""
        encoding = detect_encoding(late_accent_cp1252_file)
        raw = late_accent_cp1252_file.read_bytes()
        text = raw.decode(encoding)  # must not raise
        assert "José Muñoz Peña" in text

    def test_detect_delimiter_ignores_spaces_in_free_text(self, freetext_spaces_csv_file):
        assert detect_delimiter(freetext_spaces_csv_file, "utf-8") == ","

    def test_detect_delimiter_prefers_semicolon_over_decimal_commas(self, semicolon_decimal_comma_csv_file):
        assert detect_delimiter(semicolon_decimal_comma_csv_file, "utf-8") == ";"

    # --- Bug #7: decimal separator auto-detection --------------------------------------------

    def test_detect_decimal_separator_comma_with_semicolon_delimiter(self, semicolon_decimal_comma_csv_file):
        assert detect_decimal_separator(semicolon_decimal_comma_csv_file, "utf-8", ";") == ","

    def test_detect_decimal_separator_never_comma_when_delimiter_is_comma(self):
        # Ambiguous by construction: comma can't be both delimiter and decimal separator.
        assert detect_decimal_separator(Path("unused.csv"), "utf-8", ",") == "."

    def test_load_data_latin1_matches_ecommerce_column_types(self, tmp_output_dir):
        """Acceptance test for bug #7: a cp1252 ';' file with decimal commas loads with the same
        column types as its comma-separated UTF-8 twin.

        Both files are generated here: the test used to read data/raw/data_latin1.csv, which is
        gitignored, so it failed on any fresh clone of the repository.
        """
        df = pd.DataFrame(
            {
                "region": ["Norte", "Sur", "Centro", "Oeste"] * 5,
                "product_category": ["Electrónica", "Ropa", "Hogar", "Jardín"] * 5,
                "amount_spent": [24.598, 111.139, 5.5, 300.25] * 5,
                "is_churn": [0.0, 1.0, 0.0, 0.0] * 5,
            }
        )
        latin1_path = tmp_output_dir / "data_latin1.csv"
        utf8_path = tmp_output_dir / "ecommerce.csv"
        df.to_csv(latin1_path, index=False, sep=";", decimal=",", encoding="cp1252")
        df.to_csv(utf8_path, index=False)

        latin1 = load_data(latin1_path)
        assert latin1["amount_spent"].dtype.kind == "f" or str(latin1["amount_spent"].dtype) == "Float64"
        assert latin1["product_category"].iloc[0] == "Electrónica"
        assert infer_all_types(latin1) == infer_all_types(load_data(utf8_path))

    # --- Bug #3: batch file discovery ---------------------------------------------------------

    def test_discover_batch_files_skips_hidden_and_unsupported(self, tmp_output_dir):
        (tmp_output_dir / ".gitkeep").write_text("")
        (tmp_output_dir / "notes.txt.bak").write_text("not supported")
        (tmp_output_dir / "a.csv").write_text("x\n1\n")
        (tmp_output_dir / "b.csv").write_text("x\n2\n")

        files = discover_batch_files(tmp_output_dir)

        names = {f.name for f in files}
        assert names == {"a.csv", "b.csv"}

    def test_discover_batch_files_honors_pattern(self, tmp_output_dir):
        (tmp_output_dir / "a.csv").write_text("x\n1\n")
        (tmp_output_dir / "b.json").write_text("[]")

        files = discover_batch_files(tmp_output_dir, "*.csv")

        assert [f.name for f in files] == ["a.csv"]

    # --- Misc error paths / legacy helpers ----------------------------------------------------

    def test_list_excel_sheets(self, multi_sheet_excel_file):
        assert list_excel_sheets(multi_sheet_excel_file) == ["Primera", "Segunda"]

    def test_detect_file_format_rejects_unsupported_extension(self, tmp_output_dir):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_file_format(tmp_output_dir / "data.exe")

    def test_load_data_rejects_unknown_explicit_format(self, csv_file):
        with pytest.raises(ValueError, match="Unknown file format"):
            load_data(csv_file, file_format="bogus")

    def test_load_json_raises_on_totally_invalid_content(self, tmp_output_dir):
        bad_json = tmp_output_dir / "bad.json"
        bad_json.write_text("this is not json at all {{{", encoding="utf-8")
        with pytest.raises(Exception):
            load_data(bad_json)

    def test_load_excel_raises_on_corrupt_file(self, tmp_output_dir):
        fake_xlsx = tmp_output_dir / "fake.xlsx"
        fake_xlsx.write_bytes(b"not a real xlsx file")
        with pytest.raises(Exception):
            load_data(fake_xlsx)

    def test_detect_decimal_separator_missing_file_returns_dot(self):
        assert detect_decimal_separator(Path("does_not_exist.csv"), "utf-8", ";") == "."

    def test_load_batch_legacy_helper_loads_supported_files(self, tmp_output_dir, synthetic_dataset):
        synthetic_dataset.to_csv(tmp_output_dir / "a.csv", index=False)
        synthetic_dataset.head(5).to_csv(tmp_output_dir / "b.csv", index=False)

        results = load_batch(tmp_output_dir)

        assert set(results.keys()) == {"a", "b"}
        assert len(results["b"]) == 5

    def test_load_batch_legacy_helper_skips_failing_file(self, tmp_output_dir, synthetic_dataset):
        synthetic_dataset.to_csv(tmp_output_dir / "good.csv", index=False)
        (tmp_output_dir / "bad.parquet").write_bytes(b"not a real parquet file")

        results = load_batch(tmp_output_dir)

        assert "good" in results
        assert "bad" not in results


class TestMissingValuePlaceholders:
    """A lone "." in penguins sex was counted as a third sex instead of a missing value."""

    def test_punctuation_only_cells_become_missing(self):
        from eda_pipeline.data_loader import replace_missing_placeholders

        df = pd.DataFrame({"sex": pd.Series(["MALE", "FEMALE", ".", "MALE", " - ", None], dtype="string")})
        cleaned, found = replace_missing_placeholders(df)
        assert cleaned["sex"].isna().sum() == 3
        assert set(cleaned["sex"].dropna()) == {"MALE", "FEMALE"}
        assert found == {"sex": {".": 1, "-": 1}}

    def test_text_that_merely_contains_punctuation_is_kept(self):
        from eda_pipeline.data_loader import replace_missing_placeholders

        values = ["Dr.", "N/A-12", "-5", "a.b", "?"]
        df = pd.DataFrame({"nota": pd.Series(values, dtype="string"), "monto": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cleaned, found = replace_missing_placeholders(df)
        assert list(cleaned["nota"].dropna()) == ["Dr.", "N/A-12", "-5", "a.b"]
        assert found == {"nota": {"?": 1}}  # numbers and untouched columns are not reported

    def test_a_column_of_numbers_with_placeholders_becomes_numeric(self):
        from eda_pipeline.data_loader import replace_missing_placeholders

        # "?" in a column of weights makes pandas read every weight as text.
        df = pd.DataFrame({"peso": pd.Series(["3750", "?", "3800.5", "4100"], dtype="string")})
        cleaned, _ = replace_missing_placeholders(df)
        assert pd.api.types.is_numeric_dtype(cleaned["peso"])
        assert cleaned["peso"].sum() == pytest.approx(11650.5)

    def test_nothing_to_replace_leaves_the_frame_as_it_was(self):
        from eda_pipeline.data_loader import replace_missing_placeholders

        df = pd.DataFrame({"sex": pd.Series(["MALE", "FEMALE"], dtype="string"), "masa": [1, 2]})
        cleaned, found = replace_missing_placeholders(df)
        assert found == {}
        pd.testing.assert_frame_equal(cleaned, df)


class TestTypeInference:
    """Test semantic type inference."""

    def test_numeric_continuous(self):
        series = pd.Series(np.random.normal(0, 1, 100), name="numeric")
        dtype = infer_semantic_type(series)
        assert dtype == "numeric_continuous"

    def test_numeric_discrete(self):
        series = pd.Series(np.random.randint(1, 10, 100), name="discrete")
        dtype = infer_semantic_type(series)
        assert dtype == "numeric_discrete"

    def test_categorical(self):
        series = pd.Series(np.random.choice(["A", "B", "C"], 100), name="cat")
        dtype = infer_semantic_type(series)
        assert dtype == "categorical"

    def test_boolean(self):
        series = pd.Series(np.random.choice([True, False], 100), name="bool")
        dtype = infer_semantic_type(series)
        assert dtype == "boolean"

    def test_datetime(self):
        series = pd.to_datetime(pd.date_range("2020-01-01", periods=100))
        dtype = infer_semantic_type(series)
        assert dtype == "datetime"

    @staticmethod
    def _clock_times(n: int = 300) -> pd.Series:
        # n distinct times of day between 09:00 and 20:59, like transaction_time on Vistara.
        return pd.Series(
            [f"{9 + (i // 25) % 12:02d}:{i % 60:02d}:{(i * 7) % 60:02d}" for i in range(n)], name="hora", dtype="string"
        )

    def test_time_of_day_is_a_time(self):
        # transaction_time ("11:43:47") was read as a category with 31702 values.
        assert infer_semantic_type(self._clock_times()) == "time"

    def test_hours_and_minutes_are_a_time(self):
        series = pd.Series([f"{i % 24:02d}:{(i * 5) % 60:02d}" for i in range(200)], name="hora", dtype="string")
        assert infer_semantic_type(series) == "time"

    def test_values_shaped_like_times_that_no_clock_shows_stay_categorical(self):
        series = pd.Series(["25:30", "31:45", "48:00"] * 40, name="marcador", dtype="string")
        assert infer_semantic_type(series) == "categorical"

    def test_date_with_a_time_stays_datetime(self):
        series = pd.Series([f"2024-01-{1 + i % 28:02d} 11:43:47" for i in range(100)], name="momento", dtype="string")
        assert infer_semantic_type(series) == "datetime"

    @staticmethod
    def _egg_dates() -> pd.Series:
        # Like Date Egg in penguins_lter: month/day/two-digit year, several seasons.
        days = [f"11/{d}/{y:02d}" for y in (7, 8, 9) for d in range(9, 29)]
        return pd.Series(days * 3, name="Date Egg", dtype="string")

    def test_dates_with_a_two_digit_year_are_dates(self):
        assert infer_semantic_type(self._egg_dates()) == "datetime"

    def test_two_digit_year_dates_parse_to_the_right_day(self):
        from eda_pipeline.type_inference import coerce_to_datetime

        parsed = coerce_to_datetime(self._egg_dates())
        assert parsed.notna().all()
        assert parsed.min() == pd.Timestamp("2007-11-09")
        assert parsed.max() == pd.Timestamp("2009-11-28")

    def test_ambiguous_two_digit_year_dates_read_day_first(self):
        from eda_pipeline.type_inference import coerce_to_datetime

        series = pd.Series(["05/03/07", "06/03/07", "07/03/07"] * 10, name="fecha", dtype="string")
        assert coerce_to_datetime(series).iloc[0] == pd.Timestamp("2007-03-05")

    def test_text(self):
        series = pd.Series([f"Text {i} " * 10 for i in range(100)], name="text")
        dtype = infer_semantic_type(series)
        assert dtype == "text"

    def test_long_repeated_labels_are_categorical(self):
        # 8 labels longer than 50 characters over 400 rows, like "evento_origen_siniestro":
        # the old rule called any column with long labels free text.
        labels = [f"Evento de origen del siniestro número {i}, con una descripción larga" for i in range(8)]
        series = pd.Series(labels * 50, name="evento")
        assert infer_semantic_type(series) == "categorical"

    def test_many_repeated_categories_are_categorical(self):
        # 28 values over 452 rows, like "causa_incumplimiento": above the old fixed cut-off of
        # 20 distinct values, but every value repeats, so it is a category.
        causes = [f"Causa de incumplimiento {i}" for i in range(28)]
        series = pd.Series((causes * 17)[:452], name="causa")
        assert infer_semantic_type(series) == "categorical"

    def test_rarely_repeated_strings_stay_free_text(self):
        comments = [f"Comentario del ejecutivo sobre el cliente {i}" for i in range(300)]
        series = pd.Series(comments + comments[:30], name="comentario")
        assert infer_semantic_type(series) == "text"

    def test_constant(self):
        series = pd.Series(["X"] * 100, name="const")
        dtype = infer_semantic_type(series)
        assert dtype == "constant"

    def test_identifier(self):
        series = pd.Series([f"ID{i:06d}" for i in range(100)], name="id")
        dtype = infer_semantic_type(series)
        assert dtype == "identifier"

    def test_repeated_codes_are_identifiers(self):
        # A foreign key repeats, so the "mostly unique" rule misses it: 1000 customer codes over
        # 3000 rows were read as a category and then flagged as high cardinality (Vistara).
        codes = [f"CUST-{i:05d}" for i in range(1000)]
        assert infer_semantic_type(pd.Series(codes * 3, name="customer_id")) == "identifier"

    def test_few_repeated_codes_stay_categorical(self):
        # 40 product codes over 2000 rows are a useful grouping, not an identifier, and they
        # never reach the high-cardinality threshold either.
        codes = [f"PROD-{i:03d}" for i in range(40)]
        assert infer_semantic_type(pd.Series(codes * 50, name="product_id")) == "categorical"

    def test_identifier_floor_follows_the_cardinality_threshold(self):
        # The floor is the high-cardinality threshold, so a column of codes is either analysed
        # as a category or recognised as an ID, but never flagged as high cardinality.
        codes = [f"C-{i:04d}" for i in range(60)]
        series = pd.Series(codes * 10, name="cliente")
        assert infer_semantic_type(series) == "categorical"
        assert infer_semantic_type(series, identifier_min_unique=50) == "identifier"

    def test_plain_labels_are_not_identifiers(self):
        # Only code-shaped values (letters and digits, no spaces) become identifiers; plain
        # labels keep their type, and their alert.
        labels = [f"Municipio {i}" for i in range(150)]
        assert infer_semantic_type(pd.Series(labels * 3, name="municipio")) == "categorical"

    def test_numeric_primary_key_is_an_identifier(self):
        # On the stroke dataset `id` held 5110 distinct values and drew a histogram, a boxplot,
        # a VIF entry and 3 of the 6 scatter plots.
        assert infer_semantic_type(pd.Series(range(1, 501), name="id")) == "identifier"

    def test_numeric_foreign_key_is_an_identifier(self):
        series = pd.Series([100 + (i % 60) for i in range(600)], name="customer_id")
        assert infer_semantic_type(series) == "identifier"

    def test_amount_with_almost_unique_values_stays_numeric(self):
        # mx: monto_siniestro holds 451 distinct integers over 452 rows. Uniqueness alone would
        # call it a key, but it is money, so the name is what decides.
        # Shuffled: amounts that climb by exactly one per row would be a row counter, not amounts.
        amounts = np.random.default_rng(0).permutation(np.arange(10_000, 10_452))
        series = pd.Series(amounts, name="monto_siniestro")
        assert infer_semantic_type(series) == "numeric_continuous"

    def test_small_numeric_code_stays_discrete(self):
        # month_id has 12 values: still useful as a grouping, so it keeps its categorical analysis.
        series = pd.Series([(i % 12) + 1 for i in range(366)], name="month_id")
        assert infer_semantic_type(series) == "numeric_discrete"

    def test_float_column_named_id_stays_numeric(self):
        series = pd.Series([float(i) + 0.5 for i in range(100)], name="id")
        assert infer_semantic_type(series) == "numeric_continuous"

    def test_sequence_number_is_an_identifier(self):
        # Sample Number in penguins_lter: 1 to 152, restarting per study, 99.4% of steps are +1.
        values = list(range(1, 153)) + list(range(1, 125)) + list(range(1, 69))
        series = pd.Series(values, name="Sample Number", dtype="Int64")
        assert infer_semantic_type(series) == "identifier"

    def test_shuffled_count_covering_every_value_stays_numeric(self):
        # units_sold in sales.csv holds every value from 1 to 99, but in no particular order.
        rng = np.random.default_rng(0)
        series = pd.Series(rng.permutation(np.tile(np.arange(1, 100), 10)), name="units_sold", dtype="Int64")
        assert infer_semantic_type(series) == "numeric_continuous"

    def test_sorted_measure_with_repeats_stays_numeric(self):
        series = pd.Series(sorted(np.random.default_rng(1).integers(18, 90, 500)), name="edad", dtype="Int64")
        assert infer_semantic_type(series) == "numeric_continuous"

    def test_name_merely_ending_in_id_is_not_an_identifier(self):
        # "humid" ends in those two letters without being a key.
        readings = np.random.default_rng(0).integers(20, 100, 500)  # not 1, 2, 3...: that is a counter
        assert infer_semantic_type(pd.Series(readings, name="humid")) == "numeric_continuous"

    # --- Bug #6: date inference must be precise and warning-free -----------------------------

    def test_string_dates_iso_format_detected_without_warning(self, recwarn):
        series = pd.Series(["2023-03-24", "2023-05-19", "2024-01-31"] * 10, dtype="str")
        dtype = infer_semantic_type(series)
        assert dtype == "datetime"
        assert len(recwarn) == 0

    def test_string_dates_day_first_detected(self, recwarn):
        series = pd.Series(["24/03/2023", "19/05/2023", "31/01/2024"] * 10, dtype="str")
        dtype = infer_semantic_type(series)
        assert dtype == "datetime"
        assert len(recwarn) == 0

    def test_string_datetime_iso_with_millis_detected(self, recwarn):
        series = pd.Series(["2023-01-01T00:00:00.000", "2023-01-01T01:00:00.000"] * 10, dtype="str")
        assert infer_semantic_type(series) == "datetime"
        assert len(recwarn) == 0

    def test_comma_decimal_numeric_strings_not_misclassified_as_datetime(self, recwarn):
        """Regression for the reported bug: amount_spent-like values must not become 'datetime'."""
        series = pd.Series(["24,598029558215444", "111,13951165532036", "151,62979648890638"] * 10, dtype="str")
        dtype = infer_semantic_type(series)
        assert dtype != "datetime"
        assert len(recwarn) == 0

    # --- Bug #13: column_types overrides -----------------------------------------------------

    def test_apply_column_type_overrides_coerces_numeric_and_reports_failures(self):
        df = pd.DataFrame({"a": ["1", "2", "not_a_number"]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(numeric=["a"])

        new_df, result = apply_column_type_overrides(df, types, overrides)

        assert result.column_types["a"] == "numeric_continuous"
        assert pd.api.types.is_numeric_dtype(new_df["a"])
        assert result.coercion_failures["a"] == 1
        assert new_df["a"].isna().sum() == 1

    def test_apply_column_type_overrides_coerces_datetime(self):
        df = pd.DataFrame({"d": ["2020-01-01", "not_a_date"]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(datetime=["d"])

        new_df, result = apply_column_type_overrides(df, types, overrides)

        assert result.column_types["d"] == "datetime"
        assert pd.api.types.is_datetime64_any_dtype(new_df["d"])
        assert result.coercion_failures["d"] == 1

    def test_apply_column_type_overrides_datetime_prefers_day_first(self, recwarn):
        """
        Plain pd.to_datetime(errors="coerce") locks onto the first row's format and
        would wrongly turn '15/03/2021' into NaT after inferring month-first from
        '01/02/2020'. Only the genuinely invalid value should fail to convert.
        """
        df = pd.DataFrame({"d": ["01/02/2020", "15/03/2021", "28/04/2022", "not_a_date"]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(datetime=["d"])

        new_df, result = apply_column_type_overrides(df, types, overrides)

        assert result.coercion_failures["d"] == 1
        assert new_df["d"].iloc[0] == pd.Timestamp("2020-02-01")  # day-first: 01/02 -> Feb 1st
        assert new_df["d"].iloc[1] == pd.Timestamp("2021-03-15")
        assert len(recwarn) == 0

    def test_apply_column_type_overrides_ignore_drops_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "drop_me": ["x", "y", "z"]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(ignore=["drop_me"])

        new_df, result = apply_column_type_overrides(df, types, overrides)

        assert "drop_me" not in new_df.columns
        assert "drop_me" not in result.column_types
        assert result.ignored_columns == ["drop_me"]

    def test_apply_column_type_overrides_categorical_and_text_relabel(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(categorical=["a"], text=["b"])

        _, result = apply_column_type_overrides(df, types, overrides)

        assert result.column_types["a"] == "categorical"
        assert result.column_types["b"] == "text"

    def test_apply_column_type_overrides_reports_unknown_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        types = infer_all_types(df)
        overrides = ColumnTypeConfig(numeric=["does_not_exist"])

        _, result = apply_column_type_overrides(df, types, overrides)

        assert result.unknown_columns == ["does_not_exist"]


class TestDataQuality:
    """Test data quality analysis."""

    def test_missing_values(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        assert report.shape == synthetic_dataset.shape
        assert report.missing_per_column is not None
        assert "nullable_col" in report.missing_per_column

    def test_duplicates(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        # The fixture appends exactly 10 exact-duplicate rows (see conftest.py).
        assert report.duplicates == 10
        assert len(report.duplicate_rows) == 20  # 10 originals + their 10 copies

    def test_constant_columns(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        assert "constant" in report.constant_columns

    def test_quasi_constant(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        assert "quasi_constant" in report.quasi_constant_columns

    def test_high_cardinality(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        assert "high_cardinality" in report.high_cardinality_columns

    def test_high_cardinality_ignores_numeric_and_date_columns(self):
        # Amounts and dates naturally have thousands of distinct values; only categorical-like
        # columns deserve the alert (18 of the 20 alerts on the Mexican mortgage data were noise).
        n = 300
        df = pd.DataFrame(
            {
                "monto": np.linspace(1_000.0, 2_000_000.0, n),
                "fecha": pd.date_range("2020-01-01", periods=n, freq="D"),
                "municipio": [f"Municipio {i % 150}" for i in range(n)],
            }
        )
        report = analyze_data_quality(df, cardinality_threshold=100)
        assert set(report.high_cardinality_columns) == {"municipio"}
        assert [a.column for a in report.alerts if "alta cardinalidad" in a.message] == ["municipio"]

    def test_high_cardinality_ignores_identifier_columns(self):
        # An ID holds one value per entity by definition, so "too many distinct values" is not a
        # finding: 11 of the 14 alerts on the Vistara reports were this noise.
        n = 300
        df = pd.DataFrame(
            {
                "customer_id": [f"CUST-{i:05d}" for i in range(n)],
                "municipio": [f"Municipio {i % 150}" for i in range(n)],
            }
        )
        column_types = {"customer_id": "identifier", "municipio": "categorical"}
        report = analyze_data_quality(df, cardinality_threshold=100, column_types=column_types)
        assert set(report.high_cardinality_columns) == {"municipio"}
        assert [a.column for a in report.alerts if "alta cardinalidad" in a.message] == ["municipio"]

    def test_high_cardinality_ignores_time_of_day_columns(self):
        # A time of day takes thousands of values by nature, like an amount or a date.
        n = 300
        df = pd.DataFrame(
            {
                "hora": [f"{9 + (i // 25) % 12:02d}:{i % 60:02d}:00" for i in range(n)],
                "municipio": [f"Municipio {i % 150}" for i in range(n)],
            }
        )
        column_types = {"hora": "time", "municipio": "categorical"}
        report = analyze_data_quality(df, cardinality_threshold=100, column_types=column_types)
        assert set(report.high_cardinality_columns) == {"municipio"}

    def test_high_cardinality_uses_semantic_types_for_dates_stored_as_text(self):
        dates = pd.date_range("2020-01-01", periods=300, freq="D").strftime("%Y-%m-%d")
        df = pd.DataFrame({"fecha": dates})
        report = analyze_data_quality(df, cardinality_threshold=100, column_types={"fecha": "datetime"})
        assert report.high_cardinality_columns == {}

    def test_alerts_generated(self, synthetic_dataset):
        report = analyze_data_quality(synthetic_dataset)
        assert len(report.alerts) > 0

    # --- Bug #12: mixed-type duplicate sorting must not crash --------------------------------

    def test_duplicates_with_mixed_type_object_column_does_not_raise(self):
        """sort_values(by=...) used to raise TypeError comparing int/str in the same column."""
        df = pd.DataFrame(
            {
                "mixed": [1, "a", 1, "a", 2.5, None],
                "val": [1, 2, 1, 2, 3, 4],
            }
        )
        n_dup, dup_rows, alerts = analyze_duplicates(df)
        assert n_dup == 2
        assert len(dup_rows) == 4
        assert alerts[0].severity in ("low", "medium", "high")

    # --- Bug #14: duplicate_threshold must control alert severity ----------------------------

    @pytest.mark.parametrize(
        "threshold,expected_severity",
        [(0.1, "high"), (0.5, "medium")],
    )
    def test_duplicate_threshold_controls_severity(self, threshold, expected_severity):
        # 3 duplicate rows out of 10 -> 30% duplicate ratio.
        df = pd.DataFrame({"a": [1, 1, 1, 1, 2, 3, 4, 5, 6, 7]})
        _, _, alerts = analyze_duplicates(df, duplicate_threshold=threshold)
        assert alerts[0].severity == expected_severity

    # --- Numbers stored as text: what float() accepts is not what a number looks like ---------

    def test_a_column_of_numbers_written_as_text_is_flagged(self):
        from eda_pipeline.data_quality import detect_numeric_as_text

        df = pd.DataFrame({"rating": ["4.2", "4.0", "3.9", "5", ".5", "1e3", " 42 "]})
        flagged, alerts = detect_numeric_as_text(df)
        assert flagged == {"rating": 7}
        assert alerts[0].column == "rating"

    def test_an_id_with_an_underscore_is_not_a_number(self):
        """float('0001_01') is 101.0: Python reads '_' as a digit separator, PassengerId does not."""
        from eda_pipeline.data_quality import detect_numeric_as_text

        df = pd.DataFrame({"PassengerId": [f"{i:04d}_01" for i in range(1, 40)]})
        flagged, alerts = detect_numeric_as_text(df)
        assert flagged == {}
        assert alerts == []

    def test_the_words_float_accepts_are_not_numbers_either(self):
        from eda_pipeline.data_quality import detect_numeric_as_text

        df = pd.DataFrame({"estado": ["nan", "NaN", "inf", "-Infinity", "nan", "inf"]})
        assert detect_numeric_as_text(df)[0] == {}

    def test_a_mostly_numeric_column_is_still_flagged(self):
        from eda_pipeline.data_quality import detect_numeric_as_text

        df = pd.DataFrame({"precio": ["10.5"] * 9 + ["sin dato"]})  # 90%, above the 80% share
        assert detect_numeric_as_text(df)[0] == {"precio": 9}

    def test_labels_that_only_start_with_digits_are_not_flagged(self):
        from eda_pipeline.data_quality import detect_numeric_as_text

        df = pd.DataFrame({"codigo": ["64%", "1.2.3", "3 piezas", "2024-01-01", "1,5"]})
        assert detect_numeric_as_text(df)[0] == {}


class TestTargetAnalysis:
    """Test target variable analysis (bug #20: imbalance rule)."""

    def test_majority_share_above_threshold_is_imbalanced(self):
        # 85% one class, 15% the other -> imbalanced at the default 0.8 threshold.
        series = pd.Series([0] * 85 + [1] * 15)
        balance = analyze_class_balance(series, imbalance_threshold=0.8)
        assert balance.is_imbalanced is True

    def test_majority_share_below_threshold_is_not_imbalanced(self):
        # 70/30 split must NOT be flagged imbalanced at the default 0.8 threshold,
        # unlike the old "smallest class < 10%" rule which ignored the threshold's meaning.
        series = pd.Series([0] * 70 + [1] * 30)
        balance = analyze_class_balance(series, imbalance_threshold=0.8)
        assert balance.is_imbalanced is False

    def test_three_classes_evenly_split_not_imbalanced(self):
        series = pd.Series(["A"] * 34 + ["B"] * 33 + ["C"] * 33)
        balance = analyze_class_balance(series, imbalance_threshold=0.8)
        assert balance.is_imbalanced is False


class TestTargetFeatureTable:
    """What the «Features más relacionadas con Target» table may hold, and which test produces it."""

    @staticmethod
    def _frame(n: int = 300, target="bool", seed: int = 0) -> pd.DataFrame:
        """A table shaped like the spaceship data: an id, free text, a category and two numbers."""
        rng = np.random.default_rng(seed)
        edad = rng.normal(40, 12, n)
        transported = (edad < 40) == (np.arange(n) % 10 < 8)  # age separates the two classes
        return pd.DataFrame(
            {
                "PassengerId": [f"{i:04d}_01" for i in range(n)],
                "Name": [f"nombre apellido numero {i}" for i in range(n)],
                "planeta": [["Earth", "Europa", "Mars"][i % 3] for i in range(n)],
                "edad": edad,
                "gasto": rng.lognormal(mean=4, sigma=1.2, size=n),
                "objetivo": transported if target == "bool" else transported.astype(int),
            }
        )

    @staticmethod
    def _tested(report) -> dict:
        return {rel.feature: rel for rel in report.feature_relationships}

    def _report(self, df, target_column="objetivo", **kwargs):
        from eda_pipeline.target_analysis import analyze_target
        from eda_pipeline.type_inference import infer_all_types

        return analyze_target(df, target_column, column_types=infer_all_types(df), **kwargs)

    def test_identifiers_and_free_text_stay_out(self):
        """PassengerId led the table with an effect of 1.000: Cramér's V over 8 693 categories."""
        tested = self._tested(self._report(self._frame()))
        assert "PassengerId" not in tested
        assert "Name" not in tested
        assert {"planeta", "edad", "gasto"} <= set(tested)

    def test_a_classification_target_stored_as_a_boolean_still_compares_groups(self):
        """Transported is a boolean, so every numeric feature took the regression branch and lost its p-value."""
        rel = self._tested(self._report(self._frame(target="bool")))["edad"]
        assert rel.test_name == "kruskal_wallis"
        assert rel.p_value < 0.01

    def test_a_classification_target_stored_as_zero_and_one_does_too(self):
        rel = self._tested(self._report(self._frame(target="int")))["edad"]
        assert rel.test_name == "kruskal_wallis"

    def test_a_numeric_feature_gets_an_effect_between_zero_and_one(self):
        """The same measure the association map uses, so the column can be read next to Cramér's V."""
        tested = self._tested(self._report(self._frame()))
        assert 0.0 <= tested["edad"].effect_size <= 1.0
        assert tested["edad"].effect_size > tested["gasto"].effect_size  # age is what separates them

    def test_a_regression_target_keeps_mutual_information(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 300), "ruido": rng.normal(0, 1, 300)})
        df["precio"] = df["x"] * 1000 + rng.normal(0, 50, 300)
        rel = self._tested(self._report(df, target_column="precio"))["x"]
        assert rel.test_name == "mutual_info"
        assert rel.effect_size > 0

    def test_without_the_types_every_column_is_still_tested(self):
        """The types are optional: callers that do not pass them keep the old behaviour."""
        from eda_pipeline.target_analysis import analyze_target

        report = analyze_target(self._frame(), "objetivo")
        assert "PassengerId" in {rel.feature for rel in report.feature_relationships}


class TestMeasureFormatting:
    """A measured 0 is a result, not a missing value."""

    def test_zero_is_shown_as_zero(self):
        from eda_pipeline.html_report import format_measure

        assert format_measure(0.0, 3) == "0.000"

    def test_a_missing_measure_says_na(self):
        from eda_pipeline.html_report import format_measure

        assert format_measure(None) == "N/A"
        assert format_measure(float("nan")) == "N/A"

    def test_decimals_are_respected(self):
        from eda_pipeline.html_report import format_measure

        assert format_measure(0.0123456, 4) == "0.0123"
        assert format_measure(0.196, 3) == "0.196"

    def test_a_tiny_p_value_does_not_read_as_certainty(self):
        from eda_pipeline.html_report import format_p_value

        assert format_p_value(0.0) == "<0.0001"  # it only underflowed
        assert format_p_value(1e-40) == "<0.0001"
        assert format_p_value(0.0005) == "0.0005"
        assert format_p_value(None) == "N/A"


class TestUnivariateAnalysis:
    """Test univariate analysis."""

    def test_numeric_stats(self, synthetic_dataset):
        stats = analyze_numeric(synthetic_dataset["age"])
        assert stats.count > 0
        assert not np.isnan(stats.mean)
        assert not np.isnan(stats.median)
        assert not np.isnan(stats.std)

    def test_categorical_stats(self, synthetic_dataset):
        stats = analyze_categorical(synthetic_dataset["gender"])
        assert stats.count > 0
        assert stats.nunique > 0
        assert stats.mode in ["M", "F", "Otro"]

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        stats = analyze_numeric(empty)
        assert stats.count == 0

    def test_time_of_day_stats(self):
        series = pd.Series(["09:15:00", "18:30:10", "18:45:00", "21:05:59", None], name="hora", dtype="string")
        stats = analyze_time_of_day(series)
        assert stats.count == 4
        assert stats.missing == 1
        assert stats.earliest == "09:15:00"
        assert stats.latest == "21:05:59"
        assert stats.peak_hour == 18
        assert len(stats.hour_counts) == 24  # every hour, the empty ones included
        assert stats.hour_counts[18] == 2
        assert stats.hour_counts[3] == 0

    def test_time_columns_get_their_own_stats(self):
        df = pd.DataFrame({"hora": pd.Series(["09:15:00", "18:30:10", "18:45:00"], dtype="string")})
        report = analyze_univariate(df, {"hora": "time"})
        assert "hora" in report.time_stats
        assert "hora" not in report.categorical_stats

    def test_time_of_day_plot_is_written(self, tmp_output_dir):
        series = pd.Series(["09:15:00", "18:30:10", "18:45:00", "21:05:59"], name="hora", dtype="string")
        output = tmp_output_dir / "time_of_day_hora.png"
        assert plot_time_of_day(series, output) is True
        assert output.exists()


class TestDatesOverTime:
    """Dates were charted as row number against the date text, with one axis label per distinct day."""

    def test_datetime_stats_read_two_digit_year_dates(self, recwarn):
        from eda_pipeline.univariate_analysis import analyze_datetime

        series = pd.Series(["11/9/07", "11/28/08", "12/1/09", None], name="Date Egg", dtype="string")
        stats = analyze_datetime(series)
        assert (stats.count, stats.missing) == (3, 1)
        assert (stats.min_date, stats.max_date) == ("2007-11-09", "2009-12-01")
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]  # no format guessing

    def test_datetime_stats_read_ambiguous_dates_day_first(self):
        from eda_pipeline.univariate_analysis import analyze_datetime

        # The same day-first preference type detection uses; a bare pd.to_datetime reads May 3rd.
        series = pd.Series(["05/03/07", "20/03/07", "06/04/07"], name="fecha", dtype="string")
        assert analyze_datetime(series).min_date == "2007-03-05"

    def test_rows_are_counted_per_day_over_a_few_weeks(self):
        from eda_pipeline.visualizations import rows_per_period

        dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-03", "2023-01-17"]))
        counts, period = rows_per_period(dates)
        assert period == "día"
        assert len(counts) == 17  # every day of the span, the empty ones included
        assert counts.iloc[0] == 2
        assert counts.iloc[1] == 0
        assert counts.sum() == 4

    def test_a_longer_span_is_counted_per_week_or_month(self):
        from eda_pipeline.visualizations import rows_per_period

        year = pd.Series(pd.date_range("2024-01-01", "2024-12-31", freq="D"))
        counts, period = rows_per_period(year)
        assert period == "semana"
        assert counts.sum() == len(year)

        years = pd.Series(pd.date_range("2019-01-01", "2024-05-31", freq="D"))
        counts, period = rows_per_period(years)
        assert period == "mes"
        assert len(counts) == 65

    def test_the_time_series_plot_is_written_for_text_dates(self, tmp_output_dir):
        from eda_pipeline.visualizations import plot_time_series

        series = pd.Series(["11/9/07", "11/12/07", "11/28/08", "12/1/09"] * 5, name="Date Egg", dtype="string")
        output = tmp_output_dir / "timeseries_Date Egg.png"
        assert plot_time_series(series, output) is True
        assert output.exists()


class TestOutlierDetection:
    """Test outlier detection."""

    def test_iqr_detection(self, synthetic_dataset):
        info = detect_outliers_iqr(synthetic_dataset["age"])
        assert info.method == "iqr"
        assert info.n_outliers > 0  # Should detect outliers we added

    def test_mad_detection(self, synthetic_dataset):
        info = detect_outliers_mad(synthetic_dataset["age"])
        assert info.method == "mad_zscore"
        assert info.n_outliers >= 0

    def test_empty_series(self):
        empty = pd.Series([], dtype=float, name="empty")
        info = detect_outliers_iqr(empty)
        assert info.n_outliers == 0

    @staticmethod
    def _discounts() -> pd.Series:
        # Like discount_pct on Vistara: no discount on 77% of the rows, then a few fixed tiers.
        return pd.Series([0.0] * 77 + [0.1] * 17 + [0.2] * 3 + [0.3] * 3, name="discount_pct")

    def test_iqr_without_spread_flags_nothing(self):
        # Q1 = Q3 = 0, so the "normal" range was [0, 0] and every discount counted as an outlier.
        info = detect_outliers_iqr(self._discounts())
        assert info.n_outliers == 0
        assert "IQR = 0" in info.note

    def test_iqr_with_spread_still_flags_extremes(self):
        series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 500.0], name="monto")
        info = detect_outliers_iqr(series)
        assert info.values == [500.0]
        assert info.note == ""

    def test_mad_without_spread_says_why_it_flags_nothing(self):
        info = detect_outliers_mad(self._discounts())
        assert info.n_outliers == 0
        assert "MAD = 0" in info.note


class TestIqrOnARepeatedValue:
    """A value that fills a quarter of the column pins a quartile, and the fence then measures it."""

    @staticmethod
    def _spending(zeros: int = 650, spenders: int = 350, seed: int = 0) -> pd.Series:
        """Like RoomService on the spaceship data: most passengers spend nothing, the rest spend a lot."""
        rng = np.random.default_rng(seed)
        spent = rng.lognormal(mean=4.0, sigma=1.3, size=spenders)
        return pd.Series(np.concatenate([np.zeros(zeros), spent]), name="RoomService")

    def test_a_column_of_mostly_zeros_is_measured_over_the_rows_that_are_not_zero(self):
        # Before: the fence sat at 2.5 x Q3 and flagged a fifth of the column.
        series = self._spending()
        info = detect_outliers_iqr(series)
        assert info.n_outliers / len(series) * 100 < 8
        assert info.n_outliers > 0  # the real extremes are still found

    def test_the_note_says_which_value_took_over_the_quartiles(self):
        info = detect_outliers_iqr(self._spending())
        assert "0" in info.note and "65.0%" in info.note

    def test_a_column_without_a_repeated_value_keeps_its_fence(self):
        rng = np.random.default_rng(0)
        series = pd.Series(np.concatenate([rng.normal(100, 10, 500), [400.0, 450.0]]), name="medida")
        info = detect_outliers_iqr(series)
        assert info.note == ""
        assert {400.0, 450.0} <= set(info.values)

    def test_a_repeated_value_under_a_quarter_of_the_rows_changes_nothing(self):
        # penguins: 190 mm is exactly Q1 of Flipper Length, but only 6.4% of the rows.
        rng = np.random.default_rng(1)
        values = np.concatenate([np.full(30, 190.0), rng.normal(200, 14, 312)])
        info = detect_outliers_iqr(pd.Series(values, name="Flipper Length (mm)"))
        assert info.note == ""

    def test_a_column_with_no_spread_at_all_still_says_so(self):
        series = pd.Series([0.0] * 77 + [0.1] * 17 + [0.2] * 3 + [0.3] * 3, name="discount_pct")
        info = detect_outliers_iqr(series)
        assert info.n_outliers == 0
        assert "IQR = 0" in info.note


class TestIsolationForest:
    """Isolation Forest flagged a fixed 10% of every dataset, whatever its data looked like."""

    @staticmethod
    def _with_planted_anomalies(n: int = 1000, n_anomalies: int = 10) -> tuple[pd.DataFrame, list[int]]:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
        planted = list(range(0, n, n // n_anomalies))[:n_anomalies]
        df.loc[planted, ["a", "b", "c"]] = 8.0  # far from everything else
        return df, planted

    def test_cutoff_is_tukeys_upper_fence_over_the_scores(self):
        scores = np.array([0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.80])
        q1, q3 = np.percentile(scores, [25, 75])
        assert isolation_forest_cutoff(scores) == pytest.approx(q3 + 1.5 * (q3 - q1))
        assert isolation_forest_cutoff(scores, multiplier=3.0) == pytest.approx(q3 + 3.0 * (q3 - q1))

    def test_cutoff_without_spread_is_none(self):
        assert isolation_forest_cutoff(np.full(50, 0.5)) is None

    def test_flags_the_isolated_rows_and_not_a_fixed_share(self):
        df, planted = self._with_planted_anomalies()
        results, note = detect_outliers_isolation_forest(df, ["a", "b", "c"])
        flagged = set(results[0].outlier_indices)
        assert set(planted) <= flagged
        assert len(flagged) < 0.05 * len(df)  # the fixed rule flagged exactly 10%
        assert "Q3" in note

    def test_explicit_contamination_keeps_the_fixed_share(self):
        df, _ = self._with_planted_anomalies()
        results, note = detect_outliers_isolation_forest(df, ["a", "b", "c"], contamination=0.1)
        assert results[0].n_outliers == 100
        assert "10%" in note

    def test_too_few_rows_are_not_scored(self):
        df, _ = self._with_planted_anomalies(n=ISOLATION_FOREST_MIN_ROWS - 1, n_anomalies=1)
        results, note = detect_outliers_isolation_forest(df, ["a", "b", "c"])
        assert results == []
        assert "No aplicado" in note


class TestRelationships:
    """Test relationship analysis."""

    def test_pearson_correlation(self):
        x = pd.Series(np.random.normal(0, 1, 100))
        y = x + np.random.normal(0, 0.1, 100)  # Highly correlated
        corr, p = pearson_correlation(x, y)
        assert corr > 0.9
        assert p < 0.05

    def test_cramers_v(self):
        x = pd.Series(np.random.choice(["A", "B"], 100))
        y = x  # Perfectly associated
        v = cramers_v(x, y)
        assert v > 0.9

    def test_cramers_v_matches_bergsma_bias_correction(self):
        # 2x2 table [[30, 20], [20, 30]], n=100 -> chi2=4 (no Yates) -> bias-corrected V ~= 0.1738.
        x = pd.Series(["A"] * 50 + ["B"] * 50)
        y = pd.Series(["A"] * 30 + ["B"] * 20 + ["A"] * 20 + ["B"] * 30)
        assert cramers_v(x, y) == pytest.approx(0.1738, abs=1e-3)

    def test_cramers_v_perfect_association_is_one(self):
        x = pd.Series(["A", "B", "C", "D"] * 25)
        assert cramers_v(x, x) == pytest.approx(1.0)

    def test_cramers_v_id_like_column_on_tiny_sample_is_zero_without_warnings(self, recwarn):
        # As many categories as rows: the old formula took the sqrt of a negative number here
        # (RuntimeWarning) and silently turned the NaN into 0.
        x = pd.Series(list("abcdefgh"))
        y = pd.Series(["x", "y", "z", "w"] * 2)
        assert cramers_v(x, y) == 0.0
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_correlation_ratio(self):
        cat = pd.Series(np.repeat(["A", "B", "C"], 100))
        num = pd.Series(
            np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100), np.random.normal(10, 1, 100)])
        )
        eta = correlation_ratio(cat, num)
        assert eta > 0.5  # Should be highly associated


class TestLogScale:
    """Rule D: log scale only when the middle of the data is squeezed, the log fixes it, and few values are <= 0."""

    @staticmethod
    def _amounts(n: int = 2000, seed: int = 0) -> pd.Series:
        return pd.Series(np.random.default_rng(seed).lognormal(mean=13, sigma=2.5, size=n), name="monto")

    def test_skewed_amounts_need_a_log_scale(self):
        from eda_pipeline.visualizations import LOG_SCALE_MAX_CENTRAL_SHARE, check_log_scale

        check = check_log_scale(self._amounts())
        assert check.needed
        assert check.central_share < LOG_SCALE_MAX_CENTRAL_SHARE
        assert check.central_share_log >= 2 * check.central_share

    def test_a_symmetric_measure_does_not(self):
        from eda_pipeline.visualizations import check_log_scale

        series = pd.Series(np.random.default_rng(0).normal(4000, 500, 2000), name="masa")
        assert not check_log_scale(series).needed

    def test_many_values_at_or_below_zero_rule_it_out(self):
        # prima_cedida: 84% zeros. A log axis would silently drop most of the rows.
        from eda_pipeline.visualizations import check_log_scale

        values = self._amounts()
        values.iloc[::10] = 0.0  # 10% zeros
        check = check_log_scale(values)
        assert not check.needed
        assert check.nonpositive_pct == pytest.approx(10.0)

    def test_a_few_values_at_or_below_zero_are_allowed_and_counted(self):
        from eda_pipeline.visualizations import check_log_scale

        values = self._amounts()
        values.iloc[::50] = -5.0  # 2%
        check = check_log_scale(values)
        assert check.needed
        assert check.nonpositive_count == 40

    def test_no_gain_from_the_log_means_no_log(self):
        # The middle 90% is a single value: no scale spreads it, linear or log.
        from eda_pipeline.visualizations import check_log_scale

        values = pd.Series([100.0] * 1940 + list(np.linspace(1e3, 1e7, 60)), name="tarifa")
        assert not check_log_scale(values).needed

    def test_only_continuous_columns_are_checked(self):
        from eda_pipeline.visualizations import log_scale_columns

        df = pd.DataFrame(
            {"monto": self._amounts(), "saldo": self._amounts(seed=1), "masa": np.linspace(3000, 5000, 2000)}
        )
        types = {"monto": "numeric_continuous", "saldo": "numeric_discrete", "masa": "numeric_continuous"}
        assert set(log_scale_columns(df, types)) == {"monto"}

    @staticmethod
    def _width(path) -> int:
        import matplotlib.image as mpimg

        return mpimg.imread(path).shape[1]

    def test_histogram_with_log_scale_keeps_the_linear_one_beside_it(self, tmp_output_dir):
        from eda_pipeline.visualizations import plot_histogram

        amounts = self._amounts()
        assert plot_histogram(amounts, tmp_output_dir / "lineal.png") is True
        assert plot_histogram(amounts, tmp_output_dir / "log.png", log_scale=True) is True
        assert self._width(tmp_output_dir / "log.png") > 1.5 * self._width(tmp_output_dir / "lineal.png")

    def test_boxplot_with_log_scale_keeps_the_linear_one_beside_it(self, tmp_output_dir):
        from eda_pipeline.visualizations import plot_boxplot

        amounts = self._amounts()
        assert plot_boxplot(amounts, tmp_output_dir / "lineal.png") is True
        assert plot_boxplot(amounts, tmp_output_dir / "log.png", log_scale=True) is True
        assert self._width(tmp_output_dir / "log.png") > 1.5 * self._width(tmp_output_dir / "lineal.png")

    def test_scatter_with_log_scale_keeps_the_linear_one_beside_it(self, tmp_output_dir):
        from eda_pipeline.visualizations import plot_scatter

        x = self._amounts()
        y = (x * np.random.default_rng(2).uniform(0.5, 1.0, len(x))).rename("saldo")
        assert plot_scatter(x, y, tmp_output_dir / "lineal.png") is True
        assert plot_scatter(x, y, tmp_output_dir / "log.png", log_x=True, log_y=True) is True
        assert self._width(tmp_output_dir / "log.png") > 1.5 * self._width(tmp_output_dir / "lineal.png")

    def test_pair_plot_log_version_is_written(self, tmp_output_dir):
        from eda_pipeline.visualizations import choose_pair_plot, plot_pair_plot

        x = self._amounts()
        df = pd.DataFrame({"monto": x, "saldo": x * 0.8, "plazo": np.linspace(12, 360, len(x))})
        spec = choose_pair_plot(df, dict.fromkeys(df.columns, "numeric_continuous"))
        output = tmp_output_dir / "pair_plot_log.png"
        assert plot_pair_plot(df, spec, output, log_columns={"monto", "saldo"}) is True
        assert output.exists()


class TestFloorSplit:
    """A repeated lowest value hides the rest of the column, and a log scale cannot fix a zero."""

    @staticmethod
    def _spending(zeros: int = 650, spenders: int = 350, seed: int = 0) -> pd.Series:
        """Like RoomService: most rows spend nothing, the rest spend amounts spread over decades."""
        rng = np.random.default_rng(seed)
        spent = rng.lognormal(mean=4.0, sigma=1.3, size=spenders)
        return pd.Series(np.concatenate([np.zeros(zeros), spent]), name="RoomService")

    def test_a_column_of_mostly_zeros_is_split(self):
        from eda_pipeline.visualizations import check_floor_split

        check = check_floor_split(self._spending())
        assert check.needed
        assert check.floor == 0.0
        assert check.floor_count == 650
        assert check.floor_pct == pytest.approx(65.0)
        assert check.rest_count == 350

    def test_the_rest_carries_its_own_log_scale_when_it_needs_one(self):
        from eda_pipeline.visualizations import check_floor_split

        assert check_floor_split(self._spending()).rest_log

    def test_a_column_that_already_gets_a_log_panel_is_not_split(self):
        """numero_creditos repeats its minimum on 55% of the rows, but its minimum is 1, so the log works."""
        from eda_pipeline.visualizations import check_floor_split, check_log_scale

        rng = np.random.default_rng(0)
        values = pd.Series(np.concatenate([np.ones(550), rng.lognormal(mean=3, sigma=2.0, size=450)]), name="creditos")
        assert check_log_scale(values).needed
        assert not check_floor_split(values).needed

    def test_a_few_values_at_the_floor_do_not_split_anything(self):
        from eda_pipeline.visualizations import check_floor_split

        check = check_floor_split(self._spending(zeros=50, spenders=950))
        assert not check.needed
        assert check.floor_pct == pytest.approx(5.0)

    def test_the_rest_must_be_more_than_a_handful_of_values(self):
        """Order_Details.discount_pct is 0 on 77% of the rows and holds four other values."""
        from eda_pipeline.visualizations import check_floor_split

        tiers = [0.0] * 77 + [0.1] * 17 + [0.2] * 3 + [0.3] * 3
        values = pd.Series([tiers[i % 100] for i in range(600)], name="discount_pct")
        assert not check_floor_split(values).needed

    def test_a_chart_that_already_reads_well_is_not_split(self):
        """The solar data is 0 at night, and the daytime values still fill the axis."""
        from eda_pipeline.visualizations import check_floor_split

        rng = np.random.default_rng(1)
        values = pd.Series(np.concatenate([np.zeros(460), rng.uniform(0, 1000, 540)]), name="AC_POWER")
        assert not check_floor_split(values).needed

    def test_too_few_values_to_judge(self):
        from eda_pipeline.visualizations import check_floor_split

        assert check_floor_split(pd.Series([0.0] * 8 + [1.0, 2.0], name="corta")) is None

    def test_only_continuous_columns_are_considered(self):
        from eda_pipeline.visualizations import floor_split_columns

        df = pd.DataFrame(
            {
                "gasto": self._spending(),
                "grupo": [["a", "b"][i % 2] for i in range(1000)],
                "conteo": [i % 4 for i in range(1000)],
            }
        )
        chosen = floor_split_columns(
            df, {"gasto": "numeric_continuous", "grupo": "categorical", "conteo": "numeric_discrete"}
        )
        assert set(chosen) == {"gasto"}

    def test_the_whole_column_stays_beside_the_split_histogram(self, tmp_output_dir):
        from eda_pipeline.visualizations import check_floor_split, plot_histogram

        series = self._spending()
        plain = tmp_output_dir / "hist_plain.png"
        split = tmp_output_dir / "hist_split.png"
        assert plot_histogram(series, plain)
        assert plot_histogram(series, split, floor_split=check_floor_split(series))

        import matplotlib.image as mpimg

        assert mpimg.imread(split).shape[1] > 1.5 * mpimg.imread(plain).shape[1]

    def test_the_split_boxplot_keeps_both_panels(self, tmp_output_dir):
        from eda_pipeline.visualizations import check_floor_split, plot_boxplot

        series = self._spending()
        plain = tmp_output_dir / "box_plain.png"
        split = tmp_output_dir / "box_split.png"
        assert plot_boxplot(series, plain)
        assert plot_boxplot(series, split, floor_split=check_floor_split(series))

        import matplotlib.image as mpimg

        assert mpimg.imread(split).shape[1] > 1.4 * mpimg.imread(plain).shape[1]


class TestPairPlot:
    """One pair plot of the numeric variables, coloured by the group that best separates them."""

    @staticmethod
    def _penguins(n: int = 300) -> tuple[pd.DataFrame, dict[str, str]]:
        rng = np.random.default_rng(0)
        species = np.array(["Adelie", "Gentoo", "Chinstrap"])[np.arange(n) % 3]
        offset = pd.Series(species).map({"Adelie": 0.0, "Gentoo": 10.0, "Chinstrap": 5.0}).to_numpy()
        df = pd.DataFrame(
            {
                "pico": 40 + offset + rng.normal(0, 1, n),
                "aleta": 190 + 2 * offset + rng.normal(0, 2, n),
                "masa": 3500 + 100 * offset + rng.normal(0, 50, n),
                "especie": species,
                "isla": rng.choice(["Biscoe", "Dream"], n),  # unrelated to the measures
            }
        )
        types = {"pico": "numeric_continuous", "aleta": "numeric_continuous", "masa": "numeric_continuous"}
        types |= {"especie": "categorical", "isla": "categorical"}
        return df, types

    def test_colours_by_the_group_that_best_separates_the_variables(self):
        df, types = self._penguins()
        spec = choose_pair_plot(df, types)
        assert spec.columns == ["pico", "aleta", "masa"]
        assert spec.hue == "especie"
        assert spec.hue_eta > 0.9

    def test_a_group_that_separates_nothing_is_not_used(self):
        df, types = self._penguins()
        df = df.drop(columns="especie")
        types.pop("especie")
        spec = choose_pair_plot(df, types)
        assert spec.hue is None  # isla is unrelated to the measures
        assert spec.best_eta < PAIR_PLOT_MIN_HUE_ETA

    def test_a_classification_target_is_the_group_even_if_weak(self):
        df, types = self._penguins()
        df["objetivo"] = np.arange(len(df)) % 2
        types["objetivo"] = "boolean"
        spec = choose_pair_plot(df, types, target_column="objetivo", target_type="classification")
        assert spec.hue == "objetivo"

    def test_groups_too_many_or_too_small_are_skipped(self):
        df, types = self._penguins()
        df["especie"] = df["especie"].where(np.arange(len(df)) >= PAIR_PLOT_MIN_GROUP_ROWS * 3, "Emperador")
        df.loc[: PAIR_PLOT_MIN_GROUP_ROWS - 2, "especie"] = "Rey"  # a group below the minimum size
        df["lote"] = [f"L{i % 8}" for i in range(len(df))]  # eight groups: too many to tell apart
        types["lote"] = "categorical"
        spec = choose_pair_plot(df, types)
        assert spec.hue != "especie"
        assert spec.hue != "lote"

    def test_needs_at_least_three_continuous_columns(self):
        df, types = self._penguins()
        types["masa"] = "numeric_discrete"  # a discrete column is not an axis
        assert choose_pair_plot(df, types) is None

    def test_keeps_the_most_related_columns_when_there_are_too_many(self):
        df, types = self._penguins()
        rng = np.random.default_rng(1)
        for i in range(PAIR_PLOT_MAX_COLUMNS):
            df[f"ruido_{i}"] = rng.normal(size=len(df))
            types[f"ruido_{i}"] = "numeric_continuous"
        spec = choose_pair_plot(df, types)
        assert len(spec.columns) == PAIR_PLOT_MAX_COLUMNS
        assert {"pico", "aleta", "masa"} <= set(spec.columns)

    def test_large_datasets_are_sampled(self):
        df, types = self._penguins(n=6000)
        spec = choose_pair_plot(df, types, max_rows=1000)
        assert spec.rows_plotted == 1000
        assert spec.rows_available == 6000

    def test_writes_the_pair_plot(self, tmp_output_dir):
        df, types = self._penguins()
        output = tmp_output_dir / "pair_plot.png"
        assert plot_pair_plot(df, choose_pair_plot(df, types), output) is True
        assert output.exists()


class TestScatterPairs:
    """The scatter plots promise the most correlated pairs; they used to take the first pairs in column order."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        rng = np.random.default_rng(0)
        base = rng.normal(size=300)
        return pd.DataFrame(
            {
                "ruido_1": rng.normal(size=300),
                "ruido_2": rng.normal(size=300),
                "debil": base + rng.normal(scale=3, size=300),
                "fuerte": base,
                "inverso": -base + rng.normal(scale=0.1, size=300),
            }
        )

    def test_pairs_come_strongest_first_whatever_their_sign(self):
        corr = self._frame().corr()
        pairs = top_correlated_pairs(corr, limit=3)
        assert {pairs[0][0], pairs[0][1]} == {"fuerte", "inverso"}
        assert pairs[0][2] < -0.9  # the sign is kept, the ranking uses |r|
        strengths = [abs(r) for _, _, r in pairs]
        assert strengths == sorted(strengths, reverse=True)

    def test_limit_is_respected_and_undefined_correlations_are_left_out(self):
        df = self._frame()
        df["constante"] = 1.0
        pairs = top_correlated_pairs(df.corr(), limit=4)
        assert len(pairs) == 4
        assert all("constante" not in (a, b) for a, b, _ in pairs)

    def test_no_matrix_no_pairs(self):
        assert top_correlated_pairs(pd.DataFrame(), limit=10) == []


class TestHtmlReport:
    """Test HTML report helpers."""

    def test_preview_rows_limits_to_ten_and_returns_text(self):
        df = pd.DataFrame({"a": range(25), "b": [f"v{i}" for i in range(25)]})
        rows = preview_rows(df)
        assert len(rows) == 10
        assert rows[0] == ["0", "v0"]
        assert all(isinstance(cell, str) for row in rows for cell in row)

    def test_preview_rows_marks_missing_values_with_a_dash(self):
        df = pd.DataFrame({"a": [1.0, None], "b": ["x", None]})
        assert preview_rows(df) == [["1.0", "x"], ["—", "—"]]

    def test_preview_rows_truncates_very_long_values(self):
        # A free-text column must not blow the table up.
        cell = preview_rows(pd.DataFrame({"nota": ["x" * 500]}))[0][0]
        assert len(cell) < 500
        assert cell.endswith("…")

    def test_overall_missing_pct_is_mean_of_column_percentages(self):
        # Inputs are already percentages: 25% and 0% missing -> 12.5% of all cells (not 0.125).
        assert overall_missing_pct({"a": 25.0, "b": 0.0}) == 12.5

    def test_overall_missing_pct_rounds_to_two_decimals(self):
        assert overall_missing_pct({"a": 100 / 3, "b": 0.0, "c": 0.0}) == 11.11

    def test_overall_missing_pct_without_columns_is_zero(self):
        assert overall_missing_pct({}) == 0.0

    def test_column_quality_rows_show_dtype_and_inferred_category(self):
        # The quality table must say how each column was classified: quantity int / discrete.
        # The loader reads with the nullable backend, hence "Int64" rather than "int64".
        rows = column_quality_rows(
            missing_per_column={"quantity": 0.0, "unit_price": 2.5},
            column_types={"quantity": "numeric_discrete", "unit_price": "numeric_continuous"},
            column_dtypes={"quantity": "Int64", "unit_price": "Float64"},
        )
        assert [row["column"] for row in rows] == ["unit_price", "quantity"]  # still sorted by missing
        assert rows[1] == {
            "column": "quantity",
            "dtype": "int",
            "dtype_raw": "Int64",
            "type_label": "Numérica discreta",
            "missing_pct": 0.0,
        }

    def test_column_quality_rows_label_identifiers(self):
        rows = column_quality_rows(
            missing_per_column={"customer_id": 0.0},
            column_types={"customer_id": "identifier"},
            column_dtypes={"customer_id": "string"},
        )
        assert rows[0]["type_label"] == "Identificador"
        assert rows[0]["dtype"] == "texto"

    def test_column_quality_rows_without_types_are_still_written(self):
        # The type maps are optional arguments of generate_html_report.
        assert column_quality_rows({"a": 1.0}) == [
            {"column": "a", "dtype": "—", "dtype_raw": "—", "type_label": "—", "missing_pct": 1.0}
        ]


class TestAssociationMatrix:
    """The heatmap must cover categorical variables, not only the numeric ones."""

    @staticmethod
    def _frame():
        rng = np.random.default_rng(0)
        n = 200
        return pd.DataFrame(
            {
                "edad": rng.normal(40, 10, n),
                "grupo": np.array(["a", "b", "c"])[rng.integers(0, 3, n)],
                "gasto": rng.normal(100, 20, n),
                "activo": rng.choice(["si", "no"], n),
            }
        )

    def test_covers_numeric_and_categorical_columns_in_dataset_order(self):
        df = self._frame()
        matrix = association_matrix(df, ["edad", "gasto"], ["grupo", "activo"])
        assert list(matrix.columns) == ["edad", "grupo", "gasto", "activo"]
        assert list(matrix.index) == list(matrix.columns)
        assert all(matrix.loc[col, col] == 1.0 for col in matrix.columns)

    def test_uses_cramers_v_between_two_categoricals(self):
        df = self._frame()
        matrix = association_matrix(df, ["edad", "gasto"], ["grupo", "activo"])
        assert matrix.loc["grupo", "activo"] == pytest.approx(cramers_v(df["grupo"], df["activo"]))

    def test_uses_the_correlation_ratio_between_categorical_and_numeric(self):
        df = self._frame()
        matrix = association_matrix(df, ["edad", "gasto"], ["grupo", "activo"])
        expected = correlation_ratio(df["grupo"], df["edad"])
        assert matrix.loc["grupo", "edad"] == pytest.approx(expected)
        assert matrix.loc["edad", "grupo"] == pytest.approx(expected)  # symmetric

    def test_keeps_the_sign_of_pearson_between_numerics(self):
        # Pearson carries a direction; Cramér's V and eta do not, and must never gain a fake one.
        df = pd.DataFrame({"a": range(50), "b": [-x for x in range(50)], "g": ["x", "y"] * 25})
        matrix = association_matrix(df, ["a", "b"], ["g"])
        assert matrix.loc["a", "b"] == pytest.approx(-1.0)
        assert matrix.loc["a", "g"] >= 0.0


class TestCategoricalPieChart:
    """A pie per categorical column, in percentages, only while its slices can still be told apart."""

    def test_shares_add_up_to_100_in_frequency_order(self):
        series = pd.Series(["b"] * 3 + ["a"] * 6 + ["c"] * 1 + [None] * 2, name="canal")
        shares = pie_chart_shares(series)
        assert list(shares.index) == ["a", "b", "c"]
        assert list(shares.round(6)) == [60.0, 30.0, 10.0]  # missing values are not a slice

    def test_the_limit_itself_still_gets_a_pie(self):
        series = pd.Series([f"c{i % PIE_MAX_CATEGORIES}" for i in range(120)], name="canal")
        assert pie_chart_shares(series) is not None

    def test_too_many_categories_get_no_pie(self):
        # año in the mortgage data: ten slices of about 10% each, no longer readable as a pie.
        series = pd.Series([f"c{i % (PIE_MAX_CATEGORIES + 1)}" for i in range(120)], name="año")
        assert pie_chart_shares(series) is None

    def test_a_single_category_gets_no_pie(self):
        assert pie_chart_shares(pd.Series(["a"] * 20, name="constante")) is None

    def test_small_slices_carry_no_label_inside_the_pie(self):
        # discount_pct 0.15 is 0.7% of the rows: its label would sit on top of its neighbours'.
        assert pie_slice_label(0.7) == ""
        assert pie_slice_label(16.6) == "16.6 %"

    def test_a_share_that_rounds_to_zero_is_not_shown_as_zero(self):
        assert format_share(0.02) == "<0.1 %"
        assert format_share(76.94) == "76.9 %"

    def test_writes_the_pie(self, tmp_output_dir):
        series = pd.Series([0.0] * 77 + [0.1] * 17 + [0.3] * 3 + [0.2] * 3, name="discount_pct")
        output = tmp_output_dir / "pie_discount_pct.png"
        assert plot_pie_chart(series, output) is True
        assert output.exists()

    def test_no_file_for_a_column_with_too_many_categories(self, tmp_output_dir):
        series = pd.Series([f"c{i % 12}" for i in range(120)], name="municipio")
        output = tmp_output_dir / "pie_municipio.png"
        assert plot_pie_chart(series, output) is False
        assert not output.exists()


class TestTargetCategoricalBars:
    """Grouped bars comparing the target against each categorical variable."""

    @staticmethod
    def _frame(n: int = 120) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "grupo": [["a", "a", "b", "c"][i % 4] for i in range(n)],
                "target": [["si", "no"][i % 2] for i in range(n)],
            }
        )

    def test_counts_cross_the_target_classes_with_the_feature_categories(self):
        df = self._frame()
        counts = target_vs_categorical_counts(df["grupo"], df["target"])
        assert list(counts.index) == ["no", "si"]  # target classes, in a stable order
        assert set(counts.columns) == {"a", "b", "c"}
        assert counts.to_numpy().sum() == len(df)

    def test_categories_are_ordered_by_frequency(self):
        df = self._frame()
        counts = target_vs_categorical_counts(df["grupo"], df["target"])
        assert list(counts.columns) == ["a", "b", "c"]  # "a" appears twice per cycle

    def test_rows_without_feature_or_target_are_dropped(self):
        # The labels are percentages of the total plotted, so incomplete rows must not count.
        df = pd.DataFrame(
            {
                "grupo": ["a", "b", None, "a"],
                "target": ["si", "no", "si", None],
            }
        )
        counts = target_vs_categorical_counts(df["grupo"], df["target"])
        assert counts.to_numpy().sum() == 2

    def test_rare_categories_are_grouped_into_otros(self):
        df = pd.DataFrame(
            {
                "grupo": [f"c{i % 12}" for i in range(240)],
                "target": [["si", "no"][i % 2] for i in range(240)],
            }
        )
        counts = target_vs_categorical_counts(df["grupo"], df["target"], max_categories=3)
        assert len(counts.columns) == 4
        assert counts.columns[-1] == "Otros"
        assert counts.to_numpy().sum() == 240  # grouping never loses rows

    def test_numeric_target_classes_are_ordered_as_numbers(self):
        """Read as text, class 10 would sit between 1 and 2."""
        n = 132
        df = pd.DataFrame(
            {
                "grupo": [["a", "b"][i % 2] for i in range(n)],
                "meses_mora": [i % 11 for i in range(n)],
            }
        )
        counts = target_vs_categorical_counts(df["grupo"], df["meses_mora"])
        assert list(counts.index) == [str(i) for i in range(11)]

    def test_never_shows_more_categories_than_the_palette_has_colours(self):
        """Two categories drawn in the same colour read as one; the default cycle sets the limit."""
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "grupo": [f"c{i % 30}" for i in range(600)],
                "target": [["si", "no"][i % 2] for i in range(600)],
            }
        )
        counts = target_vs_categorical_counts(df["grupo"], df["target"])
        assert len(counts.columns) <= len(plt.rcParams["axes.prop_cycle"])

    def test_writes_the_plot(self, tmp_output_dir):
        df = self._frame()
        output = tmp_output_dir / "target_bars_grupo.png"
        assert plot_target_vs_categorical(df["grupo"], df["target"], output) is True
        assert output.exists()

    def test_single_class_target_has_nothing_to_compare(self, tmp_output_dir):
        df = self._frame()
        df["target"] = "si"
        output = tmp_output_dir / "target_bars_grupo.png"
        assert plot_target_vs_categorical(df["grupo"], df["target"], output) is False
        assert not output.exists()

    def test_target_with_too_many_classes_is_skipped(self, tmp_output_dir):
        n = (TARGET_BAR_MAX_CLASSES + 1) * 10
        df = pd.DataFrame(
            {
                "grupo": [["a", "b"][i % 2] for i in range(n)],
                "target": [f"clase{i % (TARGET_BAR_MAX_CLASSES + 1)}" for i in range(n)],
            }
        )
        output = tmp_output_dir / "target_bars_grupo.png"
        assert plot_target_vs_categorical(df["grupo"], df["target"], output) is False

    def test_constant_feature_is_skipped(self, tmp_output_dir):
        df = self._frame()
        df["grupo"] = "a"
        output = tmp_output_dir / "target_bars_grupo.png"
        assert plot_target_vs_categorical(df["grupo"], df["target"], output) is False


class TestPreprocessingRecommendations:
    """The preprocessing plan: only the decisions the measurements settle on their own."""

    @staticmethod
    def _plan(df, target_column=None):
        """Build the plan from the same analyses the pipeline runs."""
        from eda_pipeline.recommendations import build_recommendations
        from eda_pipeline.relationships import compute_correlation_matrix
        from eda_pipeline.type_inference import get_numeric_columns
        from eda_pipeline.visualizations import log_scale_columns

        types = infer_all_types(df)
        numeric_cols = get_numeric_columns(types)
        return build_recommendations(
            df,
            types,
            quality=analyze_data_quality(df, column_types=types),
            numeric_stats=analyze_univariate(df, types).numeric_stats,
            correlation_matrix=compute_correlation_matrix(df, numeric_cols),
            log_scale_columns=log_scale_columns(df, types),
            target_column=target_column,
        )

    @staticmethod
    def _rows(plan, step=None, column=None):
        return [r for r in plan if (step is None or r.step == step) and (column is None or r.column == column)]

    @staticmethod
    def _frame(n=300, seed=0, **columns):
        """A usable base frame: one amount, one measure and one group, plus whatever the test adds."""
        rng = np.random.default_rng(seed)
        data = {
            "monto": rng.lognormal(mean=10, sigma=1.2, size=n),
            "medida": rng.normal(50, 5, n),
            "grupo": [["norte", "sur", "centro"][i % 3] for i in range(n)],
        }
        data.update(columns)
        return pd.DataFrame(data)

    # --- convertir -------------------------------------------------------------------------

    def test_numbers_kept_as_text_are_converted_before_anything_else(self):
        # amazon.rating: 1464 numbers read as 28 categories, so the plan offered to encode a rating.
        from eda_pipeline.recommendations import STEP_CONVERT

        plan = self._plan(self._frame(rating=[f"{3 + i % 12 / 10:.1f}" for i in range(300)]))

        rec = self._rows(plan, column="rating")
        assert len(rec) == 1
        assert rec[0].step == STEP_CONVERT
        assert "300 de sus 300 valores" in rec[0].evidence
        assert plan[0].step == STEP_CONVERT  # nothing else can be decided until it is a number

    # --- descartar -------------------------------------------------------------------------

    def test_a_column_that_is_almost_all_gaps_is_dropped(self):
        # penguins_lter.Comments: 92.4% missing, and nothing else in the data comes near it.
        from eda_pipeline.recommendations import STEP_DROP

        notas = [None] * 280 + [f"obs {i}" for i in range(20)]
        plan = self._plan(self._frame(notas=notas))

        rec = self._rows(plan, step=STEP_DROP, column="notas")
        assert len(rec) == 1
        assert "93.3%" in rec[0].evidence

    def test_a_column_with_a_single_value_is_dropped(self):
        from eda_pipeline.recommendations import STEP_DROP

        plan = self._plan(self._frame(moneda=["MXN"] * 300))
        assert self._rows(plan, step=STEP_DROP, column="moneda")

    def test_an_identifier_is_excluded_from_the_model(self):
        from eda_pipeline.recommendations import STEP_DROP

        plan = self._plan(self._frame(cliente_id=[f"C{i:05d}" for i in range(300)]))

        rec = self._rows(plan, step=STEP_DROP, column="cliente_id")
        assert len(rec) == 1
        assert "300 valores distintos" in rec[0].evidence

    def test_repeated_rows_are_reported_for_the_whole_table(self):
        from eda_pipeline.recommendations import STEP_DROP

        df = self._frame(n=100)
        df = pd.concat([df, df.head(10)], ignore_index=True)
        plan = self._plan(df)

        rec = [r for r in self._rows(plan, step=STEP_DROP) if r.column is None]
        assert len(rec) == 1
        assert "10 de 110 filas" in rec[0].evidence

    def test_two_columns_that_say_the_same_thing_keep_only_one(self):
        # credito_asegurado: monto_credito ~ saldo_principal, r = 0.995.
        from eda_pipeline.recommendations import STEP_DROP

        df = self._frame()
        rng = np.random.default_rng(7)
        df["saldo"] = df["monto"] * 0.98 + rng.normal(0, df["monto"].std() * 0.08, len(df))
        plan = self._plan(df)

        rec = [r for r in self._rows(plan, step=STEP_DROP) if r.column and "monto" in r.column and "saldo" in r.column]
        assert len(rec) == 1
        assert "r = 0.99" in rec[0].evidence

    def test_a_dropped_column_gets_no_further_recommendation(self):
        from eda_pipeline.recommendations import STEP_DROP

        plan = self._plan(self._frame(cliente_id=[f"C{i:05d}" for i in range(300)]))

        steps = {r.step for r in self._rows(plan, column="cliente_id")}
        assert steps == {STEP_DROP}

    # --- imputar ---------------------------------------------------------------------------

    def test_a_skewed_column_is_imputed_with_the_median_because_the_mean_is_pulled(self):
        from eda_pipeline.recommendations import STEP_IMPUTE

        df = self._frame()
        df.loc[df.index[:20], "monto"] = None
        plan = self._plan(df)

        rec = self._rows(plan, step=STEP_IMPUTE, column="monto")
        assert len(rec) == 1
        assert "mediana" in rec[0].action.lower()
        assert "desplazada" in rec[0].evidence

    def test_a_symmetric_column_says_both_averages_agree(self):
        from eda_pipeline.recommendations import MEAN_MEDIAN_GAP, STEP_IMPUTE

        df = self._frame()
        df.loc[df.index[:20], "medida"] = None
        plan = self._plan(df)

        rec = self._rows(plan, step=STEP_IMPUTE, column="medida")
        assert len(rec) == 1
        assert "coinciden" in rec[0].evidence
        assert f"{MEAN_MEDIAN_GAP:.2f}" in rec[0].evidence

    def test_a_category_with_few_gaps_is_filled_with_the_most_common_value(self):
        from eda_pipeline.recommendations import STEP_IMPUTE

        df = self._frame()
        df.loc[df.index[:9], "grupo"] = None  # 3%
        plan = self._plan(df)

        rec = self._rows(plan, step=STEP_IMPUTE, column="grupo")
        assert len(rec) == 1
        assert "moda" in rec[0].action.lower()

    def test_a_category_with_many_gaps_gets_its_own_unknown_value(self):
        from eda_pipeline.recommendations import STEP_IMPUTE

        df = self._frame()
        df.loc[df.index[:60], "grupo"] = None  # 20%
        plan = self._plan(df)

        rec = self._rows(plan, step=STEP_IMPUTE, column="grupo")
        assert len(rec) == 1
        assert "Desconocido" in rec[0].action

    def test_a_column_without_gaps_is_not_imputed(self):
        from eda_pipeline.recommendations import STEP_IMPUTE

        plan = self._plan(self._frame())
        assert self._rows(plan, step=STEP_IMPUTE) == []

    # --- codificar -------------------------------------------------------------------------

    def test_a_two_valued_column_becomes_one_zero_one_column(self):
        from eda_pipeline.recommendations import STEP_ENCODE

        plan = self._plan(self._frame(activo=[["si", "no"][i % 2] for i in range(300)]))

        rec = self._rows(plan, step=STEP_ENCODE, column="activo")
        assert len(rec) == 1
        assert "0/1" in rec[0].action

    def test_a_handful_of_categories_get_one_hot(self):
        from eda_pipeline.recommendations import STEP_ENCODE

        plan = self._plan(self._frame())

        rec = self._rows(plan, step=STEP_ENCODE, column="grupo")
        assert len(rec) == 1
        assert "one-hot" in rec[0].action.lower()
        assert "3 categorías" in rec[0].evidence

    def test_too_many_categories_rule_one_hot_out(self):
        # clientes.tipo_empleo: 21 categories. credito_asegurado.clave_administrador: 36.
        from eda_pipeline.recommendations import ONE_HOT_MAX_CATEGORIES, STEP_ENCODE

        n_cats = ONE_HOT_MAX_CATEGORIES + 5
        plan = self._plan(self._frame(sector=[f"sector_{i % n_cats}" for i in range(300)]))

        rec = self._rows(plan, step=STEP_ENCODE, column="sector")
        assert len(rec) == 1
        assert rec[0].action.startswith("Evitar one-hot")
        assert f"{n_cats} categorías" in rec[0].evidence

    def test_one_hot_is_ruled_out_when_it_would_add_a_column_per_few_rows(self):
        # Vistara Products.color: 14 categories over 40 rows.
        from eda_pipeline.recommendations import STEP_ENCODE

        df = self._frame(n=40, color=[f"color_{i % 14}" for i in range(40)])
        plan = self._plan(df)

        rec = self._rows(plan, step=STEP_ENCODE, column="color")
        assert len(rec) == 1
        assert rec[0].action.startswith("Evitar one-hot")
        assert "40 filas" in rec[0].evidence
        # Three categories over the same 40 rows are still fine.
        assert "one-hot" in self._rows(plan, step=STEP_ENCODE, column="grupo")[0].action.lower()

    def test_rare_categories_are_grouped_before_one_hot(self):
        from eda_pipeline.recommendations import STEP_ENCODE

        valores = ["comun"] * 290 + ["raro_a"] * 2 + ["raro_b"] * 2 + ["raro_c"] * 2 + ["otro"] * 4
        plan = self._plan(self._frame(canal=valores))

        rec = self._rows(plan, step=STEP_ENCODE, column="canal")
        assert len(rec) == 1
        assert "Otros" in rec[0].action
        assert "3 categorías" in rec[0].evidence and "1%" in rec[0].evidence

    def test_numbers_and_dates_are_not_encoded(self):
        from eda_pipeline.recommendations import STEP_ENCODE

        plan = self._plan(self._frame(fecha=pd.date_range("2024-01-01", periods=300)))
        assert {r.column for r in self._rows(plan, step=STEP_ENCODE)} == {"grupo"}

    # --- escalar ---------------------------------------------------------------------------

    def test_scaling_is_called_for_when_the_spreads_are_incomparable(self):
        # cobranza: the widest column spreads 2 100 million times more than the narrowest.
        from eda_pipeline.recommendations import STEP_SCALE

        plan = self._plan(self._frame())

        table_level = [r for r in self._rows(plan, step=STEP_SCALE) if r.column is None]
        assert len(table_level) == 1
        assert "Escalar" in table_level[0].action
        assert "árboles" in table_level[0].action  # says which models do not need it

    def test_comparable_spreads_need_no_scaling(self):
        from eda_pipeline.recommendations import STEP_SCALE

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"peso": rng.normal(70, 10, 300), "altura": rng.normal(170, 12, 300)})
        plan = self._plan(df)

        table_level = [r for r in self._rows(plan, step=STEP_SCALE) if r.column is None]
        assert len(table_level) == 1
        assert table_level[0].action.startswith("No hace falta")
        assert [r for r in self._rows(plan, step=STEP_SCALE) if r.column] == []

    def test_a_single_numeric_column_is_not_judged_on_scale(self):
        """With one column there is nothing to compare its spread against, so the plan says nothing."""
        from eda_pipeline.recommendations import STEP_SCALE

        df = pd.DataFrame(
            {
                "precio": [10.0 + i * 1.7 for i in range(60)],
                "grupo": [["norte", "sur"][i % 2] for i in range(60)],
            }
        )
        assert self._rows(self._plan(df), step=STEP_SCALE) == []

    def test_a_column_with_a_heavy_tail_gets_a_robust_scaler(self):
        from eda_pipeline.recommendations import STEP_SCALE

        rng = np.random.default_rng(1)
        valores = np.concatenate([rng.normal(100, 5, 270), rng.normal(900, 50, 30)])  # 10% far out
        plan = self._plan(self._frame(sesgada=valores))

        rec = self._rows(plan, step=STEP_SCALE, column="sesgada")
        assert len(rec) == 1
        assert "RobustScaler" in rec[0].action
        assert "veces el rango intercuartil" in rec[0].evidence

    def test_a_column_of_mostly_zeros_with_a_tail_gets_a_robust_scaler(self):
        """RoomService: the corrected IQR fence flags only 2% of it, but its tail still rules the mean."""
        from eda_pipeline.recommendations import STEP_SCALE

        rng = np.random.default_rng(2)
        gasto = np.concatenate([np.zeros(200), rng.lognormal(mean=4.0, sigma=1.3, size=100)])
        plan = self._plan(self._frame(gasto=gasto))

        rec = self._rows(plan, step=STEP_SCALE, column="gasto")
        assert len(rec) == 1
        assert "RobustScaler" in rec[0].action

    def test_a_column_with_no_spread_is_not_sent_to_a_robust_scaler(self):
        """Its interquartile range is 0: RobustScaler would have nothing to divide by."""
        from eda_pipeline.recommendations import STEP_SCALE

        tiers = [0.0] * 77 + [0.1] * 17 + [0.2] * 3 + [0.3] * 3
        plan = self._plan(self._frame(descuento=[tiers[i % 100] for i in range(300)]))

        rec = self._rows(plan, step=STEP_SCALE, column="descuento")
        assert len(rec) == 1
        assert "StandardScaler" in rec[0].action
        assert "IQR = 0" in rec[0].evidence

    def test_a_well_behaved_column_gets_a_standard_scaler(self):
        from eda_pipeline.recommendations import STEP_SCALE

        plan = self._plan(self._frame())

        rec = self._rows(plan, step=STEP_SCALE, column="medida")
        assert len(rec) == 1
        assert "StandardScaler" in rec[0].action
        assert "concuerdan" in rec[0].evidence

    def test_a_column_already_drawn_on_a_log_scale_is_transformed_first(self):
        from eda_pipeline.recommendations import STEP_SCALE

        rng = np.random.default_rng(0)
        plan = self._plan(self._frame(monto=rng.lognormal(mean=13, sigma=2.5, size=300)))

        rec = self._rows(plan, step=STEP_SCALE, column="monto")
        assert len(rec) == 1
        assert "log10" in rec[0].action

    # --- target y orden --------------------------------------------------------------------

    def test_the_target_is_neither_encoded_nor_scaled(self):
        from eda_pipeline.recommendations import STEP_ENCODE, STEP_SCALE

        df = self._frame(resultado=[["alta", "baja"][i % 2] for i in range(300)])
        plan = self._plan(df, target_column="resultado")

        assert self._rows(plan, step=STEP_ENCODE, column="resultado") == []
        assert self._rows(plan, step=STEP_SCALE, column="resultado") == []

    def test_rows_without_a_target_value_are_dropped_instead_of_imputed(self):
        from eda_pipeline.recommendations import STEP_DROP, STEP_IMPUTE

        df = self._frame(resultado=[["alta", "baja"][i % 2] for i in range(300)])
        df.loc[df.index[:12], "resultado"] = None
        plan = self._plan(df, target_column="resultado")

        assert self._rows(plan, step=STEP_IMPUTE, column="resultado") == []
        rec = self._rows(plan, step=STEP_DROP, column="resultado")
        assert len(rec) == 1
        assert "12 filas" in rec[0].evidence

    def test_the_plan_runs_in_preprocessing_order(self):
        from eda_pipeline.recommendations import STEP_ORDER

        df = self._frame(
            cliente_id=[f"C{i:05d}" for i in range(300)],
            rating=[f"{3 + i % 12 / 10:.1f}" for i in range(300)],
        )
        df.loc[df.index[:10], "medida"] = None
        plan = self._plan(df)

        positions = [STEP_ORDER.index(r.step) for r in plan]
        assert positions == sorted(positions)
        assert {r.step for r in plan} == set(STEP_ORDER)

    def test_free_text_alone_produces_no_plan(self):
        df = pd.DataFrame({"comentario": [f"el cliente {i} dejo una nota larga sobre el servicio" for i in range(60)]})
        assert self._plan(df) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
