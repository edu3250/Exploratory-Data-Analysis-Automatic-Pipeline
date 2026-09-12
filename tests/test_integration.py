"""
Integration tests for the complete EDA Pipeline.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from eda_pipeline.config import Config, OutlierConfig, TargetConfig
from eda_pipeline.pipeline import EDAPipeline


def _make_pipeline(**config_kwargs) -> EDAPipeline:
    pipeline = EDAPipeline(Config(**config_kwargs))
    pipeline.setup()
    return pipeline


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_pipeline_with_synthetic_data(self, synthetic_dataset, tmp_output_dir):
        """Test full pipeline on synthetic data."""
        # Save data
        csv_file = tmp_output_dir / "synthetic.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        # Configure
        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))

        # Run
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        # Verify
        assert isinstance(results, dict)
        assert "synthetic" in results
        assert results["synthetic"]["success"] is True
        assert Path(results["synthetic"]["html_report"]).exists()
        assert Path(results["synthetic"]["summary_json"]).exists()

    def test_pipeline_with_classification_target(self, classification_dataset, tmp_output_dir):
        """Test pipeline with classification target."""
        csv_file = tmp_output_dir / "classification.csv"
        classification_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.target.target_column = "target"

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert results["classification"]["success"] is True

    def test_pipeline_with_regression_target(self, regression_dataset, tmp_output_dir):
        """Test pipeline with regression target."""
        csv_file = tmp_output_dir / "regression.csv"
        regression_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.target.target_column = "target"

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert results["regression"]["success"] is True

    def test_pipeline_with_latin1_encoding(self, latin1_csv_file, tmp_output_dir):
        """Test pipeline with special encoding."""
        config = Config(
            input_file=str(latin1_csv_file),
            encoding="latin-1",
            delimiter=";",
            decimal=",",
            output_dir=str(tmp_output_dir),
        )

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert "data_latin1" in results

    def test_pipeline_generates_outputs(self, synthetic_dataset, tmp_output_dir):
        """Verify all expected outputs are generated."""
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        result = results["data"]
        report_dir = Path(result["output_dir"])

        # Check all expected files exist
        assert (report_dir / "report.html").exists()
        assert (report_dir / "summary.json").exists()
        assert (report_dir / "plots").exists()

        # Check plots exist
        plots_dir = report_dir / "plots"
        plot_files = list(plots_dir.glob("*.png"))
        assert len(plot_files) > 0

    def test_pipeline_with_sample_size(self, synthetic_dataset, tmp_output_dir):
        """Test pipeline with sampling for large datasets."""
        csv_file = tmp_output_dir / "large.csv"
        large_df = pd.concat([synthetic_dataset] * 10, ignore_index=True)
        large_df.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir), sample_size=100)

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert results["large"]["success"] is True

    def test_pipeline_with_batch_processing(self, synthetic_dataset, classification_dataset, tmp_output_dir):
        """Test batch processing of multiple files."""
        data_dir = tmp_output_dir / "batch_data"
        data_dir.mkdir()

        synthetic_dataset.to_csv(data_dir / "synthetic.csv", index=False)
        classification_dataset.to_csv(data_dir / "classification.csv", index=False)

        config = Config(input_folder=str(data_dir), output_dir=str(tmp_output_dir))

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert len(results) == 2
        assert all(r["success"] for r in results.values() if "success" in r)

    def test_pipeline_handles_edge_cases(self, tmp_output_dir):
        """Test pipeline with edge case datasets."""
        # Single row
        single_row = pd.DataFrame({"a": [1], "b": ["x"]})
        csv_file = tmp_output_dir / "single_row.csv"
        single_row.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))

        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert "single_row" in results


class TestBatchProcessing:
    """Bug #3/#4: batch mode must skip bad input gracefully and never abort the whole run."""

    def test_batch_skips_hidden_and_unsupported_files(self, tmp_output_dir, synthetic_dataset):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        (data_dir / ".gitkeep").write_text("")
        (data_dir / "notes.txt.bak").write_text("unsupported extension")
        synthetic_dataset.to_csv(data_dir / "data.csv", index=False)

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        assert list(results.keys()) == ["data"]
        assert results["data"]["success"] is True

    def test_batch_one_bad_file_does_not_abort_the_rest(self, tmp_output_dir, synthetic_dataset):
        """A single corrupt file must show up as ❌ for itself while its siblings still succeed."""
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        synthetic_dataset.to_csv(data_dir / "good.csv", index=False)
        (data_dir / "broken.csv").write_bytes(b"\xff\xfe\x00not,valid,,,\x01")

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        assert results["good"]["success"] is True
        # "broken.csv" is malformed bytes, not an unreadable encoding per se, so it may still
        # load; the important contract is that it never takes 'good' down with it either way.
        assert "broken" in results

    def test_batch_stem_collision_gets_unique_dataset_names(self, tmp_output_dir):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(data_dir / "sales.csv", index=False)
        pd.DataFrame({"b": [4, 5, 6]}).to_parquet(data_dir / "sales.parquet")

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        assert set(results.keys()) == {"sales_csv", "sales_parquet"}
        assert all(r["success"] for r in results.values())

    def test_batch_writes_every_report_under_one_run_folder(self, tmp_output_dir, synthetic_dataset):
        """Eight files used to scatter eight timestamped folders across the output directory."""
        data_dir = tmp_output_dir / "Vistara"
        data_dir.mkdir()
        out_dir = tmp_output_dir / "reports"
        synthetic_dataset.to_csv(data_dir / "ventas.csv", index=False)
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(data_dir / "clientes.csv", index=False)

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(out_dir))
        results = pipeline.run()

        dirs = {name: Path(result["output_dir"]) for name, result in results.items()}
        assert set(dirs) == {"ventas", "clientes"}

        parents = {d.parent for d in dirs.values()}
        assert len(parents) == 1  # one folder holds the whole run
        batch_dir = parents.pop()
        assert batch_dir.parent == out_dir
        assert re.fullmatch(r"Vistara_batch_\d{8}_\d{6}", batch_dir.name)

        # Inside it, one subfolder per dataset: the run folder already carries the timestamp.
        assert sorted(p.name for p in batch_dir.iterdir()) == ["clientes", "ventas"]
        assert (dirs["ventas"] / "report.html").exists()
        assert (dirs["clientes"] / "summary.json").exists()

    def test_single_file_keeps_its_own_timestamped_folder(self, tmp_output_dir, synthetic_dataset):
        """Only batches are grouped; one file still writes reports/<dataset>_<timestamp>/."""
        out_dir = tmp_output_dir / "reports"
        csv_file = tmp_output_dir / "ventas.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(out_dir))
        results = pipeline.run()

        output_dir = Path(results["ventas"]["output_dir"])
        assert output_dir.parent == out_dir
        assert re.fullmatch(r"ventas_\d{8}_\d{6}", output_dir.name)

    def test_batch_run_folder_name_slugifies_the_input_folder(self):
        name = EDAPipeline._batch_run_folder_name(Path("data") / "power Bi", "20260911_162444")
        assert name == "power_Bi_batch_20260911_162444"

    def test_batch_honors_pattern_option(self, tmp_output_dir, synthetic_dataset):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        synthetic_dataset.to_csv(data_dir / "data.csv", index=False)
        pd.DataFrame({"x": [1, 2]}).to_json(data_dir / "other.json", orient="records")

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(tmp_output_dir), batch_pattern="*.csv")
        results = pipeline.run()

        assert list(results.keys()) == ["data"]

    def test_batch_loads_and_analyzes_one_file_at_a_time(self, tmp_output_dir, synthetic_dataset, monkeypatch):
        """No more than one DataFrame should be loaded into memory at any given moment."""
        import eda_pipeline.pipeline as pipeline_module

        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        synthetic_dataset.to_csv(data_dir / "one.csv", index=False)
        synthetic_dataset.to_csv(data_dir / "two.csv", index=False)

        in_flight = []
        max_in_flight = []
        original_load_data = pipeline_module.load_data

        def tracking_load_data(*args, **kwargs):
            in_flight.append(1)
            max_in_flight.append(len(in_flight))
            try:
                return original_load_data(*args, **kwargs)
            finally:
                in_flight.pop()

        monkeypatch.setattr(pipeline_module, "load_data", tracking_load_data)

        pipeline = _make_pipeline(input_folder=str(data_dir), output_dir=str(tmp_output_dir))
        pipeline.run()

        assert max(max_in_flight) == 1


class TestTargetColumnHandling:
    """Bug #9: an unknown --target must fail fast (single file) or warn-and-continue (batch)."""

    def test_single_file_missing_target_fails_fast_with_spanish_message(self, tmp_output_dir, synthetic_dataset):
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.target.target_column = "columna_inexistente"
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert "error" in results["data"]
        assert "columna_inexistente" in results["data"]["error"]
        # Fail-fast: no report should have been produced for this dataset.
        assert not any((tmp_output_dir / p).exists() for p in Path(tmp_output_dir).glob("data_*"))

    def test_single_file_missing_target_suggests_close_match(self, tmp_output_dir, classification_dataset):
        csv_file = tmp_output_dir / "classification.csv"
        classification_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.target.target_column = "targe"  # close to 'target'
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert "target" in results["classification"]["error"]

    def test_batch_missing_target_warns_and_continues(self, tmp_output_dir, synthetic_dataset):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        synthetic_dataset.to_csv(data_dir / "data.csv", index=False)

        config = Config(input_folder=str(data_dir), output_dir=str(tmp_output_dir))
        config.target.target_column = "columna_inexistente"
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        assert results["data"]["success"] is True
        summary = json.loads(Path(results["data"]["summary_json"]).read_text(encoding="utf-8"))
        assert "target" not in summary
        assert any("columna_inexistente" in a["message"] for a in summary["alerts"])


class TestColumnTypeOverrides:
    """Bug #13: config.column_types must actually be applied."""

    def test_ignore_removes_column_from_report(self, tmp_output_dir, synthetic_dataset):
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.column_types.ignore = ["comment"]
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        summary = json.loads(Path(results["data"]["summary_json"]).read_text(encoding="utf-8"))
        assert "comment" not in summary["column_types"]
        assert summary["column_overrides"]["ignored_columns"] == ["comment"]

    def test_numeric_override_reports_coercion_failures(self, tmp_output_dir):
        df = pd.DataFrame({"code": ["1", "2", "not_numeric", "4"]})
        csv_file = tmp_output_dir / "data.csv"
        df.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.column_types.numeric = ["code"]
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        summary = json.loads(Path(results["data"]["summary_json"]).read_text(encoding="utf-8"))
        assert summary["column_types"]["code"] == "numeric_continuous"
        assert summary["column_overrides"]["coercion_failures"]["code"] == 1

    def test_unknown_override_column_is_reported(self, tmp_output_dir, synthetic_dataset):
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.column_types.numeric = ["no_existe_esta_columna"]
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        summary = json.loads(Path(results["data"]["summary_json"]).read_text(encoding="utf-8"))
        assert "no_existe_esta_columna" in summary["column_overrides"]["unknown_columns"]


class TestStepIsolation:
    """Bug #5: a single failing analysis step must not take down the whole report."""

    def test_outliers_step_failure_is_isolated(self, tmp_output_dir, synthetic_dataset, monkeypatch):
        import eda_pipeline.pipeline as pipeline_module

        def boom(*args, **kwargs):
            raise RuntimeError("simulated outlier detection crash")

        monkeypatch.setattr(pipeline_module, "analyze_outliers", boom)

        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["data"]
        assert result["success"] is False  # a step failed
        assert result["failed_steps"] == ["outliers"]
        assert Path(result["html_report"]).exists()  # the report was still generated
        assert Path(result["summary_json"]).exists()

        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["failed_steps"] == ["outliers"]

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "outliers" in html  # shown in the "Pasos con Error" section

    def test_strict_mode_reraises_step_failure_immediately(self, tmp_output_dir, synthetic_dataset, monkeypatch):
        import eda_pipeline.pipeline as pipeline_module

        def boom(*args, **kwargs):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(pipeline_module, "analyze_outliers", boom)

        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir), strict_mode=True)

        with pytest.raises(RuntimeError, match="simulated crash"):
            pipeline.run()


class TestOutputCompleteness:
    """Bugs #18/#19/#20: every plot embedded, CSV tables written, target alerts surfaced."""

    def test_csv_tables_are_written(self, tmp_output_dir, synthetic_dataset):
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        tables_dir = Path(results["data"]["output_dir"]) / "tables"
        expected = {
            "numeric_stats.csv",
            "categorical_stats.csv",
            "missing_per_column.csv",
            "correlations.csv",
            "outlier_summary.csv",
            "alerts.csv",
        }
        assert expected.issubset({p.name for p in tables_dir.glob("*.csv")})

    def test_a_column_without_spread_is_not_reported_as_outliers(self, tmp_output_dir):
        """IQR marked 34% of Order_Details as outliers: discount_pct is 0 on 77% of its rows."""
        n = 400
        tiers = [0.0] * 77 + [0.1] * 17 + [0.2] * 3 + [0.3] * 3
        df = pd.DataFrame(
            {
                "discount_pct": [tiers[i % 100] for i in range(n)],
                "unit_price": [100.0 + i % 50 for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "pedidos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(
            input_file=str(csv_file),
            output_dir=str(tmp_output_dir),
            outliers=OutlierConfig(isolation_forest_enabled=False),  # it flags 10% by construction
        )
        results = pipeline.run()

        result = results["pedidos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["outliers"]["total_outlier_rows"] == 0

        table = pd.read_csv(Path(result["output_dir"]) / "tables" / "outlier_summary.csv")
        iqr_row = table[(table["columna"] == "discount_pct") & (table["metodo"] == "iqr")].iloc[0]
        assert iqr_row["n_outliers"] == 0
        assert "IQR = 0" in iqr_row["nota"]

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        outliers_section = html[html.index('<section id="outliers">') :]
        assert "discount_pct" in outliers_section[: outliers_section.index("</section>")]

    def test_report_shows_the_first_rows_before_the_univariate_section(self, tmp_output_dir):
        """A preview of the data must sit ahead of «Análisis Univariado» in the report."""
        n = 30
        df = pd.DataFrame({"ciudad": [f"Ciudad {i % 3}" for i in range(n)], "monto": [100.0 + i for i in range(n)]})
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        html = Path(results["ventas"]["html_report"]).read_text(encoding="utf-8")
        assert "Primeras Filas" in html
        assert '<li><a href="#muestra">' in html  # reachable from the table of contents

        preview_at = html.index('<section id="muestra">')
        univariate_at = html.index('<section id="univariado">')
        assert preview_at < univariate_at

        section = html[preview_at:univariate_at]
        assert section.count("<tr>") == 11  # one header row plus ten data rows
        assert "<th>ciudad</th>" in section
        assert "<th>monto</th>" in section
        assert "Ciudad 0" in section

    @staticmethod
    def _target_dataset(n: int = 200) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "edad": [20 + i % 40 for i in range(n)],
                "genero": [["M", "F"][i % 2] for i in range(n)],
                "fumador": [["si", "no", "no"][i % 3] for i in range(n)],
                "enfermo": [["0", "1"][i % 2] for i in range(n)],
            }
        )

    def test_target_analysis_adds_grouped_bars_per_categorical(self, tmp_output_dir):
        """With a target, the report compares it against every categorical variable."""
        csv_file = tmp_output_dir / "pacientes.csv"
        self._target_dataset().to_csv(csv_file, index=False)
        pipeline = _make_pipeline(
            input_file=str(csv_file),
            output_dir=str(tmp_output_dir),
            target=TargetConfig(target_column="enfermo"),
        )
        results = pipeline.run()

        result = results["pacientes"]
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "target_bars_genero.png" in plots
        assert "target_bars_fumador.png" in plots
        assert "target_bars_enfermo.png" not in plots  # the target against itself says nothing

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Target vs Variables Categóricas" in html
        categorical_at = html.index("Distribuciones Categóricas")
        target_bars_at = html.index("Target vs Variables Categóricas")
        assert categorical_at < target_bars_at

    def test_a_regression_target_gets_no_grouped_bars(self, tmp_output_dir):
        """Bars per class need classes: a continuous target would draw one group per value."""
        n = 200
        df = self._target_dataset(n)
        df["monto"] = [1000.0 + i * 3.7 for i in range(n)]
        csv_file = tmp_output_dir / "pacientes.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(
            input_file=str(csv_file),
            output_dir=str(tmp_output_dir),
            target=TargetConfig(target_column="monto"),
        )
        results = pipeline.run()

        result = results["pacientes"]
        assert json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))["target"]["target_type"] == (
            "regression"
        )
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert not any(name.startswith("target_bars_") for name in plots)

    def test_without_a_target_there_are_no_grouped_bars(self, tmp_output_dir):
        csv_file = tmp_output_dir / "pacientes.csv"
        self._target_dataset().to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["pacientes"]
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert not any(name.startswith("target_bars_") for name in plots)
        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Target vs Variables Categóricas" not in html

    def test_numeric_id_columns_are_not_analysed_as_variables(self, tmp_output_dir):
        """A key stored as a number used to get a histogram, a boxplot, VIF and scatter plots."""
        n = 300
        df = pd.DataFrame(
            {
                "id": range(1, n + 1),
                "edad": [20 + i % 50 for i in range(n)],
                "monto": [100.0 + i for i in range(n)],
                "ciudad": [["a", "b", "c"][i % 3] for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "datos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["datos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        # A numeric identifier reaches the text statistics, which must handle it without failing.
        assert summary["failed_steps"] == []
        assert summary["column_types"]["id"] == "identifier"
        assert "id" not in summary["relationships"]["multicollinearity_vif"]

        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "histogram_id.png" not in plots
        assert "boxplot_id.png" not in plots
        assert {p for p in plots if p.startswith("scatter_")} == {"scatter_edad_vs_monto.png"}
        assert "histogram_edad.png" in plots  # the real variables are untouched

    def test_report_heatmap_includes_categorical_variables(self, tmp_output_dir):
        """The heatmap covered only numeric columns, leaving every categorical one invisible."""
        n = 120
        df = pd.DataFrame(
            {
                "edad": [20 + i % 40 for i in range(n)],
                "grupo": [["a", "b", "c"][i % 3] for i in range(n)],
                "gasto": [100.0 + i for i in range(n)],
                "activo": [["si", "no"][i % 2] for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "datos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["datos"]
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "association_heatmap.png" in plots

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Asociación entre Variables" in html

    def test_quality_table_shows_how_each_column_was_classified(self, tmp_output_dir):
        """The «Calidad de Datos» table must show the dtype and the inferred category per column."""
        n = 120
        df = pd.DataFrame(
            {
                "quantity": [i % 5 + 1 for i in range(n)],
                "unit_price": [10.5 + i for i in range(n)],
                "ciudad": [f"Ciudad {i % 4}" for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["ventas"]
        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "<th>Tipo de dato</th>" in html
        assert "<th>Categoría inferida</th>" in html
        for cell in (
            '<td><code title="Int64">int</code></td>',  # exact pandas dtype kept in the tooltip
            '<td><code title="Float64">float</code></td>',
            "<td>Numérica discreta</td>",
            "<td>Numérica continua</td>",
            "<td>Categórica</td>",
        ):
            assert cell in html

        # The CSV twin of that table carries the same two columns.
        csv_text = (Path(result["output_dir"]) / "tables" / "missing_per_column.csv").read_text(encoding="utf-8")
        assert "columna,tipo_dato,categoria_inferida,pct_faltante" in csv_text
        assert "quantity,int,Numérica discreta" in csv_text

    def test_html_report_embeds_boxplots_and_timeseries(self, tmp_output_dir, synthetic_dataset):
        """Bug #18: boxplots and time series were generated but never embedded in the report."""
        csv_file = tmp_output_dir / "data.csv"
        synthetic_dataset.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["data"]
        plots_dir = Path(result["output_dir"]) / "plots"
        n_plot_files = len(list(plots_dir.glob("*.png")))
        assert n_plot_files == result["n_plots"]  # every generated plot is accounted for

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Boxplots" in html
        assert "Series Temporales" in html
        # There must be one embedded <img> per generated plot.
        assert html.count('<img src="data:image/png;base64,') == n_plot_files

    def test_target_imbalance_and_leakage_alerts_surface_in_summary_and_html(self, tmp_output_dir):
        n = 200
        target_values = [0] * 180 + [1] * 20  # 90% majority -> imbalanced at the default 0.8 threshold
        df = pd.DataFrame(
            {
                "feature": range(n),
                "leaky_copy": target_values,  # identical to target -> correlation 1.0 -> leakage alert
                "target": target_values,
            }
        )
        csv_file = tmp_output_dir / "data.csv"
        df.to_csv(csv_file, index=False)

        config = Config(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        config.target.target_column = "target"
        pipeline = EDAPipeline(config)
        pipeline.setup()
        results = pipeline.run()

        result = results["data"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["target"]["class_balance"]["is_imbalanced"] is True

        alert_messages = " ".join(a["message"] for a in summary["alerts"])
        assert "Desbalance" in alert_messages
        assert any("fuga" in a["message"].lower() for a in summary["alerts"])

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Desbalance" in html

    def test_html_summary_shows_overall_missing_percentage(self, tmp_output_dir):
        """The «Valores Faltantes» stat box divided an already-percent value by 100 (showed 0.125%)."""
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0],  # 25% missing
                "b": ["x", "y", "z", "w", "x", "y", "z", "w"],  # 0% missing -> 12.5% of all cells
            }
        )
        csv_file = tmp_output_dir / "data.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        html = Path(results["data"]["html_report"]).read_text(encoding="utf-8")
        assert '<div class="value">12.5%</div>' in html


class TestIdentifierColumns:
    """ID columns must not be reported as high cardinality (found on the Vistara dataset)."""

    def test_id_columns_are_not_flagged_as_high_cardinality(self, tmp_output_dir):
        n = 600
        df = pd.DataFrame(
            {
                "transaction_id": [f"TXN-{i:06d}" for i in range(n)],  # primary key: unique
                "customer_id": [f"CUST-{i % 300:05d}" for i in range(n)],  # foreign key: repeats
                "ciudad": [f"Ciudad {i % 150}" for i in range(n)],  # a genuine high-cardinality category
                "monto": [100.0 + i for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        summary = json.loads(Path(results["ventas"]["summary_json"]).read_text(encoding="utf-8"))
        assert summary["column_types"]["transaction_id"] == "identifier"
        assert summary["column_types"]["customer_id"] == "identifier"
        assert set(summary["data_quality"]["high_cardinality_columns"]) == {"ciudad"}
        flagged = [a["column"] for a in summary["alerts"] if "alta cardinalidad" in a["message"]]
        assert flagged == ["ciudad"]


class TestVerboseLogging:
    """Bug #17: --verbose (Config.verbose) must actually enable DEBUG-level logging."""

    def test_verbose_sets_debug_level(self, tmp_output_dir):
        pipeline = _make_pipeline(output_dir=str(tmp_output_dir), verbose=True)
        assert pipeline.logger.level == 10  # logging.DEBUG

    def test_non_verbose_sets_info_level(self, tmp_output_dir):
        pipeline = _make_pipeline(output_dir=str(tmp_output_dir), verbose=False)
        assert pipeline.logger.level == 20  # logging.INFO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
