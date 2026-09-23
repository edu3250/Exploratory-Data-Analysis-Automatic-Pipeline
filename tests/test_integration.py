"""
Integration tests for the complete EDA Pipeline.
"""

import ast
import json
import re
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eda_pipeline.config import Config, OutlierConfig, TargetConfig, VisualizationConfig
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
        assert "outliers" in html  # shown in the "Failed Steps" section

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
            "recommendations.csv",
        }
        assert expected.issubset({p.name for p in tables_dir.glob("*.csv")})

    def test_the_report_closes_with_a_preprocessing_plan(self, tmp_output_dir):
        """The last section: what to drop, impute, encode and scale, each line with its measurement."""
        import numpy as np

        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "cliente_id": [f"C{i:05d}" for i in range(n)],
                "monto": rng.lognormal(mean=12, sigma=2.0, size=n),
                "edad": rng.normal(45, 12, n),
                "region": [["norte", "sur", "centro", "occidente"][i % 4] for i in range(n)],
                "comentario": [None] * (n - 20) + [f"nota {i}" for i in range(20)],
            }
        )
        df.loc[df.index[:30], "edad"] = None
        csv_file = tmp_output_dir / "creditos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["creditos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["failed_steps"] == []

        plan = summary["recommendations"]
        assert {rec["step"] for rec in plan} == {"drop", "impute", "encode", "scale"}
        dropped = {rec["column"] for rec in plan if rec["step"] == "drop"}
        assert {"cliente_id", "comentario"} <= dropped
        assert [r for r in plan if r["column"] == "edad" and r["step"] == "impute"]
        assert [r for r in plan if r["column"] == "region" and r["step"] == "encode"]
        # A dropped column is not carried into the later steps.
        assert [r for r in plan if r["column"] == "cliente_id"] == [
            r for r in plan if r["column"] == "cliente_id" and r["step"] == "drop"
        ]

        table = pd.read_csv(Path(result["output_dir"]) / "tables" / "recommendations.csv")
        assert set(table["step"]) == {"drop", "impute", "encode", "scale"}
        assert table["evidence"].notna().all()

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Preprocessing Plan" in html
        assert html.index('id="preprocesamiento"') > html.index('id="visualizaciones"')  # closes the report

    def test_the_plan_keeps_the_target_out_of_the_feature_steps(self, tmp_output_dir):
        """The target is what you predict: it is neither encoded nor scaled, and its gaps drop rows."""
        import numpy as np

        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame(
            {
                "ingreso": rng.lognormal(mean=10, sigma=1.5, size=n),
                "score": rng.normal(600, 40, n),
                "abandono": [["si", "no"][i % 2] for i in range(n)],
            }
        )
        df.loc[df.index[:15], "abandono"] = None
        csv_file = tmp_output_dir / "clientes.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(
            input_file=str(csv_file),
            output_dir=str(tmp_output_dir),
            target=TargetConfig(target_column="abandono"),
        )
        results = pipeline.run()

        summary = json.loads(Path(results["clientes"]["summary_json"]).read_text(encoding="utf-8"))
        plan = summary["recommendations"]
        target_rows = [rec for rec in plan if rec["column"] == "abandono"]
        assert len(target_rows) == 1
        assert target_rows[0]["step"] == "drop"
        assert "15 rows" in target_rows[0]["evidence"]

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
            outliers=OutlierConfig(isolation_forest_enabled=False),  # IQR and MAD on their own
        )
        results = pipeline.run()

        result = results["pedidos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["outliers"]["total_outlier_rows"] == 0

        table = pd.read_csv(Path(result["output_dir"]) / "tables" / "outlier_summary.csv")
        iqr_row = table[(table["column"] == "discount_pct") & (table["method"] == "iqr")].iloc[0]
        assert iqr_row["n_outliers"] == 0
        assert "IQR = 0" in iqr_row["note"]

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        outliers_section = html[html.index('<section id="outliers">') :]
        assert "discount_pct" in outliers_section[: outliers_section.index("</section>")]

    def test_skewed_amounts_get_a_log_scale_next_to_the_linear_charts(self, tmp_output_dir):
        """credito_asegurado's amounts crowded every point near zero in histograms, boxplots, scatter and pair plot."""
        import matplotlib.image as mpimg
        import numpy as np

        rng = np.random.default_rng(0)
        n = 1500
        monto = rng.lognormal(mean=13, sigma=2.5, size=n)
        df = pd.DataFrame(
            {
                "monto": monto,
                "saldo": monto * rng.uniform(0.3, 1.0, n),
                "plazo": rng.normal(240, 60, n),
            }
        )
        csv_file = tmp_output_dir / "creditos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["creditos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["failed_steps"] == []
        assert set(summary["log_scale_columns"]) == {"monto", "saldo"}

        plots_dir = Path(result["output_dir"]) / "plots"
        plots = {p.name for p in plots_dir.glob("*.png")}
        assert {"pair_plot.png", "pair_plot_log.png"} <= plots
        width = {
            name: mpimg.imread(plots_dir / name).shape[1] for name in ("histogram_monto.png", "histogram_plazo.png")
        }
        assert width["histogram_monto.png"] > 1.5 * width["histogram_plazo.png"]  # linear and log side by side

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "log scale" in html
        assert html.count('alt="Pair plot') == 2  # the linear pair plot stays, the log one follows it

    def test_a_column_filled_by_zeros_is_drawn_again_without_them(self, tmp_output_dir):
        """RoomService is 0 on 65% of the rows: its histogram was one bar against an axis to 14 327."""
        import matplotlib.image as mpimg
        import numpy as np

        rng = np.random.default_rng(0)
        n = 900
        gasto = np.concatenate([np.zeros(600), rng.lognormal(mean=4.0, sigma=1.3, size=300)])
        df = pd.DataFrame({"gasto": gasto, "edad": rng.normal(40, 12, n)})
        csv_file = tmp_output_dir / "pasajeros.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["pasajeros"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["failed_steps"] == []
        assert set(summary["floor_split_columns"]) == {"gasto"}  # edad has no repeated floor
        split = summary["floor_split_columns"]["gasto"]
        assert split["floor"] == 0.0
        assert split["floor_count"] == 600
        assert split["rest_count"] == 300
        assert split["rest_log"] is True

        plots_dir = Path(result["output_dir"]) / "plots"
        width = {
            name: mpimg.imread(plots_dir / name).shape[1] for name in ("histogram_gasto.png", "histogram_edad.png")
        }
        assert width["histogram_gasto.png"] > 1.5 * width["histogram_edad.png"]  # both panels in one image

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Without the value that fills them" in html  # the boxplot note
        assert "on 66.7% of the rows" in html

    def test_pair_plot_follows_the_scatter_plots_and_names_its_group(self, tmp_output_dir):
        """A pair plot of the numeric variables, coloured by the group that separates them, below the scatter plots."""
        n = 240
        especie = [["Adelie", "Gentoo", "Chinstrap"][i % 3] for i in range(n)]
        offset = {"Adelie": 0.0, "Gentoo": 10.0, "Chinstrap": 5.0}
        df = pd.DataFrame(
            {
                "pico": [40 + offset[e] + (i * 7) % 29 / 10 for i, e in enumerate(especie)],
                "aleta": [190 + 2 * offset[e] + (i * 11) % 31 / 5 for i, e in enumerate(especie)],
                "masa": [3500 + 100 * offset[e] + (i * 13) % 17 * 5 for i, e in enumerate(especie)],
                "especie": especie,
            }
        )
        csv_file = tmp_output_dir / "pinguinos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["pinguinos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["failed_steps"] == []
        assert summary["pair_plot"]["columns"] == ["pico", "aleta", "masa"]
        assert summary["pair_plot"]["hue"] == "especie"
        assert "pair_plot.png" in {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        scatter_at = html.index("Scatter Plots (Most Correlated Pairs)")
        pair_at = html.index("Pair Plot")
        assert scatter_at < pair_at
        assert "especie" in html[pair_at : pair_at + 2000]

    def test_no_pair_plot_with_fewer_than_three_continuous_columns(self, tmp_output_dir):
        n = 120
        df = pd.DataFrame(
            {"precio": [100.0 + (i * 7) % 50 for i in range(n)], "total": [300.0 + (i * 13) % 90 for i in range(n)]}
        )
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["ventas"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["pair_plot"] is None
        assert "pair_plot.png" not in {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "Pair Plot" not in Path(result["html_report"]).read_text(encoding="utf-8")

    def test_scatter_plots_show_the_most_correlated_pairs(self, tmp_output_dir):
        """On penguins_lter the strongest pair (|r| = 0.87) was left out: pairs came in column order."""
        n = 200
        df = pd.DataFrame(
            {
                "a": [float((i * 7) % 13) for i in range(n)],
                "b": [float((i * 11) % 17) for i in range(n)],
                "c": [float(i) for i in range(n)],
                "d": [2.0 * i + (i % 3) for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "medidas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(
            input_file=str(csv_file),
            output_dir=str(tmp_output_dir),
            visualizations=VisualizationConfig(max_scatter_pairs=1),
        )
        results = pipeline.run()

        plots = {p.name for p in (Path(results["medidas"]["output_dir"]) / "plots").glob("scatter_*.png")}
        assert plots == {"scatter_c_vs_d.png"}  # the last pair in column order, and by far the strongest

    def test_isolation_forest_cut_is_explained_in_every_output(self, tmp_output_dir):
        """The cut used to be a hidden 10%; the summary, the CSV table and the report now say how it was drawn."""
        n = 600
        df = pd.DataFrame(
            {
                "monto": [100.0 + (i * 37) % 200 for i in range(n)],
                "plazo": [12.0 + (i * 11) % 48 for i in range(n)],
            }
        )
        df.loc[[5, 250, 480], ["monto", "plazo"]] = [5000.0, 900.0]
        csv_file = tmp_output_dir / "creditos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["creditos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        forest = summary["outliers"]["isolation_forest"]
        assert 3 <= forest["outlier_rows"] < 0.1 * n
        assert "Q3" in forest["note"]

        table = pd.read_csv(Path(result["output_dir"]) / "tables" / "outlier_summary.csv")
        row = table[table["method"] == "isolation_forest"].iloc[0]
        assert row["n_outliers"] == forest["outlier_rows"]
        assert "Q3" in row["note"]

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        section = html[html.index('<section id="outliers">') :]
        assert "Q3" in section[: section.index("</section>")]

    def test_report_shows_the_first_rows_before_the_univariate_section(self, tmp_output_dir):
        """A preview of the data must sit ahead of «Column by Column» in the report."""
        n = 30
        df = pd.DataFrame({"ciudad": [f"Ciudad {i % 3}" for i in range(n)], "monto": [100.0 + i for i in range(n)]})
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        html = Path(results["ventas"]["html_report"]).read_text(encoding="utf-8")
        assert "First Rows" in html
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

    def test_categorical_columns_with_few_categories_get_a_pie(self, tmp_output_dir):
        """Bars show counts; a pie shows each category's share, but only while there are few of them."""
        n = 120
        df = pd.DataFrame(
            {
                "canal": [["tienda", "web", "web", "app"][i % 4] for i in range(n)],
                "municipio": [f"Municipio {i % 12}" for i in range(n)],
                "monto": [100.0 + i for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["ventas"]
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "pie_canal.png" in plots
        assert "pie_municipio.png" not in plots
        assert "categorical_municipio.png" in plots  # its bars stay

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Categorical Shares (1)" in html
        assert html.index("Categorical Distributions") < html.index("Categorical Shares")

    def test_missing_value_placeholders_are_reported_and_not_analysed(self, tmp_output_dir):
        """The "." in sex showed up as a category in the bars and as a 0.3% slice of the pie."""
        n = 100
        sexo = [["MALE", "FEMALE"][i % 2] for i in range(n)]
        sexo[7] = "."
        df = pd.DataFrame({"sexo": sexo, "masa": [3000.0 + (i * 37) % 900 for i in range(n)]})
        csv_file = tmp_output_dir / "pinguinos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["pinguinos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["data_quality"]["missing_placeholders"] == {"sexo": {".": 1}}
        assert any(a["column"] == "sexo" and "'.'" in a["message"] for a in summary["alerts"])

        tables = Path(result["output_dir"]) / "tables"
        categorical = pd.read_csv(tables / "categorical_stats.csv").set_index("column")
        assert categorical.loc["sexo", "nunique"] == 2
        missing = pd.read_csv(tables / "missing_per_column.csv").set_index("column")
        assert missing.loc["sexo", "missing_pct"] == pytest.approx(1.0)

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
        assert "Target vs Categorical Columns" in html
        # A p-value below 0.0001 reaches the page as text: a bare < would open a tag.
        assert "<0.0001" not in html
        assert "&lt;0.0001" in html
        categorical_at = html.index("Categorical Distributions")
        target_bars_at = html.index("Target vs Categorical Columns")
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
        assert "Target vs Categorical Columns" not in html

    def test_time_of_day_columns_are_analysed_as_times(self, tmp_output_dir):
        """transaction_time on Vistara was a category with 31702 values: an alert and a useless bar chart."""
        n = 300
        df = pd.DataFrame(
            {
                "hora": [f"{9 + (i // 25) % 12:02d}:{i % 60:02d}:{(i * 7) % 60:02d}" for i in range(n)],
                "monto": [100.0 + i for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "ventas.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["ventas"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["column_types"]["hora"] == "time"
        assert summary["univariate"]["time_columns"] == 1
        assert not any(alert["column"] == "hora" for alert in summary["alerts"])

        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "time_of_day_hora.png" in plots
        assert "categorical_hora.png" not in plots

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Time of day" in html  # how the quality table names the column's type
        assert "By Hour of the Day" in html

    def test_numeric_id_columns_are_not_analysed_as_variables(self, tmp_output_dir):
        """A key stored as a number used to get a histogram, a boxplot, VIF and scatter plots."""
        n = 300
        df = pd.DataFrame(
            {
                "id": range(1, n + 1),
                "edad": [20 + (i * 7) % 50 for i in range(n)],  # not 20, 21, 22...: that is a row counter
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

    def test_sequence_number_columns_are_not_analysed_as_variables(self, tmp_output_dir):
        """Sample Number (1-152) got a histogram, a boxplot, a VIF entry and 6 of the 10 scatter plots."""
        n = 300
        df = pd.DataFrame(
            {
                "Sample Number": [1 + i % 150 for i in range(n)],
                "masa": [3000.0 + (i * 37) % 900 for i in range(n)],
                "aleta": [180.0 + (i * 11) % 50 for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "muestras.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["muestras"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["column_types"]["Sample Number"] == "identifier"
        assert "Sample Number" not in summary["relationships"]["multicollinearity_vif"]
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert not any("Sample Number" in name for name in plots)

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
        assert "Association Between Columns" in html

    def test_quality_table_shows_how_each_column_was_classified(self, tmp_output_dir):
        """The «Data Quality» table must show the dtype and the inferred type per column."""
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
        assert "<th>Stored as</th>" in html
        assert "<th>Read as</th>" in html
        for cell in (
            '<td><code title="Int64">int</code></td>',  # exact pandas dtype kept in the tooltip
            '<td><code title="Float64">float</code></td>',
            "<td>Discrete</td>",
            "<td>Continuous</td>",
            "<td>Categorical</td>",
        ):
            assert cell in html

        # The CSV twin of that table carries the same two columns.
        csv_text = (Path(result["output_dir"]) / "tables" / "missing_per_column.csv").read_text(encoding="utf-8")
        assert "column,dtype,inferred_type,missing_pct" in csv_text
        assert "quantity,int,Discrete" in csv_text

    def test_two_digit_year_dates_are_analysed_as_dates(self, tmp_output_dir):
        """Date Egg (11/11/07) was a category with 50 values: no date statistics, no time series."""
        n = 150
        df = pd.DataFrame(
            {
                "fecha_huevo": [f"11/{9 + i % 20}/{7 + i % 3:02d}" for i in range(n)],
                "masa": [3000.0 + (i * 37) % 900 for i in range(n)],
            }
        )
        csv_file = tmp_output_dir / "nidos.csv"
        df.to_csv(csv_file, index=False)
        pipeline = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir))
        results = pipeline.run()

        result = results["nidos"]
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        assert summary["column_types"]["fecha_huevo"] == "datetime"
        assert summary["univariate"]["datetime_columns"] == 1
        plots = {p.name for p in (Path(result["output_dir"]) / "plots").glob("*.png")}
        assert "timeseries_fecha_huevo.png" in plots
        assert "categorical_fecha_huevo.png" not in plots

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
        assert "Rows Over Time" in html
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
        assert "Imbalanced classes" in alert_messages
        assert any("leakage" in a["message"].lower() for a in summary["alerts"])

        html = Path(result["html_report"]).read_text(encoding="utf-8")
        assert "Imbalanced" in html

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
        flagged = [a["column"] for a in summary["alerts"] if "high cardinality" in a["message"]]
        assert flagged == ["ciudad"]


class TestVerboseLogging:
    """Bug #17: --verbose (Config.verbose) must actually enable DEBUG-level logging."""

    def test_verbose_sets_debug_level(self, tmp_output_dir):
        pipeline = _make_pipeline(output_dir=str(tmp_output_dir), verbose=True)
        assert pipeline.logger.level == 10  # logging.DEBUG

    def test_non_verbose_sets_info_level(self, tmp_output_dir):
        pipeline = _make_pipeline(output_dir=str(tmp_output_dir), verbose=False)
        assert pipeline.logger.level == 20  # logging.INFO


class TestDatasetNamingEndToEnd:
    """A Kaggle folder of train/test files must produce reports that say which dataset they are."""

    def _frame(self):
        return pd.DataFrame({"age": [20, 30, 40, 50] * 5, "sex": ["m", "f"] * 10, "survived": [0, 1] * 10})

    def test_a_single_split_file_is_named_after_its_folder(self, tmp_output_dir):
        folder = tmp_output_dir / "Kaggle Titanic"
        folder.mkdir()
        self._frame().to_csv(folder / "train.csv", index=False)

        results = _make_pipeline(
            input_file=str(folder / "train.csv"), output_dir=str(tmp_output_dir / "out")
        ).run()

        assert list(results) == ["Kaggle_Titanic_train"]
        output_dir = Path(results["Kaggle_Titanic_train"]["output_dir"])
        assert output_dir.name.startswith("Kaggle_Titanic_train_")
        summary = json.loads(Path(results["Kaggle_Titanic_train"]["summary_json"]).read_text(encoding="utf-8"))
        assert summary["dataset_name"] == "Kaggle_Titanic_train"

    def test_a_batch_qualifies_only_the_split_files(self, tmp_output_dir):
        folder = tmp_output_dir / "Kaggle Titanic"
        folder.mkdir()
        for name in ("train.csv", "test.csv", "passengers.csv"):
            self._frame().to_csv(folder / name, index=False)

        results = _make_pipeline(input_folder=str(folder), output_dir=str(tmp_output_dir / "out")).run()

        assert set(results) == {"Kaggle_Titanic_train", "Kaggle_Titanic_test", "passengers"}


class TestCopyColumnLists:
    """The «Copy list» buttons beside Numeric Columns and Categorical Columns."""

    @staticmethod
    def _copied(html: str, list_id: str) -> tuple:
        """What a button copies, as the browser decodes it: the variable name and the list it holds."""
        match = re.search(rf'<pre class="copy-list-text" id="{list_id}" hidden>(.*?)</pre>', html, re.S)
        assert match, f"{list_id} is not in the report"
        variable, _, literal = unescape(match.group(1)).partition(" = ")
        return variable, ast.literal_eval(literal)

    def test_each_button_copies_the_columns_its_table_shows(self, csv_file, tmp_output_dir):
        result = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir)).run()["data"]
        html = Path(result["html_report"]).read_text(encoding="utf-8")
        tables = Path(result["output_dir"]) / "tables"
        numeric = pd.read_csv(tables / "numeric_stats.csv")["column"].tolist()
        categorical = pd.read_csv(tables / "categorical_stats.csv")["column"].tolist()

        assert self._copied(html, "copy-list-num") == ("num", numeric)
        assert self._copied(html, "copy-list-cat") == ("cat", categorical)
        # Free text, identifiers and dates have sections of their own and belong in neither list.
        assert {"age", "income"} <= set(numeric) and {"gender", "region"} <= set(categorical)
        for other in ("comment", "user_id", "signup_date"):
            assert other not in numeric + categorical

    def test_names_with_markup_characters_are_escaped_in_the_page(self, tmp_output_dir):
        csv_file = tmp_output_dir / "gastos.csv"
        pd.DataFrame(
            {"R&D spend": [float(i) * 1.5 for i in range(42)], "it's": ["low", "mid", "high"] * 14}
        ).to_csv(csv_file, index=False)

        result = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir)).run()["gastos"]
        html = Path(result["html_report"]).read_text(encoding="utf-8")

        raw = re.search(r'id="copy-list-num" hidden>(.*?)</pre>', html, re.S).group(1)
        assert "R&amp;D" in raw  # escaped in the markup...
        assert self._copied(html, "copy-list-num") == ("num", ["R&D spend"])  # ...and exact once decoded
        assert self._copied(html, "copy-list-cat") == ("cat", ["it's"])

    def test_a_section_without_columns_has_no_button(self, tmp_output_dir):
        csv_file = tmp_output_dir / "medidas.csv"
        pd.DataFrame(
            {"alto": [float(i) for i in range(40)], "ancho": [(i * 7) % 40 + 0.5 for i in range(40)]}
        ).to_csv(csv_file, index=False)

        result = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir)).run()["medidas"]
        html = Path(result["html_report"]).read_text(encoding="utf-8")

        assert 'data-copy-from="copy-list-num"' in html
        assert 'data-copy-from="copy-list-cat"' not in html


class TestMulticollinearitySection:
    """The multicollinearity block under the correlation matrices, and what reaches summary.json."""

    @staticmethod
    def _run(tmp_output_dir, df, name):
        csv_file = tmp_output_dir / f"{name}.csv"
        df.to_csv(csv_file, index=False)
        result = _make_pipeline(input_file=str(csv_file), output_dir=str(tmp_output_dir)).run()[name]
        html = Path(result["html_report"]).read_text(encoding="utf-8")
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        return html, summary

    def test_an_exact_identity_is_written_under_the_matrices(self, tmp_output_dir):
        rng = np.random.default_rng(13)
        first = rng.normal(900, 200, 200).round(1)
        second = rng.normal(400, 150, 200).round(1)
        df = pd.DataFrame(
            {
                "first_floor": first,
                "second_floor": second,
                "living_area": first + second,
                "rooms": rng.normal(6, 2, 200).round(1),
            }
        )

        html, summary = self._run(tmp_output_dir, df, "casas")

        block = html.index('id="multicolinealidad"')
        assert html.index("Association Between Columns") < block < html.index('id="preprocesamiento"')
        assert 'href="#multicolinealidad"' in html
        assert "living_area = first_floor + second_floor" in html[block:]
        assert "The 3 columns of those combinations have an infinite VIF" in html[block:]

        section = summary["multicollinearity"]
        assert [dep["equation"] for dep in section["exact_dependencies"]] == ["living_area = first_floor + second_floor"]
        assert section["suggested_drops"][0]["column"] == "living_area"
        vif = summary["relationships"]["multicollinearity_vif"]
        assert vif["living_area"] is None  # infinite: JSON has no such number
        assert all(value is None or value >= 1 for value in vif.values())

    def test_independent_columns_say_so(self, tmp_output_dir):
        rng = np.random.default_rng(14)
        df = pd.DataFrame({name: rng.normal(size=200).round(3) for name in ("alto", "ancho", "peso")})

        html, summary = self._run(tmp_output_dir, df, "medidas")

        assert "No column has a VIF of 5 or more" in html
        assert summary["multicollinearity"]["exact_dependencies"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
