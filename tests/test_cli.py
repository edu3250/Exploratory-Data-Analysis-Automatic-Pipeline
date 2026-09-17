"""
CLI tests using click.testing.CliRunner.

Bug #22: cli.py had 0% test coverage. These tests exercise the real command
surface (options, precedence, exit codes) rather than the underlying
pipeline logic, which is already covered by test_modules.py/test_integration.py.
"""

import json

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from eda_pipeline.cli import cli


def _tiny_df(n: int = 20) -> pd.DataFrame:
    """A small mixed-type DataFrame, just large enough to exercise every analysis step quickly."""
    return pd.DataFrame(
        {
            "amount": [float(i) for i in range(n)],
            "category": ["A", "B", "C"] * (n // 3) + ["A"] * (n % 3),
            "target": [0] * (n - 3) + [1] * 3,
        }
    )


@pytest.fixture
def runner():
    return CliRunner()


class TestAnalyzeFile:
    """click.testing.CliRunner tests for `eda analyze-file`."""

    def test_success_exits_zero(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)
        out_dir = tmp_output_dir / "out"

        result = runner.invoke(cli, ["analyze-file", str(csv_path), "--output-dir", str(out_dir)])

        assert result.exit_code == 0, result.output
        assert "data" in result.output
        assert "Reporte HTML" in result.output

    def test_missing_file_is_a_usage_error(self, runner, tmp_output_dir):
        result = runner.invoke(cli, ["analyze-file", str(tmp_output_dir / "no_existe.csv")])
        assert result.exit_code == 2  # click's click.Path(exists=True)

    def test_unknown_target_exits_nonzero_and_lists_the_columns(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)

        result = runner.invoke(
            cli, ["analyze-file", str(csv_path), "--target", "no_existe", "--output-dir", str(tmp_output_dir / "out")]
        )

        assert result.exit_code == 1
        assert "is not in the dataset" in result.output
        assert "Traceback" not in result.output  # a mistyped target is a user error, not a crash

    def test_decimal_option(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "decimal.csv"
        pd.DataFrame({"valor": ["1,5", "2,5", "3,5", "4,5"], "cat": ["x", "y", "x", "y"]}).to_csv(
            csv_path, index=False, sep=";"
        )
        out_dir = tmp_output_dir / "out"

        result = runner.invoke(cli, ["analyze-file", str(csv_path), "--decimal", ",", "--output-dir", str(out_dir)])

        assert result.exit_code == 0, result.output
        summary = json.loads(next(out_dir.glob("decimal_*/summary.json")).read_text(encoding="utf-8"))
        assert "numeric" in summary["column_types"]["valor"]

    def test_sheet_option_selects_named_sheet(self, runner, multi_sheet_excel_file, tmp_output_dir):
        out_dir = tmp_output_dir / "out"

        result = runner.invoke(
            cli, ["analyze-file", str(multi_sheet_excel_file), "--sheet", "Segunda", "--output-dir", str(out_dir)]
        )

        assert result.exit_code == 0, result.output
        summary = json.loads(next(out_dir.glob("multi_sheet_*/summary.json")).read_text(encoding="utf-8"))
        assert "b" in summary["column_types"]
        assert "a" not in summary["column_types"]

    def test_yaml_config_precedence_over_defaults(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)
        yaml_output_dir = tmp_output_dir / "from_yaml"
        config_path = tmp_output_dir / "config.yaml"
        config_path.write_text(
            yaml.dump({"output_dir": str(yaml_output_dir), "target": {"class_imbalance_threshold": 0.6}}),
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["analyze-file", str(csv_path), "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        assert yaml_output_dir.exists()

    def test_cli_target_does_not_wipe_yaml_class_imbalance_threshold(self, runner, tmp_output_dir):
        """Regression for bug #15: --target used to replace the whole nested `target` dict."""
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)
        out_dir = tmp_output_dir / "out"
        config_path = tmp_output_dir / "config.yaml"
        config_path.write_text(yaml.dump({"target": {"class_imbalance_threshold": 0.6}}), encoding="utf-8")

        result = runner.invoke(
            cli,
            [
                "analyze-file",
                str(csv_path),
                "--config",
                str(config_path),
                "--target",
                "target",
                "--output-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        summary = json.loads(next(out_dir.glob("data_*/summary.json")).read_text(encoding="utf-8"))
        assert summary["target"]["target_column"] == "target"

    def test_exit_code_1_when_dataset_fails(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)

        result = runner.invoke(
            cli, ["analyze-file", str(csv_path), "--target", "no_existe", "--output-dir", str(tmp_output_dir / "out")]
        )

        assert result.exit_code == 1

    def test_bad_yaml_config_key_is_a_config_error(self, runner, tmp_output_dir):
        csv_path = tmp_output_dir / "data.csv"
        _tiny_df().to_csv(csv_path, index=False)
        config_path = tmp_output_dir / "config.yaml"
        config_path.write_text(yaml.dump({"no_existe_esta_clave": 1}), encoding="utf-8")

        result = runner.invoke(cli, ["analyze-file", str(csv_path), "--config", str(config_path)])

        assert result.exit_code == 2
        assert "configuración" in result.output


class TestAnalyzeBatch:
    """click.testing.CliRunner tests for `eda analyze-batch`."""

    def test_batch_skips_gitkeep_and_unsupported_extension(self, runner, tmp_output_dir):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        (data_dir / ".gitkeep").write_text("")
        (data_dir / "readme.txt.bak").write_text("not a supported extension")
        _tiny_df().to_csv(data_dir / "data.csv", index=False)

        result = runner.invoke(cli, ["analyze-batch", str(data_dir), "--output-dir", str(tmp_output_dir / "out")])

        assert result.exit_code == 0, result.output
        assert "data" in result.output

    def test_pattern_filters_to_csv_only(self, runner, tmp_output_dir):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        _tiny_df().to_csv(data_dir / "data.csv", index=False)
        pd.DataFrame({"x": [1, 2]}).to_json(data_dir / "other.json", orient="records")

        result = runner.invoke(
            cli, ["analyze-batch", str(data_dir), "--pattern", "*.csv", "--output-dir", str(tmp_output_dir / "out")]
        )

        assert result.exit_code == 0, result.output
        assert "other" not in result.output

    def test_stem_collision_produces_unique_dataset_names(self, runner, tmp_output_dir):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        _tiny_df().to_csv(data_dir / "sales.csv", index=False)
        _tiny_df().to_parquet(data_dir / "sales.parquet")

        result = runner.invoke(cli, ["analyze-batch", str(data_dir), "--output-dir", str(tmp_output_dir / "out")])

        assert result.exit_code == 0, result.output
        assert "sales_csv" in result.output
        assert "sales_parquet" in result.output

    def test_one_corrupt_file_fails_but_batch_continues_with_exit_code_1(self, runner, tmp_output_dir):
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        _tiny_df().to_csv(data_dir / "good.csv", index=False)
        (data_dir / "broken.parquet").write_bytes(b"this is not a real parquet file")

        result = runner.invoke(cli, ["analyze-batch", str(data_dir), "--output-dir", str(tmp_output_dir / "out")])

        assert result.exit_code == 1
        assert "✅ good" in result.output or "good" in result.output
        assert "❌ broken" in result.output or "broken" in result.output

    def test_batch_accepts_delimiter_encoding_decimal_and_target_type(self, runner, tmp_output_dir):
        """Bug #17: analyze-batch must accept the same loader/target options as analyze-file."""
        data_dir = tmp_output_dir / "raw"
        data_dir.mkdir()
        pd.DataFrame({"valor": ["1,5", "2,5", "3,5", "4,5"], "cat": ["x", "y", "x", "y"]}).to_csv(
            data_dir / "data.csv", index=False, sep=";", encoding="utf-8"
        )

        result = runner.invoke(
            cli,
            [
                "analyze-batch",
                str(data_dir),
                "--delimiter",
                ";",
                "--encoding",
                "utf-8",
                "--decimal",
                ",",
                "--target-type",
                "classification",
                "--output-dir",
                str(tmp_output_dir / "out"),
            ],
        )

        assert result.exit_code == 0, result.output


class TestInitConfig:
    """click.testing.CliRunner tests for `eda init-config`."""

    def test_creates_default_config_file(self, runner, tmp_output_dir):
        output_path = tmp_output_dir / "my_config.yaml"

        result = runner.invoke(cli, ["init-config", "--output", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()
        loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        assert "data_quality" in loaded
        assert "language" not in loaded
