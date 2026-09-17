"""
Command-line interface for EDA Pipeline.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional

import click

from .config import Config, ConfigError, create_default_config_file, load_config_file, merge_configs
from .pipeline import EDAPipeline


def _make_streams_robust(streams=None) -> None:
    """
    Ensure stdout/stderr never crash on characters the terminal's encoding
    cannot represent (e.g. the ✅/❌ markers on a cp1252 Windows console).
    Unencodable characters are replaced instead of raising UnicodeEncodeError.
    """
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


_make_streams_robust()


def _parse_sheet_option(value: Optional[str]) -> Optional[str | int]:
    """A digits-only --sheet value means a sheet index; anything else is a sheet name."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _build_target_overrides(target: Optional[str], target_type: Optional[str]) -> dict:
    """Build the (possibly empty) `target` section of the CLI overrides dict."""
    overrides = {}
    if target:
        overrides["target_column"] = target
    if target_type:
        overrides["target_type"] = target_type
    return {"target": overrides} if overrides else {}


def _run_and_report(final_config: Config, verbose: bool) -> None:
    """Run the pipeline, print per-dataset results, and exit with the appropriate code."""
    pipeline = EDAPipeline(final_config)
    pipeline.setup()
    results = pipeline.run()

    if pipeline.batch_output_dir is not None:
        click.secho(f"📁 Batch folder: {pipeline.batch_output_dir}", fg="cyan")

    if "error" in results and len(results) == 1:
        click.secho(f"Error: {results['error']}", fg="red")
        sys.exit(1)

    any_failed = False
    for dataset_name, result in results.items():
        if "error" in result:
            click.secho(f"❌ {dataset_name}: {result['error']}", fg="red")
            any_failed = True
            continue

        if result.get("failed_steps"):
            click.secho(f"⚠️  {dataset_name}: finished, but some steps failed", fg="yellow")
            click.echo(f"   Failed steps: {', '.join(result['failed_steps'])}")
            any_failed = True
        else:
            click.secho(f"✅ {dataset_name}", fg="green")

        click.echo(f"   HTML report: {result.get('html_report') or '(not generated)'}")
        click.echo(f"   JSON summary: {result['summary_json']}")
        click.echo(f"   Alerts: {result['n_alerts']} | Charts: {result['n_plots']}")

    if not results:
        click.secho("No files found to process.", fg="yellow")

    if any_failed:
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Exploratory Data Analysis (EDA) - the complete pipeline."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True), help="Path to a YAML configuration file")
@click.option("--output-dir", default=None, help="Where to write the reports (default: reports)")
@click.option("--target", help="Target column to analyze")
@click.option(
    "--target-type",
    type=click.Choice(["classification", "regression"]),
    help="Target type (auto-detected if not given)",
)
@click.option("--delimiter", help="CSV delimiter (auto-detected if not given)")
@click.option("--encoding", help="File encoding (auto-detected if not given)")
@click.option("--decimal", type=click.Choice([".", ","]), help="Decimal separator (auto-detected if not given)")
@click.option("--sheet", help="Excel sheet: a name, or an index if all digits (default: the first)")
@click.option("--sample-size", type=int, help="Sample size for large datasets")
@click.option("--strict", is_flag=True, default=False, help="Strict mode: stop at the first error")
@click.option("--verbose", is_flag=True, default=False, help="Detailed logging (DEBUG level)")
def analyze_file(
    input_file,
    config,
    output_dir,
    target,
    target_type,
    delimiter,
    encoding,
    decimal,
    sheet,
    sample_size,
    strict,
    verbose,
):
    """
    Analyze a single data file.

    \b
    EXAMPLES:
        eda analyze-file data.csv
        eda analyze-file data.csv --target target_col --output-dir ./my_reports
        eda analyze-file data.xlsx --sheet 1 --target category
    """
    try:
        base_config = Config()
        file_config = load_config_file(Path(config)) if config else {}

        cli_overrides = {
            "input_file": input_file,
            "output_dir": output_dir,
            "delimiter": delimiter,
            "encoding": encoding,
            "decimal": decimal,
            "excel_sheet": _parse_sheet_option(sheet),
            "sample_size": sample_size,
            **_build_target_overrides(target, target_type),
        }
        if strict:
            cli_overrides["strict_mode"] = True
        if verbose:
            cli_overrides["verbose"] = True

        final_config = merge_configs(base_config, file_config, cli_overrides)
        _run_and_report(final_config, verbose)

    except ConfigError as e:
        click.secho(f"Configuration error: {e}", fg="red")
        sys.exit(2)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--pattern", default=None, help="Glob pattern for the files (default: every supported extension)"
)
@click.option("--config", type=click.Path(exists=True), help="Path to a YAML configuration file")
@click.option("--output-dir", default=None, help="Where to write the reports (default: reports)")
@click.option("--target", help="Target column to analyze")
@click.option(
    "--target-type",
    type=click.Choice(["classification", "regression"]),
    help="Target type (auto-detected if not given)",
)
@click.option("--delimiter", help="CSV delimiter (auto-detected if not given)")
@click.option("--encoding", help="File encoding (auto-detected if not given)")
@click.option("--decimal", type=click.Choice([".", ","]), help="Decimal separator (auto-detected if not given)")
@click.option("--sheet", help="Excel sheet: a name, or an index if all digits (default: the first)")
@click.option("--sample-size", type=int, help="Sample size for large datasets")
@click.option("--strict", is_flag=True, default=False, help="Strict mode: stop at the first error")
@click.option("--verbose", is_flag=True, default=False, help="Detailed logging (DEBUG level)")
def analyze_batch(
    input_folder,
    pattern,
    config,
    output_dir,
    target,
    target_type,
    delimiter,
    encoding,
    decimal,
    sheet,
    sample_size,
    strict,
    verbose,
):
    """
    Analyze every supported file in a folder.

    Hidden files and unsupported extensions are skipped (with a warning);
    a file that fails is reported with ❌ and the rest of the batch carries
    on, unless --strict is used.

    \b
    EXAMPLES:
        eda analyze-batch ./data
        eda analyze-batch ./data --pattern "*.csv" --target target_col
        eda analyze-batch ./raw_data --output-dir ./my_reports
    """
    try:
        base_config = Config()
        file_config = load_config_file(Path(config)) if config else {}

        cli_overrides = {
            "input_folder": input_folder,
            "batch_pattern": pattern,
            "output_dir": output_dir,
            "delimiter": delimiter,
            "encoding": encoding,
            "decimal": decimal,
            "excel_sheet": _parse_sheet_option(sheet),
            "sample_size": sample_size,
            **_build_target_overrides(target, target_type),
        }
        if strict:
            cli_overrides["strict_mode"] = True
        if verbose:
            cli_overrides["verbose"] = True

        final_config = merge_configs(base_config, file_config, cli_overrides)
        _run_and_report(final_config, verbose)

    except ConfigError as e:
        click.secho(f"Configuration error: {e}", fg="red")
        sys.exit(2)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--output", default="config/default.yaml", help="Path of the configuration file to create")
def init_config(output):
    """
    Create a template configuration file.

    \b
    EXAMPLES:
        eda init-config
        eda init-config --output my_config.yaml
    """
    try:
        output_path = Path(output)
        create_default_config_file(output_path)
        click.secho(f"✅ Configuration created: {output_path}", fg="green")
        click.echo(f"Edit it, then use it with: eda analyze-file <file> --config {output}")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
