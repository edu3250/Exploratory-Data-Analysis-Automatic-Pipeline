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
            click.secho(f"⚠️  {dataset_name}: completado con errores parciales", fg="yellow")
            click.echo(f"   Pasos con error: {', '.join(result['failed_steps'])}")
            any_failed = True
        else:
            click.secho(f"✅ {dataset_name}", fg="green")

        click.echo(f"   Reporte HTML: {result.get('html_report') or '(no generado)'}")
        click.echo(f"   Resumen JSON: {result['summary_json']}")
        click.echo(f"   Alertas: {result['n_alerts']} | Gráficos: {result['n_plots']}")

    if not results:
        click.secho("No se encontraron archivos para procesar.", fg="yellow")

    if any_failed:
        sys.exit(1)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Análisis Exploratorio de Datos (EDA) - Pipeline Completo"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--config", type=click.Path(exists=True), help="Ruta a archivo YAML de configuración")
@click.option("--output-dir", default=None, help="Directorio de salida para reportes (por defecto: reports)")
@click.option("--target", help="Columna target para análisis")
@click.option(
    "--target-type",
    type=click.Choice(["classification", "regression"]),
    help="Tipo de target (auto-detectar si no se especifica)",
)
@click.option("--delimiter", help="Delimitador de CSV (auto-detectar si no se especifica)")
@click.option("--encoding", help="Codificación de archivo (auto-detectar si no se especifica)")
@click.option("--decimal", type=click.Choice([".", ","]), help="Separador decimal (auto-detectar si no se especifica)")
@click.option("--sheet", help="Hoja de Excel: nombre, o índice si es solo dígitos (por defecto: la primera)")
@click.option("--sample-size", type=int, help="Tamaño de muestra para datasets grandes")
@click.option("--strict", is_flag=True, default=False, help="Modo estricto: fallar en el primer error")
@click.option("--verbose", is_flag=True, default=False, help="Logging detallado (nivel DEBUG)")
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
    Analizar un archivo de datos individual.

    EJEMPLO:
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
        click.secho(f"Error de configuración: {e}", fg="red")
        sys.exit(2)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--pattern", default=None, help="Patrón glob para archivos (por defecto: todas las extensiones soportadas)"
)
@click.option("--config", type=click.Path(exists=True), help="Ruta a archivo YAML de configuración")
@click.option("--output-dir", default=None, help="Directorio de salida para reportes (por defecto: reports)")
@click.option("--target", help="Columna target para análisis")
@click.option(
    "--target-type",
    type=click.Choice(["classification", "regression"]),
    help="Tipo de target (auto-detectar si no se especifica)",
)
@click.option("--delimiter", help="Delimitador de CSV (auto-detectar si no se especifica)")
@click.option("--encoding", help="Codificación de archivo (auto-detectar si no se especifica)")
@click.option("--decimal", type=click.Choice([".", ","]), help="Separador decimal (auto-detectar si no se especifica)")
@click.option("--sheet", help="Hoja de Excel: nombre, o índice si es solo dígitos (por defecto: la primera)")
@click.option("--sample-size", type=int, help="Tamaño de muestra para datasets grandes")
@click.option("--strict", is_flag=True, default=False, help="Modo estricto: fallar en el primer error")
@click.option("--verbose", is_flag=True, default=False, help="Logging detallado (nivel DEBUG)")
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
    Analizar múltiples archivos en un folder.

    Los archivos ocultos y con extensión no soportada se omiten (con aviso);
    un archivo que falla se reporta con ❌ y el resto del lote continúa,
    salvo que se use --strict.

    EJEMPLO:
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
        click.secho(f"Error de configuración: {e}", fg="red")
        sys.exit(2)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--output", default="config/default.yaml", help="Ruta del archivo de configuración a crear")
def init_config(output):
    """
    Crear un archivo de configuración de plantilla.

    EJEMPLO:
        eda init-config
        eda init-config --output my_config.yaml
    """
    try:
        output_path = Path(output)
        create_default_config_file(output_path)
        click.secho(f"✅ Configuración creada: {output_path}", fg="green")
        click.echo(f"Edite este archivo y úselo con: eda analyze-file <archivo> --config {output}")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
