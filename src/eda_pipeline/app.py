"""
The local app: ``eda app`` serves a page on this computer to add files, pick a target and run the pipeline,
then opens the report it writes.

It is a small standard-library HTTP server bound to 127.0.0.1, so nothing leaves the machine and nothing new is
installed. One analysis runs at a time (matplotlib draws through global state), in a background thread that
reports each step the pipeline starts (EDAPipeline.on_progress), which the page polls.

Requests from other sites are refused: the server only answers a Host that names itself (no DNS rebinding),
and every request that changes something must carry the ``X-EDA-App`` header, which a page on another origin
cannot add without a CORS preflight this server never approves.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pandas as pd

from .app_page import APP_PAGE
from .config import Config, ConfigError, merge_configs
from .data_loader import SUPPORTED_EXTENSIONS, detect_delimiter, detect_encoding, detect_file_format, load_data
from .pipeline import EDAPipeline

logger = logging.getLogger(__name__)

APP_HEADER = "X-EDA-App"
# A file larger than this is better analysed from the command line, with --sample-size.
MAX_UPLOAD_BYTES = 2 * 1024**3
RECENT_REPORTS = 12
_UNSAFE_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# The files the server hands out from the reports folder: the reports and what they link to.
_SERVED_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

# What the page lists while a dataset runs, in order. Several pipeline steps can share a stage.
STAGES = [
    ("load", "Reading the file"),
    ("types", "Inferring column types"),
    ("data_quality", "Data quality"),
    ("univariate", "Column by column"),
    ("relationships", "Relationships"),
    ("multicollinearity", "Multicollinearity"),
    ("outliers", "Outliers"),
    ("target", "Target"),
    ("transforms", "Skewed columns"),
    ("chart_plan", "Choosing the charts"),
    ("recommendations", "Preprocessing plan"),
    ("visualizations", "Charts"),
    ("tables", "CSV tables"),
    ("html_report", "HTML report"),
]
STEP_STAGE = {
    "log_scale": "transforms",
    "floor_split": "transforms",
    "chart_columns": "chart_plan",
    "association_matrix": "chart_plan",
    "pair_plot": "chart_plan",
}


def stage_of(step: str) -> str:
    """The stage the page shows for a pipeline step (unknown steps show as their own name)."""
    return STEP_STAGE.get(step, step)


def safe_file_name(name: str) -> str:
    """A file name the uploads folder can hold: its last component, without the characters a system refuses."""
    base = _UNSAFE_IN_FILENAME.sub("_", Path(name.replace("\\", "/")).name).strip(" .")
    return base or "file"


def read_columns(path: Path) -> list[str] | None:
    """
    The column names of a data file, read from its header where the format allows it, so the page can offer
    them as targets. None when they cannot be read cheaply; the analysis reads the file in full anyway.
    """
    try:
        fmt = detect_file_format(path)
        if fmt == "csv":
            encoding = detect_encoding(path)
            frame = pd.read_csv(path, sep=detect_delimiter(path, encoding), encoding=encoding, nrows=0)
        elif fmt == "excel":
            frame = pd.read_excel(path, nrows=0)
        elif fmt == "parquet":
            import pyarrow.parquet as pq

            return [str(name) for name in pq.read_schema(path).names]
        else:
            frame = load_data(path)
        return [str(col) for col in frame.columns]
    except Exception as e:  # the page falls back to "read when the analysis runs"
        logger.info(f"Could not read the columns of {path.name}: {e}")
        return None


def recent_reports(output_dir: Path, limit: int = RECENT_REPORTS) -> list[dict]:
    """The latest reports in the output folder, newest first, with what the page shows about each."""
    found = []
    for summary_file in list(output_dir.glob("*/summary.json")) + list(output_dir.glob("*/*/summary.json")):
        report = summary_file.parent / "report.html"
        if not report.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        shape = summary.get("shape") or [None, None]
        target = summary.get("target") or {}
        found.append(
            {
                "name": summary.get("dataset_name") or summary_file.parent.name,
                "url": "/reports/" + quote(report.relative_to(output_dir).as_posix()),
                "rows": shape[0],
                "columns": shape[1],
                "alerts": len(summary.get("alerts") or []),
                "target": target.get("target_column"),
                "generated": (summary.get("timestamp") or "")[:16].replace("T", " "),
                "failed_steps": len(summary.get("failed_steps") or []),
                "mtime": report.stat().st_mtime,
            }
        )
    found.sort(key=lambda item: item["mtime"], reverse=True)
    for item in found:
        del item["mtime"]
    return found[:limit]


@dataclass
class Upload:
    id: str
    name: str
    path: Path
    size: int
    columns: list[str] | None


@dataclass
class Run:
    id: str
    status: str = "running"  # running, done, failed
    dataset: str = ""
    stage: str = ""
    datasets_done: int = 0
    datasets_total: int = 1
    results: list[dict] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "dataset": self.dataset,
            "stage": self.stage,
            "datasets_done": self.datasets_done,
            "datasets_total": self.datasets_total,
            "results": self.results,
            "error": self.error,
        }


class RequestError(Exception):
    """A request the app refuses, with the status and the message the page shows."""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class EdaApp:
    """The app's state: uploaded files, runs, and the one worker allowed to run the pipeline."""

    def __init__(self, output_dir: Path, work_dir: Path | None = None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._own_work_dir = work_dir is None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="eda-app-"))
        self.uploads: dict[str, Upload] = {}
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._running: Run | None = None

    def close(self) -> None:
        """Remove the uploaded copies; the reports stay in the output folder."""
        if self._own_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    # ---- files
    def add_upload(self, name: str, stream, length: int) -> Upload:
        clean = safe_file_name(name)
        if Path(clean).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise RequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                f"{clean}: not a format the pipeline reads. Use one of {', '.join(SUPPORTED_EXTENSIONS)}.",
            )
        if length > MAX_UPLOAD_BYTES:
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"{clean} is larger than {MAX_UPLOAD_BYTES // 1024**3} GB. Analyze it from the command line, "
                "with --sample-size.",
            )
        upload_id = uuid.uuid4().hex[:12]
        folder = self.work_dir / "files" / upload_id
        folder.mkdir(parents=True)
        path = folder / clean
        remaining = length
        with open(path, "wb") as out:
            while remaining > 0:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        if remaining:
            shutil.rmtree(folder, ignore_errors=True)
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{clean} arrived incomplete. Add it again.")
        upload = Upload(upload_id, clean, path, length, read_columns(path))
        self.uploads[upload_id] = upload
        return upload

    def remove_upload(self, upload_id: str) -> None:
        upload = self.uploads.pop(upload_id, None)
        if upload is None:
            raise RequestError(HTTPStatus.NOT_FOUND, "That file is no longer here.")
        shutil.rmtree(upload.path.parent, ignore_errors=True)

    # ---- runs
    def start_run(self, request: dict, background: bool = True) -> Run:
        ids = request.get("files") or []
        uploads = [self.uploads.get(i) for i in ids]
        if not uploads or None in uploads:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Add a file to analyze first.")
        config = self._config_for(request, uploads)
        with self._lock:
            if self._running is not None:
                raise RequestError(HTTPStatus.CONFLICT, "An analysis is already running. Wait for it to finish.")
            run = Run(uuid.uuid4().hex[:12], datasets_total=len(uploads))
            self.runs[run.id] = run
            self._running = run
        if background:
            threading.Thread(target=self._execute, args=(run, config), daemon=True).start()
        else:
            self._execute(run, config)
        return run

    def _config_for(self, request: dict, uploads: list[Upload]) -> Config:
        if len(uploads) == 1:
            source = {"input_file": str(uploads[0].path)}
        else:
            # A batch reads a folder, so the chosen files are copied into one of their own. Its name
            # names the reports folder: uploads_batch_<timestamp>.
            folder = self.work_dir / "batches" / uuid.uuid4().hex[:12] / "uploads"
            folder.mkdir(parents=True)
            for upload in uploads:
                target = folder / upload.name
                stem, n = target.stem, 2
                while target.exists():
                    target = folder / f"{stem}_{n}{target.suffix}"
                    n += 1
                shutil.copyfile(upload.path, target)
            source = {"input_folder": str(folder)}
        sample = request.get("sample_size")
        target_overrides = {}
        if request.get("target"):
            target_overrides["target_column"] = request["target"]
        if request.get("target_type"):
            target_overrides["target_type"] = request["target_type"]
        try:
            sample_size = int(sample) if sample not in (None, "") else None
        except (TypeError, ValueError) as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Sample size must be a whole number of rows.") from e
        try:
            overrides = {
                **source,
                "output_dir": str(self.output_dir),
                "delimiter": request.get("delimiter") or None,
                "decimal": request.get("decimal") or None,
                "encoding": request.get("encoding") or None,
                "excel_sheet": request.get("sheet") or None,
                "sample_size": sample_size,
                **({"target": target_overrides} if target_overrides else {}),
            }
            return merge_configs(Config(), {}, overrides)
        except ConfigError as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(e)) from e

    def _execute(self, run: Run, config: Config) -> None:
        pipeline = EDAPipeline(config)
        seen: list[str] = []

        def progress(dataset: str, step: str) -> None:
            if dataset not in seen:
                seen.append(dataset)
            run.dataset, run.stage, run.datasets_done = dataset, stage_of(step), len(seen) - 1

        pipeline.on_progress = progress
        try:
            pipeline.setup()
            results = pipeline.run()
            if "error" in results and len(results) == 1:
                raise RuntimeError(results["error"])
            for name, result in results.items():
                report = result.get("html_report")
                run.results.append(
                    {
                        "dataset": name,
                        "error": result.get("error"),
                        "failed_steps": result.get("failed_steps") or [],
                        "alerts": result.get("n_alerts"),
                        "url": self.report_url(Path(report)) if report else None,
                    }
                )
            run.datasets_done = len(results)
            run.status = "done" if any(r["url"] for r in run.results) else "failed"
            if run.status == "failed":
                run.error = "; ".join(f"{r['dataset']}: {r['error']}" for r in run.results if r["error"]) or (
                    "No report was written. The log in the output folder has the details."
                )
        except Exception as e:
            logger.exception("The analysis failed")
            run.status, run.error = "failed", str(e)
        finally:
            with self._lock:
                self._running = None

    def report_url(self, report: Path) -> str | None:
        try:
            return "/reports/" + quote(report.resolve().relative_to(self.output_dir).as_posix())
        except ValueError:
            return None

    def report_file(self, url_path: str) -> Path:
        """The file a /reports/ URL names, only inside the output folder and only of a served type."""
        relative = unquote(url_path[len("/reports/") :])
        path = (self.output_dir / relative).resolve()
        if not path.is_relative_to(self.output_dir) or not path.is_file() or path.suffix not in _SERVED_TYPES:
            raise RequestError(HTTPStatus.NOT_FOUND, "No such report.")
        return path


class AppHandler(BaseHTTPRequestHandler):
    """Routes the app's requests to its EdaApp (set on the server as ``server.app``)."""

    server_version = "EDA-app"

    @property
    def app(self) -> EdaApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002 - the name BaseHTTPRequestHandler uses
        logger.debug("%s - %s", self.address_string(), format % args)

    # ---- plumbing
    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _check_host(self) -> None:
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in allowed:
            raise RequestError(HTTPStatus.FORBIDDEN, "This app only answers on 127.0.0.1 and localhost.")

    def _check_app_header(self) -> None:
        if self.headers.get(APP_HEADER) != "1":
            raise RequestError(HTTPStatus.FORBIDDEN, "Requests that change something must come from the app's page.")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, "The request is not valid JSON.") from e
        if not isinstance(payload, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "The request must be a JSON object.")
        return payload

    def _handle(self, route) -> None:
        try:
            self._check_host()
            route()
        except RequestError as e:
            self._json(e.status, {"error": e.message})
        except Exception:
            logger.exception("Request failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Something failed on the server. Check its log."})

    # ---- routes
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._handle(self._get)

    def do_HEAD(self):  # noqa: N802
        self._handle(self._get)

    def do_POST(self):  # noqa: N802
        self._handle(self._post)

    def do_DELETE(self):  # noqa: N802
        self._handle(self._delete)

    def _get(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, APP_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/reports":
            self._json(HTTPStatus.OK, {"reports": recent_reports(self.app.output_dir)})
        elif path == "/api/stages":
            self._json(HTTPStatus.OK, {"stages": [{"id": key, "label": label} for key, label in STAGES]})
        elif path.startswith("/api/runs/"):
            run = self.app.runs.get(path.rsplit("/", 1)[1])
            if run is None:
                raise RequestError(HTTPStatus.NOT_FOUND, "No such analysis.")
            self._json(HTTPStatus.OK, run.as_dict())
        elif path.startswith("/reports/"):
            file = self.app.report_file(path)
            self._send(HTTPStatus.OK, file.read_bytes(), _SERVED_TYPES[file.suffix])
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found.")

    def _post(self) -> None:
        self._check_app_header()
        path = urlparse(self.path).path
        if path == "/api/files":
            length = self.headers.get("Content-Length")
            if length is None:
                raise RequestError(HTTPStatus.LENGTH_REQUIRED, "The upload has no length.")
            upload = self.app.add_upload(unquote(self.headers.get("X-File-Name", "")), self.rfile, int(length))
            self._json(
                HTTPStatus.CREATED,
                {"id": upload.id, "name": upload.name, "size": upload.size, "columns": upload.columns},
            )
        elif path == "/api/runs":
            run = self.app.start_run(self._read_json())
            self._json(HTTPStatus.ACCEPTED, run.as_dict())
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found.")

    def _delete(self) -> None:
        self._check_app_header()
        path = urlparse(self.path).path
        if not path.startswith("/api/files/"):
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found.")
        self.app.remove_upload(path.rsplit("/", 1)[1])
        self._json(HTTPStatus.OK, {"removed": True})


def make_server(app: EdaApp, port: int = 8765) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 for the app (port 0 picks a free one)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    server.app = app  # type: ignore[attr-defined]
    server.daemon_threads = True
    return server


def serve(output_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    """Run the app until Ctrl+C, then remove the uploaded copies."""
    app = EdaApp(output_dir)
    server = make_server(app, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"EDA app running at {url}  (reports in {app.output_dir})")
    print("Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
        print(f"Stopped at {datetime.now():%H:%M}. Uploaded copies removed; reports kept.")
