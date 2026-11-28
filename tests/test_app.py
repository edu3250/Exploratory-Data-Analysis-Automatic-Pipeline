"""
The local app (eda app): its HTTP guards, uploads, runs with their progress, and the recent reports list.
"""

import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from eda_pipeline.app import STAGES, EdaApp, RequestError, make_server, recent_reports, safe_file_name, stage_of
from eda_pipeline.cli import cli
from eda_pipeline.config import Config
from eda_pipeline.pipeline import EDAPipeline


def _csv_bytes(n=120, seed=3) -> bytes:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "edad": rng.integers(18, 80, n),
            "monto": rng.lognormal(5, 1, n).round(2),
            "grupo": rng.choice(["a", "b", "c"], n),
            "sale": rng.choice([0, 1], n),
        }
    )
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def app(tmp_output_dir):
    app = EdaApp(tmp_output_dir / "reports", work_dir=tmp_output_dir / "work")
    yield app
    app.close()


@pytest.fixture
def server(app):
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _request(server, method, path, body=None, headers=None, host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    all_headers = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    conn.request(method, path, body=body, headers=all_headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _upload(server, name, body):
    status, data = _request(server, "POST", "/api/files", body, {"X-EDA-App": "1", "X-File-Name": name})
    return status, json.loads(data)


class TestAppGuards:
    """The app answers only itself: its own host name, and changes only from its own page."""

    def test_serves_the_page_offline(self, server):
        status, data = _request(server, "GET", "/")
        page = data.decode("utf-8")
        assert status == 200 and '<h1 class="page-title">New analysis</h1>' in page
        assert "http://" not in page and "https://" not in page  # nothing loaded from outside

    def test_refuses_a_host_that_is_not_itself(self, server):
        status, _ = _request(server, "GET", "/api/reports", host="attacker.example:80")
        assert status == 403

    def test_a_change_needs_the_app_header(self, server):
        status, _ = _request(server, "POST", "/api/files", b"a,b\n1,2", {"X-File-Name": "x.csv"})
        assert status == 403
        status, _ = _request(server, "POST", "/api/runs", b"{}", {"Content-Type": "application/json"})
        assert status == 403

    def test_reports_are_served_only_from_the_reports_folder(self, server, app):
        (app.output_dir.parent / "secret.html").write_text("no")
        for path in ("/reports/../secret.html", "/reports/..%2Fsecret.html", "/reports/%2e%2e/secret.html"):
            status, _ = _request(server, "GET", path)
            assert status == 404, path

    def test_only_report_files_are_served(self, server, app):
        (app.output_dir / "notes.txt").write_text("x")
        status, _ = _request(server, "GET", "/reports/notes.txt")
        assert status == 404


class TestAppUploads:
    def test_an_upload_returns_its_columns(self, server):
        status, data = _upload(server, "ventas%20mayo.csv", _csv_bytes())
        assert status == 201
        assert data["name"] == "ventas mayo.csv"
        assert data["columns"] == ["edad", "monto", "grupo", "sale"]

    def test_a_format_the_pipeline_does_not_read_is_refused(self, server):
        status, data = _upload(server, "notes.docx", b"hello")
        assert status == 415
        assert "notes.docx" in data["error"] and ".csv" in data["error"]

    def test_a_removed_file_is_gone(self, server, app):
        _, data = _upload(server, "a.csv", _csv_bytes())
        status, _ = _request(server, "DELETE", f"/api/files/{data['id']}", headers={"X-EDA-App": "1"})
        assert status == 200 and data["id"] not in app.uploads
        status, _ = _request(server, "DELETE", f"/api/files/{data['id']}", headers={"X-EDA-App": "1"})
        assert status == 404

    def test_file_names_lose_their_folders_and_unsafe_characters(self):
        assert safe_file_name("../../etc/passwd.csv") == "passwd.csv"
        assert safe_file_name("C:\\datos\\ventas: mayo?.csv") == "ventas_ mayo_.csv"
        assert safe_file_name("...") == "file"


class TestAppRuns:
    def test_a_run_writes_the_report_and_walks_every_stage_in_order(self, app, monkeypatch):
        import eda_pipeline.app as app_module

        seen = []
        monkeypatch.setattr(app_module, "stage_of", lambda step: seen.append(stage_of(step)) or seen[-1])
        upload = app.add_upload("tienda.csv", _ByteStream(_csv_bytes()), len(_csv_bytes()))
        run = app.start_run({"files": [upload.id], "target": "sale"}, background=False)
        assert run.status == "done", run.error
        assert list(dict.fromkeys(seen)) == [key for key, _ in STAGES]
        assert run.results[0]["url"].startswith("/reports/tienda_") and run.results[0]["url"].endswith("/report.html")

    def test_progress_reaches_the_run_the_page_polls(self, app):
        upload = app.add_upload("tienda.csv", _ByteStream(_csv_bytes()), len(_csv_bytes()))
        run = app.start_run({"files": [upload.id]}, background=False)
        assert run.status == "done" and run.dataset == "tienda" and run.stage == "html_report"
        assert run.datasets_done == 1

    def test_several_files_run_as_one_batch(self, app):
        ids = [
            app.add_upload(name, _ByteStream(_csv_bytes(seed=i)), len(_csv_bytes(seed=i))).id
            for i, name in enumerate(["uno.csv", "dos.csv"])
        ]
        run = app.start_run({"files": ids}, background=False)
        assert run.status == "done" and run.datasets_total == 2
        assert sorted(r["dataset"] for r in run.results) == ["dos", "uno"]
        assert all("/uploads_batch_" in r["url"] for r in run.results)

    def test_recent_reports_list_the_newest_first(self, app):
        for name in ("primero.csv", "segundo.csv"):
            upload = app.add_upload(name, _ByteStream(_csv_bytes()), len(_csv_bytes()))
            app.start_run({"files": [upload.id], "target": "sale"}, background=False)
        reports = recent_reports(app.output_dir)
        assert [r["name"] for r in reports][:2] == ["segundo", "primero"]
        assert reports[0]["rows"] == 120 and reports[0]["columns"] == 4 and reports[0]["target"] == "sale"

    def test_a_missing_target_fails_with_its_message(self, app):
        upload = app.add_upload("tienda.csv", _ByteStream(_csv_bytes()), len(_csv_bytes()))
        run = app.start_run({"files": [upload.id], "target": "precio"}, background=False)
        assert run.status == "failed" and "precio" in run.error

    def test_bad_options_are_refused_before_running(self, app):
        upload = app.add_upload("tienda.csv", _ByteStream(_csv_bytes()), len(_csv_bytes()))
        with pytest.raises(RequestError, match="whole number"):
            app.start_run({"files": [upload.id], "sample_size": "muchos"}, background=False)
        with pytest.raises(RequestError, match="Add a file"):
            app.start_run({"files": ["nope"]}, background=False)

    def test_one_analysis_at_a_time(self, app):
        upload = app.add_upload("tienda.csv", _ByteStream(_csv_bytes()), len(_csv_bytes()))
        app._running = object()
        with pytest.raises(RequestError, match="already running") as caught:
            app.start_run({"files": [upload.id]}, background=False)
        assert caught.value.status == 409


class TestPipelineProgress:
    def test_the_pipeline_reports_each_step_it_starts(self, tmp_output_dir):
        csv = tmp_output_dir / "pasos.csv"
        csv.write_bytes(_csv_bytes())
        pipeline = EDAPipeline(Config(input_file=str(csv), output_dir=str(tmp_output_dir / "reports")))
        pipeline.setup()
        events = []
        pipeline.on_progress = lambda dataset, step: events.append((dataset, step))
        pipeline.run()
        steps = [step for _, step in events]
        assert {dataset for dataset, _ in events} == {"pasos"}
        assert steps[:3] == ["load", "types", "data_quality"] and steps[-1] == "html_report"
        assert set(steps) <= {key for key, _ in STAGES} | {
            "log_scale",
            "floor_split",
            "chart_columns",
            "association_matrix",
            "pair_plot",
        }


def test_the_cli_offers_the_app():
    result = CliRunner().invoke(cli, ["app", "--help"])
    assert result.exit_code == 0 and "127.0.0.1" in result.output


class _ByteStream:
    """A minimal readable stream, as the request body the server hands to add_upload."""

    def __init__(self, data: bytes):
        self._data, self._pos = data, 0

    def read(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk
