"""
The local app's page (see app.py): add files, choose a target and reading options, run the analysis with its
progress step by step, and open the report. Same tokens and theme as the report, one inline script, no
external files.
"""

APP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Pipeline · New analysis</title>
<style>
/* Instrument panel, as in the report: cool neutral ground, white panels, a compact sans with figures in mono.
   The accent marks the primary action and the target. */
:root {
    --font-sans: system-ui, -apple-system, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas, monospace;
    --ground: #eef1f4; --surface: #ffffff; --sunken: #f5f7f9; --ink: #121a22; --muted: #55616d; --rule: #d6dce2;
    --accent: #0a6a8f; --accent-ink: #ffffff; --accent-soft: #e2f0f6;
    --danger: #a1172f; --danger-bg: #fbe3e7; --ok: #2d6a3e; --ok-bg: #e2f0e5; --focus: #1a73c7;
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --ground: #0e1318; --surface: #161d24; --sunken: #121820; --ink: #e3e9ee; --muted: #9aa7b3; --rule: #2a343e;
        --accent: #5cc0e6; --accent-ink: #06222e; --accent-soft: #15303c;
        --danger: #f28b9c; --danger-bg: #3d1b22; --ok: #8fd3a2; --ok-bg: #1c3324; --focus: #7cb8f2; color-scheme: dark;
    }
}
:root[data-theme="dark"] {
    --ground: #0e1318; --surface: #161d24; --sunken: #121820; --ink: #e3e9ee; --muted: #9aa7b3; --rule: #2a343e;
    --accent: #5cc0e6; --accent-ink: #06222e; --accent-soft: #15303c;
    --danger: #f28b9c; --danger-bg: #3d1b22; --ok: #8fd3a2; --ok-bg: #1c3324; --focus: #7cb8f2; color-scheme: dark;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--ground); color: var(--ink); font: 0.9375rem/1.55 var(--font-sans);
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
a { color: var(--accent); text-underline-offset: 0.2em; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 4px; }
.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
svg.i { width: 1.05rem; height: 1.05rem; flex: none; }
[hidden] { display: none !important; }

.appbar { position: sticky; top: 0; z-index: 10; background: var(--surface); border-bottom: 1px solid var(--rule); }
.appbar__in { max-width: 76rem; margin: 0 auto; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.25rem;
    padding: 0.55rem max(16px, env(safe-area-inset-right, 0px)) 0.55rem max(16px, env(safe-area-inset-left, 0px)); }
.brand { font-weight: 700; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.5rem; }
.brand svg { width: 1.3rem; height: 1.3rem; color: var(--accent); }
.spacer { flex: 1; }
.note { font-size: 0.8rem; color: var(--muted); }
.btn { font: inherit; font-weight: 600; border: 1px solid var(--rule); background: var(--surface); color: var(--ink); border-radius: 8px;
    padding: 0.5rem 0.85rem; cursor: pointer; display: inline-flex; align-items: center; gap: 0.45rem; text-decoration: none;
    -webkit-tap-highlight-color: transparent; transition-property: scale, background-color; transition-duration: 150ms; transition-timing-function: ease-out; }
.btn:not(:disabled):active { scale: 0.96; }
.btn--primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
@media (hover: hover) {
    .btn:not(:disabled):hover { background: var(--sunken); }
    .btn--primary:not(:disabled):hover { background: var(--accent); filter: brightness(1.08); }
}
@media (prefers-reduced-motion: reduce) { .btn { transition: none; } .btn:not(:disabled):active { scale: 1; } }

.page { max-width: 76rem; margin: 0 auto; padding: 1.5rem max(16px, env(safe-area-inset-right, 0px)) 4rem max(16px, env(safe-area-inset-left, 0px)); }
.page-title { font-size: 1.45rem; line-height: 1.2; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 0.35rem; }
.lede { color: var(--muted); max-width: 70ch; margin: 0 0 1.25rem; }
.grid { display: grid; gap: 1.25rem; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); align-items: start; }
.grid > .wide { grid-column: 1 / -1; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 1.25rem 1.4rem 1.4rem; min-width: 0; }
.panel > header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 0.4rem 1rem; margin-bottom: 1rem; }
.panel h2 { font-size: 1.05rem; line-height: 1.3; font-weight: 650; margin: 0; }

.drop { position: relative; border: 1.5px dashed var(--rule); border-radius: 12px; background: var(--sunken); padding: 2rem 1.25rem;
    text-align: center; display: grid; justify-items: center; gap: 0.5rem; transition-property: border-color, background-color; transition-duration: 150ms; }
.drop.is-over { border-color: var(--accent); background: var(--accent-soft); }
.drop svg { width: 2rem; height: 2rem; color: var(--accent); }
.drop strong { font-size: 1.02rem; }
.drop input { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
.drop:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }
.files { list-style: none; margin: 1rem 0 0; padding: 0; display: grid; gap: 0.5rem; }
.files li { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 0.75rem; align-items: center; padding: 0.65rem 0.8rem;
    border: 1px solid var(--rule); border-radius: 10px; }
.files li.is-error { border-color: var(--danger); background: var(--danger-bg); }
.files .name { font-weight: 600; overflow-wrap: anywhere; }
.files .meta { color: var(--muted); font-size: 0.82rem; overflow-wrap: anywhere; }
.files li.is-error .meta { color: var(--danger); font-family: var(--font-sans); }
.files .x { position: relative; border: 0; background: none; color: var(--muted); cursor: pointer; padding: 0.35rem; border-radius: 6px; display: grid; }
.files .x::after { content: ""; position: absolute; inset: -0.4rem; }
@media (hover: hover) { .files .x:hover { color: var(--ink); background: var(--sunken); } }
.privacy { display: flex; gap: 0.6rem; align-items: flex-start; color: var(--muted); font-size: 0.85rem; margin-top: 1rem; }

.field { display: grid; gap: 0.35rem; margin-bottom: 1rem; }
.field label { font-weight: 600; font-size: 0.88rem; }
.field .hint { color: var(--muted); font-size: 0.8rem; }
select, input[type="number"], input[type="text"] { font: inherit; font-size: max(1rem, 16px); color: var(--ink); background: var(--surface);
    border: 1px solid var(--rule); border-radius: 8px; padding: 0.5rem 0.6rem; width: 100%; min-height: 2.6rem; }
select:disabled { opacity: 0.6; }
@media (min-width: 48rem) { select, input[type="number"], input[type="text"] { font-size: 0.94rem; } }
details.adv { border-top: 1px solid var(--rule); padding-top: 0.5rem; margin-bottom: 0.75rem; }
details.adv > summary { cursor: pointer; font-weight: 600; font-size: 0.88rem; padding: 0.4rem 0; color: var(--muted); }
details.adv[open] > summary { margin-bottom: 0.6rem; }
.run-row { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
.message { margin: 0.9rem 0 0; padding: 0.7rem 0.85rem; border-radius: 10px; font-size: 0.9rem; }
.message:empty { display: none; }
.message--error { color: var(--danger); background: var(--danger-bg); }
.message--ok { color: var(--ok); background: var(--ok-bg); }

.progress { margin-top: 1.1rem; }
.progress__head { display: flex; justify-content: space-between; gap: 1rem; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem; }
.progress__head strong { color: var(--ink); overflow-wrap: anywhere; }
.steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.3rem; }
.steps li { display: flex; gap: 0.6rem; align-items: center; color: var(--muted); font-size: 0.9rem; }
.steps li::before { content: ""; width: 0.6rem; height: 0.6rem; border-radius: 50%; border: 1.5px solid var(--rule); flex: none; }
.steps li.is-done { color: var(--ink); }
.steps li.is-done::before { background: var(--ok); border-color: var(--ok); }
.steps li.is-now { color: var(--ink); font-weight: 600; }
.steps li.is-now::before { border-color: var(--accent); background: var(--accent-soft); }
.steps li .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
.results { list-style: none; margin: 1rem 0 0; padding: 0; display: grid; gap: 0.5rem; }
.results li { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem 1rem; }
.results .what { overflow-wrap: anywhere; }

.runs { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.5rem; }
.runs a { display: grid; gap: 0.15rem; padding: 0.7rem 0.8rem; border: 1px solid var(--rule); border-radius: 10px; text-decoration: none; color: var(--ink); }
@media (hover: hover) { .runs a:hover { background: var(--sunken); } }
.runs strong { overflow-wrap: anywhere; }
.runs .meta { color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; }
.empty { color: var(--muted); margin: 0; }

@media (max-width: 60rem) { .grid { grid-template-columns: minmax(0, 1fr); } .panel { padding: 1rem; } }
/* On a phone the bar wraps to three lines; it scrolls away instead of covering a quarter of the screen. */
@media (max-width: 40rem) { .appbar { position: static; } }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
    <symbol id="i-mark" viewBox="0 0 24 24"><path d="M4 19V5m0 14h16M8 15v-4m4 4V8m4 7v-6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></symbol>
    <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 15V4m0 0-4 4m4-4 4 4M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></symbol>
    <symbol id="i-file" viewBox="0 0 20 20"><path d="M5 2.5h6.5L15.5 6.5v11H5v-15Z M11.5 2.5v4h4" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></symbol>
    <symbol id="i-x" viewBox="0 0 20 20"><path d="m6 6 8 8m0-8-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></symbol>
    <symbol id="i-lock" viewBox="0 0 20 20"><rect x="4" y="9" width="12" height="8.5" rx="2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9" fill="none" stroke="currentColor" stroke-width="1.4"/></symbol>
</svg>

<header class="appbar">
    <div class="appbar__in">
        <span class="brand"><svg aria-hidden="true"><use href="#i-mark"/></svg>EDA Pipeline</span>
        <span class="spacer"></span>
        <span class="note">Running on this computer</span>
        <button type="button" class="btn" id="theme-toggle">Theme: system</button>
    </div>
</header>

<main class="page">
    <h1 class="page-title">New analysis</h1>
    <p class="lede">Add one file, or several to analyze them as a batch. Delimiter, encoding, decimal separator and column
        types are detected on their own.</p>
    <div class="grid">
        <section class="panel" aria-labelledby="h-files">
            <header><h2 id="h-files">Files</h2><span class="note">CSV, TSV, TXT, Excel, Parquet, JSON</span></header>
            <div class="drop" id="drop">
                <svg aria-hidden="true"><use href="#i-upload"/></svg>
                <strong>Drop files here, or choose them</strong>
                <span class="note">Each file gets its own report</span>
                <input type="file" id="file-input" multiple accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json,.jsonl" aria-label="Choose files to analyze">
            </div>
            <ul class="files" id="file-list"></ul>
            <p class="privacy"><svg class="i" aria-hidden="true"><use href="#i-lock"/></svg>
                <span>Files are read on this computer. The page sends them to the app running here, and nowhere else.</span></p>
        </section>

        <section class="panel" aria-labelledby="h-options">
            <header><h2 id="h-options">Options</h2></header>
            <div class="field">
                <label for="target">Target column</label>
                <select id="target" disabled><option value="">No target</option></select>
                <span class="hint" id="target-hint">Read from the first file's header once you add it.</span>
            </div>
            <div class="field">
                <label for="target-type">Target type</label>
                <select id="target-type">
                    <option value="">Detect it</option>
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                </select>
            </div>
            <details class="adv">
                <summary>Reading options</summary>
                <div class="field"><label for="opt-delimiter">Delimiter</label>
                    <select id="opt-delimiter"><option value="">Detect it</option><option value=",">Comma</option>
                        <option value=";">Semicolon</option><option value="&#9;">Tab</option><option value="|">Pipe</option></select></div>
                <div class="field"><label for="opt-decimal">Decimal separator</label>
                    <select id="opt-decimal"><option value="">Detect it</option><option value=".">Point</option><option value=",">Comma</option></select></div>
                <div class="field"><label for="opt-encoding">Encoding</label>
                    <input type="text" id="opt-encoding" placeholder="utf-8" autocomplete="off" spellcheck="false">
                    <span class="hint">Leave it empty to detect it.</span></div>
                <div class="field"><label for="opt-sheet">Excel sheet</label>
                    <input type="text" id="opt-sheet" placeholder="Sheet1" autocomplete="off" spellcheck="false">
                    <span class="hint">A name, or its position from 0. Empty means the first sheet.</span></div>
                <div class="field"><label for="opt-sample">Sample size</label>
                    <input type="number" id="opt-sample" min="1" step="1" inputmode="numeric" placeholder="All rows">
                    <span class="hint">A random sample of rows, for very large files.</span></div>
            </details>
            <div class="run-row">
                <button type="button" class="btn btn--primary" id="run" disabled>Run analysis</button>
                <span class="note" id="run-note">Add a file to start.</span>
            </div>
            <p class="message" id="message" role="status"></p>
            <div class="progress" id="progress" hidden>
                <div class="progress__head"><strong id="progress-dataset"></strong><span class="num" id="progress-count"></span></div>
                <ol class="steps" id="steps"></ol>
                <ul class="results" id="results"></ul>
            </div>
        </section>

        <section class="panel wide" aria-labelledby="h-recent">
            <header><h2 id="h-recent">Recent reports</h2><span class="note" id="recent-note"></span></header>
            <ul class="runs" id="recent"></ul>
            <p class="empty" id="recent-empty" hidden>No reports yet. Your first analysis appears here.</p>
        </section>
    </div>
</main>

<script>
/* The page's only script: theme, files, options, the run and its progress, recent reports. */
(function () {
    var root = document.documentElement;
    var HEADERS = { "X-EDA-App": "1" };

    // ---- theme, shared with the reports this app serves
    var themes = ["system", "light", "dark"], toggle = document.getElementById("theme-toggle"), theme = "system";
    try { theme = localStorage.getItem("eda-report-theme") || "system"; } catch (err) {}
    function applyTheme(next) {
        theme = themes.indexOf(next) >= 0 ? next : "system";
        if (theme === "system") { root.removeAttribute("data-theme"); } else { root.setAttribute("data-theme", theme); }
        toggle.textContent = "Theme: " + theme;
        try { localStorage.setItem("eda-report-theme", theme); } catch (err) {}
    }
    applyTheme(theme);
    toggle.addEventListener("click", function () { applyTheme(themes[(themes.indexOf(theme) + 1) % themes.length]); });

    function el(id) { return document.getElementById(id); }
    var input = el("file-input"), drop = el("drop"), list = el("file-list"), target = el("target"), hint = el("target-hint");
    var run = el("run"), runNote = el("run-note"), message = el("message");
    var files = [], running = false;

    function count(n, one, many) { return n + " " + (n === 1 ? one : many); }
    function size(n) { return n < 1024 ? n + " B" : n < 1048576 ? Math.round(n / 1024) + " KB" : (n / 1048576).toFixed(1) + " MB"; }
    function say(text, kind) { message.textContent = text || ""; message.className = "message" + (kind ? " message--" + kind : ""); }
    function errorText(response, body) { return (body && body.error) || ("The app answered " + response.status + "."); }
    function request(method, url, body, headers) {
        return fetch(url, { method: method, headers: Object.assign({}, HEADERS, headers || {}), body: body }).then(function (response) {
            return response.json().catch(function () { return null; }).then(function (data) {
                if (!response.ok) { throw new Error(errorText(response, data)); }
                return data;
            });
        });
    }

    // ---- files
    function ready() { return files.filter(function (f) { return f.id; }); }
    function render() {
        list.innerHTML = "";
        files.forEach(function (f) {
            var li = document.createElement("li");
            li.innerHTML = '<svg class="i" aria-hidden="true"><use href="#i-file"/></svg><div><div class="name"></div><div class="meta num"></div></div>' +
                '<button type="button" class="x"><svg class="i" aria-hidden="true"><use href="#i-x"/></svg></button>';
            li.querySelector(".name").textContent = f.name;
            var meta = f.error ? f.error : f.id ? size(f.size) + (f.columns ? " · " + count(f.columns.length, "column", "columns") : "") : "Adding… " + size(f.size);
            li.querySelector(".meta").textContent = meta;
            if (f.error) { li.className = "is-error"; }
            var remove = li.querySelector(".x");
            remove.setAttribute("aria-label", "Remove " + f.name);
            remove.disabled = running || (!f.id && !f.error);
            remove.addEventListener("click", function () { removeFile(f); });
            list.appendChild(li);
        });
        var first = ready()[0], chosen = target.value;
        target.innerHTML = '<option value="">No target</option>';
        if (first && first.columns) {
            first.columns.forEach(function (c) { var o = document.createElement("option"); o.value = c; o.textContent = c; target.appendChild(o); });
            target.value = first.columns.indexOf(chosen) >= 0 ? chosen : "";
            hint.textContent = "Columns of " + first.name + ".";
        } else {
            hint.textContent = first ? "This file's columns could not be read ahead; the analysis reads them." : "Read from the first file's header once you add it.";
        }
        target.disabled = running || !(first && first.columns);
        var pending = files.some(function (f) { return !f.id && !f.error; }), n = ready().length;
        run.disabled = running || pending || !n;
        runNote.textContent = running ? "Running…" : pending ? "Adding files…" : !n ? "Add a file to start." :
            n === 1 ? "One report." : n + " files: one report each, in one batch folder.";
    }
    function addFiles(fileList) {
        say("");
        Array.prototype.forEach.call(fileList, function (file) {
            var entry = { name: file.name, size: file.size, id: null, columns: null, error: null };
            files.push(entry);
            request("POST", "/api/files", file, { "X-File-Name": encodeURIComponent(file.name) }).then(function (data) {
                entry.id = data.id; entry.name = data.name; entry.columns = data.columns;
            }, function (err) { entry.error = err.message; }).then(render);
        });
        render();
    }
    function removeFile(f) {
        var done = function () { files.splice(files.indexOf(f), 1); render(); };
        if (!f.id) { done(); return; }
        request("DELETE", "/api/files/" + f.id).then(done, function (err) { say(err.message, "error"); done(); });
    }
    input.addEventListener("change", function () { addFiles(input.files); input.value = ""; });
    ["dragenter", "dragover"].forEach(function (name) { drop.addEventListener(name, function (e) { e.preventDefault(); drop.classList.add("is-over"); }); });
    ["dragleave", "drop"].forEach(function (name) { drop.addEventListener(name, function (e) { e.preventDefault(); drop.classList.remove("is-over"); }); });
    drop.addEventListener("drop", function (e) { if (e.dataTransfer && e.dataTransfer.files.length) { addFiles(e.dataTransfer.files); } });

    // ---- the run
    var stages = [], steps = el("steps"), progress = el("progress"), results = el("results");
    request("GET", "/api/stages").then(function (data) { stages = data.stages; }, function () {});
    function drawSteps(current) {
        var at = stages.findIndex(function (s) { return s.id === current; });
        steps.innerHTML = "";
        stages.forEach(function (stage, i) {
            var li = document.createElement("li");
            li.textContent = stage.label;
            var state = at < 0 ? "" : i < at ? "is-done" : i === at ? "is-now" : "";
            if (state) {
                li.className = state;
                var sr = document.createElement("span"); sr.className = "sr";
                sr.textContent = state === "is-done" ? " (done)" : " (in progress)";
                li.appendChild(sr);
            }
            if (state === "is-now") { li.setAttribute("aria-current", "step"); }
            steps.appendChild(li);
        });
    }
    function showResults(data) {
        results.innerHTML = "";
        data.results.forEach(function (r) {
            var li = document.createElement("li"), what = document.createElement("span");
            what.className = "what";
            what.textContent = r.error ? r.dataset + ": " + r.error :
                r.dataset + (r.failed_steps.length ? " · finished, but " + count(r.failed_steps.length, "step", "steps") + " failed" : "") +
                (r.alerts != null ? " · " + count(r.alerts, "alert", "alerts") : "");
            li.appendChild(what);
            if (r.url) {
                var a = document.createElement("a"); a.className = "btn btn--primary"; a.href = r.url;
                a.textContent = data.results.length > 1 ? "Open the report for " + r.dataset : "Open the report";
                li.appendChild(a);
            }
            results.appendChild(li);
        });
    }
    function poll(id) {
        request("GET", "/api/runs/" + id).then(function (data) {
            el("progress-dataset").textContent = data.dataset || "Starting…";
            el("progress-count").textContent = data.datasets_total > 1 ? "File " + Math.min(data.datasets_done + 1, data.datasets_total) + " of " + data.datasets_total : "";
            if (data.status === "running") { drawSteps(data.stage); setTimeout(function () { poll(id); }, 700); return; }
            running = false;
            drawSteps(data.status === "done" ? "__all__" : data.stage);
            if (data.status === "done") { steps.querySelectorAll("li").forEach(function (li) { li.className = "is-done"; }); }
            showResults(data);
            say(data.status === "done" ? "Done. The report is ready." : "The analysis did not finish: " + data.error, data.status === "done" ? "ok" : "error");
            render();
            loadRecent();
        }, function (err) { running = false; say(err.message, "error"); render(); });
    }
    run.addEventListener("click", function () {
        var body = {
            files: ready().map(function (f) { return f.id; }),
            target: target.value || null,
            target_type: el("target-type").value || null,
            delimiter: el("opt-delimiter").value || null,
            decimal: el("opt-decimal").value || null,
            encoding: el("opt-encoding").value.trim() || null,
            sheet: el("opt-sheet").value.trim() || null,
            sample_size: el("opt-sample").value || null
        };
        running = true; render(); say("");
        results.innerHTML = ""; progress.hidden = false; drawSteps("");
        request("POST", "/api/runs", JSON.stringify(body), { "Content-Type": "application/json" }).then(function (data) { poll(data.id); },
            function (err) { running = false; progress.hidden = true; say(err.message, "error"); render(); });
    });

    // ---- recent reports
    function loadRecent() {
        request("GET", "/api/reports").then(function (data) {
            var ul = el("recent"); ul.innerHTML = "";
            el("recent-empty").hidden = data.reports.length > 0;
            el("recent-note").textContent = data.reports.length ? "From this app's reports folder" : "";
            data.reports.forEach(function (r) {
                var li = document.createElement("li"), a = document.createElement("a"), name = document.createElement("strong"), meta = document.createElement("span");
                a.href = r.url; name.textContent = r.name; meta.className = "meta num";
                var parts = [];
                if (r.rows != null) { parts.push(r.rows.toLocaleString("en-US") + " rows"); }
                if (r.columns != null) { parts.push(count(r.columns, "column", "columns")); }
                parts.push(count(r.alerts, "alert", "alerts"));
                parts.push(r.target ? "target " + r.target : "no target");
                if (r.failed_steps) { parts.push(count(r.failed_steps, "failed step", "failed steps")); }
                if (r.generated) { parts.push(r.generated); }
                meta.textContent = parts.join(" · ");
                a.appendChild(name); a.appendChild(meta); li.appendChild(a); ul.appendChild(li);
            });
        }, function () {});
    }
    loadRecent();
    render();
})();
</script>
</body>
</html>
"""
