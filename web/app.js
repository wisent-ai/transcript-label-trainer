"use strict";

const sessionKey = "transcript-label-trainer.gui-token";
const hashToken = new URLSearchParams(window.location.hash.slice(1)).get("token");
if (hashToken) {
  window.sessionStorage.setItem(sessionKey, hashToken);
  window.history.replaceState(null, "", window.location.pathname);
}
const token = window.sessionStorage.getItem(sessionKey) || "";

const fileInput = document.getElementById("corpus-file");
const importButton = document.getElementById("import-button");
const refreshButton = document.getElementById("refresh-button");
const statusLine = document.getElementById("result-status");

function apiHeaders(extra = {}) {
  return { "X-TLT-Token": token, ...extra };
}

function setText(id, value) {
  document.getElementById(id).textContent = value == null || value === "" ? "—" : String(value);
}

function showReport(report = {}) {
  setText("count-imported", report.imported ?? 0);
  setText("count-unchanged", report.unchanged ?? 0);
  setText("count-conflicting", report.conflicting ?? 0);
  setText("count-rejected", report.rejected ?? 0);
  setText("corpus-id", report.corpusId);
  setText("source-id", report.sourceIdentity ?? report.sourcePath);
  setText("aspect", report.aspect);
  setText("records", report.records);
}

function showPlacement(placement) {
  setText("training-root", placement?.training_root);
  setText("storage-root", placement?.storage_root);
}

function showRetained(corpus) {
  const rows = Array.isArray(corpus?.corpora) ? corpus.corpora : [];
  const selectedId = corpus?.selected?.id;
  const body = document.getElementById("retained-body");
  body.replaceChildren();
  for (const entry of rows) {
    const row = document.createElement("tr");
    const cells = [
      entry.id === selectedId ? "Selected" : "Retained",
      entry.aspect,
      entry.records,
      entry.id,
      entry.sourceName || entry.sourcePath,
      entry.adoptedAt,
    ];
    cells.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value == null ? "—" : String(value);
      row.appendChild(cell);
    });
    body.appendChild(row);
  }
  document.getElementById("empty-retained").hidden = rows.length > 0;
  document.getElementById("retained-table-wrap").hidden = rows.length === 0;
  setText("registry-path", corpus?.registry);
}

async function readJson(response) {
  const payload = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw Object.assign(new Error(payload.error || `HTTP ${response.status}`), { payload });
  return payload;
}

async function refresh() {
  refreshButton.disabled = true;
  try {
    const response = await fetch("/api/state", { headers: apiHeaders(), cache: "no-store" });
    const payload = await readJson(response);
    showPlacement(payload.placement);
    showRetained(payload.corpus);
    const mib = payload.maxUploadBytes / (1024 * 1024);
    setText("upload-limit", `${mib} MiB`);
  } catch (error) {
    statusLine.dataset.kind = "error";
    statusLine.textContent = error.message;
  } finally {
    refreshButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  importButton.disabled = !file;
  document.getElementById("file-detail").textContent = file
    ? `${file.name} · ${file.size.toLocaleString()} bytes`
    : "No file selected";
  if (file) {
    statusLine.dataset.kind = "ready";
    statusLine.textContent = "Ready to validate the complete file.";
  }
});

importButton.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  importButton.disabled = true;
  fileInput.disabled = true;
  statusLine.dataset.kind = "working";
  statusLine.textContent = "Validating and retaining corpus…";
  showReport();
  try {
    const response = await fetch("/api/corpora", {
      method: "POST",
      headers: apiHeaders({
        "Content-Type": "application/json",
        "X-TLT-Filename": encodeURIComponent(file.name),
      }),
      body: file,
      cache: "no-store",
      credentials: "same-origin",
    });
    const payload = await readJson(response);
    showReport(payload.report);
    showRetained(payload.corpus);
    statusLine.dataset.kind = "success";
    statusLine.textContent = payload.report.status === "unchanged"
      ? "This corpus was already retained. It remains selected and every record is unchanged."
      : "Corpus imported, retained, and selected. Persisted state is shown below.";
  } catch (error) {
    showReport(error.payload?.report);
    if (error.payload?.corpus) showRetained(error.payload.corpus);
    statusLine.dataset.kind = "error";
    statusLine.textContent = error.message;
  } finally {
    fileInput.disabled = false;
    importButton.disabled = false;
  }
});

refreshButton.addEventListener("click", refresh);
refresh();
