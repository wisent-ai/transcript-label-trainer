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
    const actions = document.createElement("td");
    if (entry.id !== selectedId) {
      actions.appendChild(actionButton("Select", "POST", `/api/corpora/${encodeURIComponent(entry.id)}/select`));
    }
    actions.appendChild(actionButton("Remove", "DELETE", `/api/corpora/${encodeURIComponent(entry.id)}`));
    row.appendChild(actions);
    body.appendChild(row);
  }
  document.getElementById("empty-retained").hidden = rows.length > 0;
  document.getElementById("retained-table-wrap").hidden = rows.length === 0;
  setText("registry-path", corpus?.registry);
}

// corpus-select and corpus-remove from the window: the same registry
// operations, and the refusal sentence when one is refused.
function actionButton(label, method, path) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const payload = await readJson(await fetch(path, { method, headers: apiHeaders() }));
      showRetained(payload.corpus);
      statusLine.dataset.kind = "ok";
      statusLine.textContent = `${label} done. Persisted state is shown below.`;
    } catch (error) {
      statusLine.dataset.kind = "error";
      statusLine.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  return button;
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
    trainingKeys = payload.training_keys;
    trainingPresets = payload.training_presets;
    showTrainingFields();
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

// train from the window: the same run as `train`, every setting stated.
const backendSelect = document.getElementById("train-backend");
const settingsBox = document.getElementById("train-settings");
const trainButton = document.getElementById("train-button");
const trainStatus = document.getElementById("train-status");
const trainMetrics = document.getElementById("train-metrics");
let trainingKeys = {};
let trainingPresets = {};
const presetsBox = document.getElementById("train-presets");

// One button per preset of the chosen backend: it types the preset's
// documented values into the fields, which are then sent like typed ones.
function showPresetButtons() {
  presetsBox.replaceChildren();
  const presets = trainingPresets[backendSelect.value];
  if (presets === null || typeof presets !== "object") return;
  for (const [name, settings] of Object.entries(presets)) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Fill with ${name}'s documented defaults`;
    button.addEventListener("click", () => {
      settingsBox.querySelectorAll("input[data-key]").forEach((input) => {
        if (Object.hasOwn(settings, input.dataset.key)) input.value = String(settings[input.dataset.key]);
      });
      trainStatus.dataset.kind = "";
      trainStatus.textContent = `Settings filled from ${name}'s documentation; each stays editable.`;
    });
    presetsBox.append(button);
  }
}

function showTrainingFields() {
  const fine = backendSelect.value === "huggingface";
  document.getElementById("train-model-id-label").hidden = !fine;
  settingsBox.querySelectorAll("label").forEach((label) => label.remove());
  const keys = trainingKeys[backendSelect.value];
  if (!Array.isArray(keys)) {
    trainStatus.dataset.kind = "error";
    trainStatus.textContent = `The GUI state names no training settings for ${backendSelect.value}; refresh the window.`;
    return;
  }
  for (const key of keys) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "text";
    input.dataset.key = key;
    label.append(key, " ", input);
    settingsBox.append(label);
  }
  showPresetButtons();
}

backendSelect.addEventListener("change", showTrainingFields);

trainButton.addEventListener("click", async () => {
  const value = (id) => document.getElementById(id).value.trim();
  const training = {};
  settingsBox.querySelectorAll("input[data-key]").forEach((input) => {
    if (input.value.trim() !== "") training[input.dataset.key] = input.value.trim();
  });
  const holdoutPreset = document.getElementById("train-holdout-preset").checked;
  const evalSplit = document.getElementById("train-no-holdout").checked
    ? false
    : holdoutPreset
      ? "scikit-learn"
      : { fraction: Number(value("train-fraction")), seed: Number(value("train-seed")) };
  const body = {
    aspect: value("train-aspect"),
    model: backendSelect.value === "huggingface" ? value("train-model-id") : backendSelect.value,
    eval_split: evalSplit,
    training,
  };
  trainButton.disabled = true;
  trainMetrics.hidden = true;
  trainStatus.dataset.kind = "working";
  trainStatus.textContent = "Training…";
  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
      cache: "no-store",
      credentials: "same-origin",
    });
    const payload = await readJson(response);
    trainStatus.dataset.kind = "success";
    trainStatus.textContent = `Trained. Metrics written beside ${payload.metrics.model_path}.`;
    trainMetrics.textContent = JSON.stringify(payload.metrics, null, "\t");
    trainMetrics.hidden = false;
  } catch (error) {
    trainStatus.dataset.kind = "error";
    trainStatus.textContent = error.payload?.not_enough_data
      ? `Not enough labeled data: ${error.message}`
      : error.message;
  } finally {
    trainButton.disabled = false;
  }
});

refresh();
