#!/usr/bin/env python3
"""Stage an already exported goal model after an independent audit passes."""

import hashlib
import json
import os
import shutil
from pathlib import Path

WORK = Path("/mnt/wd16tb/stado/jobs/jeden-goal-prompt-v5")
OUT = Path(os.environ["OUTPUT_DIR"])
MODEL = WORK / "jeden-goal-qwen3-4b-q4_k_m.gguf"
JUDGE = WORK / "final-judge-semantic.json"
PROMPT = Path("/root/.stado/files/goal-system-prompt.md")
OUT.mkdir(parents=True, exist_ok=True)
judge = json.loads(JUDGE.read_text(encoding="utf-8"))
if judge.get("passed") is not True:
    raise SystemExit("final semantic audit did not pass")

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

model_name = MODEL.name
for path in OUT.glob(f"{model_name}.part-*"):
    path.unlink()
part_names = []
with MODEL.open("rb") as source:
    index = 0
    while chunk := source.read(128 * 1024 * 1024):
        name = f"{model_name}.part-{index:03d}"
        (OUT / name).write_bytes(chunk)
        part_names.append(name)
        index += 1
for source, name in (
    (WORK / "metrics.json", "metrics.json"),
    (WORK / "predictions.jsonl", "predictions.jsonl"),
    (WORK / "python-requirements.lock", "python-requirements.lock"),
    (JUDGE, "final-judge.json"),
    (PROMPT, "goal-system-prompt.md"),
):
    shutil.copy2(source, OUT / name)
files = {}
for path in sorted(OUT.iterdir()):
    if path.name != "model-manifest.json" and path.is_file():
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
manifest = {
    "product": "Jeden goal model",
    "format": "GGUF",
    "default_artifact": model_name,
    "base_model": "Qwen/Qwen3-4B",
    "base_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "required_quality_gate": "final-judge.json",
    "qualified": True,
    "review_model": judge.get("review_model"),
    "files": files,
    "transport": {
        "kind": "ordered-parts",
        "parts": part_names,
        "assembled_bytes": MODEL.stat().st_size,
        "assembled_sha256": digest(MODEL),
    },
}
(OUT / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(OUT)
