#!/usr/bin/env bash
# Stado GPU job: curate, train, audit, and privately publish Echo's humanizer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARGETS="${1:?usage: run.sh TARGETS_JSONL}"
JOB_ID="${WC_JOB_ID:?WC_JOB_ID is required}"
WORK="${HUMANIZER_WORK_DIR:-/tmp/echo-humanizer-$JOB_ID}"
OUT="/tmp/wc-$JOB_ID/output"
VENV="$WORK/venv"
mkdir -p "$WORK" "$OUT"
if [[ ! "$TARGETS" -ef "$WORK/targets.jsonl" ]]; then
  cp "$TARGETS" "$WORK/targets.jsonl"
fi
cd "$WORK"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet \
  'torch>=2.6,<3' 'transformers>=4.51,<5' 'datasets>=3.5,<5' \
  'accelerate>=1.6,<2' 'sentencepiece>=0.2,<1' 'safetensors>=0.5,<1' \
  'peft>=0.17,<1' 'bitsandbytes>=0.46,<1' \
  'requests>=2.32,<3' 'huggingface-hub>=0.34,<1'

export HUMANIZER_TARGETS="$WORK/targets.jsonl"
export HUMANIZER_DATASET_DIR="$WORK"
export HUMANIZER_TRAIN_DATASET="$WORK/train.jsonl"
export HUMANIZER_VALIDATION_DATASET="$WORK/validation.jsonl"
export HUMANIZER_TEST_DATASET="$WORK/test.jsonl"
export HUMANIZER_PREDICTIONS="$WORK/predictions.jsonl"
export HUMANIZER_AUDIT_OUTPUT="$WORK/audit.json"
export HUMANIZER_MODEL_DIR="$WORK/student"
export HUMANIZER_METRICS="$WORK/metrics.json"
export HUMANIZER_PREPARATION="$WORK/preparation.json"
export HUMANIZER_BASE_MODEL="TheDrummer/Cydonia-24B-v4.3"
export HUMANIZER_BASE_REVISION="db0426d39d4bd4a6d34fdc71db97569da68f55e1"

"$VENV/bin/python" "$ROOT/training/humanizer-model/prepare.py"
"$VENV/bin/python" "$ROOT/training/humanizer-model/train.py"
"$VENV/bin/python" "$ROOT/training/humanizer-model/audit.py"
"$VENV/bin/python" "$ROOT/training/humanizer-model/publish.py"
"$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"

cp "$WORK/preparation.json" "$WORK/metrics.json" "$WORK/audit.json" \
   "$WORK/predictions.jsonl" "$WORK/publication.json" \
   "$WORK/python-requirements.lock" "$OUT/"
cp -R "$WORK/student" "$OUT/adapter"

OUT="$OUT" "$VENV/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
publication = json.loads((out / "publication.json").read_text(encoding="utf-8"))
metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

qualified = publication.get("qualified") is True and audit.get("passed") is True
manifest = {
    "schema_version": 1,
    "product": "Echo Łukasz humanizer",
    "contract": "echo-lukasz-humanizer-v1",
    "format": "vLLM LoRA adapter",
    "base_model": metrics.get("base_model"),
    "base_revision": metrics.get("base_revision"),
    "repo_id": publication.get("repo_id"),
    "revision": publication.get("revision"),
    "private": publication.get("private"),
    "qualified": qualified,
    "metrics": {
        "base": metrics.get("base"),
        "student": metrics.get("student"),
        "target_chrf_gain": metrics.get("target_chrf_gain"),
        "audit_base": audit.get("base"),
        "audit_student": audit.get("student"),
        "voice_match_gain": audit.get("voice_match_gain"),
        "semantic_fidelity_delta": audit.get("semantic_fidelity_delta"),
    },
    "evidence": {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(out.iterdir()) if path.is_file()
    },
}
(out / "model-manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
if not qualified:
    raise SystemExit("humanizer model did not satisfy its publication gate")
PY

echo "qualified private humanizer model and evidence staged in $OUT"
