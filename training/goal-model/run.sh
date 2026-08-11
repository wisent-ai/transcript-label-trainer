#!/usr/bin/env bash
# Stado GPU job: train, independently audit, export, and stage the Jeden goal model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${1:?usage: run.sh REVIEWED_GOALS_JSONL}"
WORK="${GOAL_MODEL_WORK_DIR:-/tmp/jeden-goal-model}"
OUT="/tmp/wc-${WC_JOB_ID:?WC_JOB_ID is required}/output"
VENV="$WORK/venv"
mkdir -p "$WORK" "$OUT"
cp "$DATASET" "$WORK/reviewed-goals.jsonl"
cd "$WORK"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet \
  'torch>=2.6,<3' 'transformers>=4.51,<5' 'datasets>=3.5,<5' \
  'accelerate>=1.6,<2' 'sentencepiece>=0.2,<1' 'safetensors>=0.5,<1'

export GOAL_DATASET="$WORK/reviewed-goals.jsonl"
export GOAL_STUDENT_MODEL="Qwen/Qwen3-0.6B"
export GOAL_STUDENT_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
"$VENV/bin/python" "$ROOT/training/goal-model/train.py"

audit_model="${TLT_BRAMA_MODEL:-}"
audit_args=(goal-audit "$WORK/predictions.jsonl" --output "$WORK/final-judge.json")
if [ -n "$audit_model" ]; then
  audit_args+=(--brama-model "$audit_model")
else
  audit_args+=(--best)
fi
cargo run --manifest-path "$ROOT/Cargo.toml" --release -- "${audit_args[@]}"

LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
"$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
  --outfile "$WORK/jeden-goal-qwen3-0.6b-f16.gguf" --outtype f16
"$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
  --outfile "$WORK/jeden-goal-qwen3-0.6b-q8_0.gguf" --outtype q8_0

"$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"
cp "$WORK/jeden-goal-qwen3-0.6b-f16.gguf" \
   "$WORK/jeden-goal-qwen3-0.6b-q8_0.gguf" \
   "$WORK/metrics.json" "$WORK/predictions.jsonl" \
   "$WORK/final-judge.json" "$WORK/python-requirements.lock" \
   "$ROOT/training/goal-model/goal-system-prompt.md" "$OUT/"

OUT="$OUT" "$VENV/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
files = {}
for path in sorted(out.iterdir()):
    if path.name == "model-manifest.json" or not path.is_file():
        continue
    files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
manifest = {
    "product": "Jeden goal model",
    "format": "GGUF",
    "default_artifact": "jeden-goal-qwen3-0.6b-q8_0.gguf",
    "base_model": "Qwen/Qwen3-0.6B",
    "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "quality_gate": "final-judge.json",
    "files": files,
}
(out / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "qualified model staged in $OUT"
