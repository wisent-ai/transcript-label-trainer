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
export GOAL_STUDENT_MODEL="Qwen/Qwen3-4B"
export GOAL_STUDENT_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
"$VENV/bin/python" "$ROOT/training/goal-model/train.py"

"$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"

LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
"$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
  --outfile "$WORK/jeden-goal-qwen3-4b-f16.gguf" --outtype f16
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-quantize -j "$(nproc)"
"$LLAMA_CPP/build/bin/llama-quantize" \
  "$WORK/jeden-goal-qwen3-4b-f16.gguf" \
  "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" Q4_K_M

set +e
cargo run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  goal-audit "$WORK/predictions.jsonl" \
  --output "$WORK/final-judge.json" \
  --best
AUDIT_EXIT=$?
set -e
[ -s "$WORK/final-judge.json" ]

cp "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" \
   "$WORK/metrics.json" "$WORK/predictions.jsonl" \
   "$WORK/python-requirements.lock" "$WORK/final-judge.json" \
   "$ROOT/training/goal-model/goal-system-prompt.md" "$OUT/"

OUT="$OUT" "$VENV/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
judge = json.loads((out / "final-judge.json").read_text(encoding="utf-8"))
files = {}
for path in sorted(out.iterdir()):
    if path.name == "model-manifest.json" or not path.is_file():
        continue
    files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
qualified = judge.get("passed") is True
manifest = {
    "product": "Jeden goal model",
    "format": "GGUF",
    "default_artifact": "jeden-goal-qwen3-4b-q4_k_m.gguf",
    "base_model": "Qwen/Qwen3-4B",
    "base_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "required_quality_gate": "final-judge.json",
    "qualified": qualified,
    "review_model": judge.get("review_model"),
    "files": files,
}
(out / "model-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "goal model candidate staged but rejected by final audit"
  exit "$AUDIT_EXIT"
fi
echo "qualified goal model staged in $OUT"
