#!/usr/bin/env bash
# Stado GPU job: train, audit, export, and stage Oko's lifecycle model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_DATASET="${1:?usage: run.sh REVIEWED_TRAIN_JSONL REVIEWED_EVAL_JSONL}"
EVAL_DATASET="${2:?usage: run.sh REVIEWED_TRAIN_JSONL REVIEWED_EVAL_JSONL}"
JOB_ID="${WC_JOB_ID:?WC_JOB_ID is required}"
STAGING_ROOT="${TMPDIR:-/tmp}"
WORK="${LIFECYCLE_MODEL_WORK_DIR:-$STAGING_ROOT/oko-lifecycle-model-$JOB_ID}"
OUT="/tmp/wc-$JOB_ID/output"
VENV="$WORK/venv"
mkdir -p "$WORK" "$OUT"
cp "$TRAIN_DATASET" "$WORK/reviewed-train.jsonl"
cp "$EVAL_DATASET" "$WORK/reviewed-eval.jsonl"
cd "$WORK"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet \
  'torch>=2.6,<3' 'transformers>=4.51,<5' 'datasets>=3.5,<5' \
  'accelerate>=1.6,<2' 'sentencepiece>=0.2,<1' 'safetensors>=0.5,<1'

export LIFECYCLE_TRAIN_DATASET="$WORK/reviewed-train.jsonl"
export LIFECYCLE_EVAL_DATASET="$WORK/reviewed-eval.jsonl"
export LIFECYCLE_STUDENT_MODEL="Qwen/Qwen3-4B"
export LIFECYCLE_STUDENT_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
if [ ! -s "$WORK/student/config.json" ] \
  || [ ! -s "$WORK/predictions.jsonl" ] \
  || [ ! -s "$WORK/metrics.json" ]; then
  "$VENV/bin/python" "$ROOT/training/lifecycle-model/train.py"
fi

if [ ! -s "$WORK/python-requirements.lock" ]; then
  "$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"
fi

LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
if [ ! -s "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" ]; then
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
    --outfile "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" --outtype f16
fi
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-quantize llama-server \
  -j "${LIFECYCLE_LLAMA_BUILD_JOBS:-8}"
if [ ! -s "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" ]; then
  "$LLAMA_CPP/build/bin/llama-quantize" \
    "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" \
    "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" Q4_K_M
fi

# The gate must measure what production serves: the quantized GGUF answering
# Oko's loopback chat contract with decoding constrained to the checked-in
# decision schema. train.py's own in-process generation is a different surface
# and cannot stand in for it.
if [ ! -s "$WORK/metrics-gguf.json" ]; then
  LIFECYCLE_EVAL_WORK="$WORK" \
  LIFECYCLE_EVAL_MODEL="$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" \
  LIFECYCLE_EVAL_DATASET="$WORK/reviewed-eval.jsonl" \
  LIFECYCLE_EVAL_PREDICTIONS="$WORK/predictions-gguf.jsonl" \
  LIFECYCLE_EVAL_METRICS="$WORK/metrics-gguf.json" \
  LIFECYCLE_EVAL_SERVER_LOG="$WORK/llama-server-eval.log" \
  LIFECYCLE_EVAL_SERVER="$LLAMA_CPP/build/bin/llama-server" \
  LIFECYCLE_EVAL_SYSTEM_PROMPT="$ROOT/training/lifecycle-model/lifecycle-system-prompt.txt" \
  LIFECYCLE_EVAL_OUTPUT_SCHEMA="$ROOT/training/lifecycle-model/lifecycle-output-schema.json" \
  LIFECYCLE_EVAL_PARALLEL="${LIFECYCLE_EVAL_PARALLEL:-8}" \
    "$VENV/bin/python" "$ROOT/training/lifecycle-model/evaluate-gguf-host.py"
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
set +e
"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  lifecycle-audit "$WORK/predictions-gguf.jsonl" \
  --output "$WORK/final-judge.json" \
  --brama-model "${LIFECYCLE_AUDIT_MODEL:-claude-code/claude-sonnet-4-6}"
AUDIT_EXIT=$?
set -e
[ -s "$WORK/final-judge.json" ]

MODEL="$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf"
MODEL_NAME="$(basename "$MODEL")"
rm -f "$OUT/$MODEL_NAME".part-*
split -b "${LIFECYCLE_MODEL_PART_BYTES:-128M}" -d -a 3 \
  "$MODEL" "$OUT/$MODEL_NAME.part-"
cp "$WORK/metrics.json" "$WORK/predictions.jsonl" \
   "$WORK/metrics-gguf.json" "$WORK/predictions-gguf.jsonl" \
   "$WORK/python-requirements.lock" "$WORK/final-judge.json" \
   "$ROOT/training/lifecycle-model/lifecycle-system-prompt.txt" \
   "$ROOT/training/lifecycle-model/lifecycle-output-schema.json" "$OUT/"

OUT="$OUT" MODEL="$MODEL" MODEL_NAME="$MODEL_NAME" "$VENV/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
model = Path(os.environ["MODEL"])
model_name = os.environ["MODEL_NAME"]
parts = sorted(path.name for path in out.glob(f"{model_name}.part-*"))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

judge = json.loads((out / "final-judge.json").read_text(encoding="utf-8"))
# The gate reads the served surface. The trainer's in-process numbers are kept
# beside them for comparison, never as the thing being gated.
metrics = json.loads((out / "metrics-gguf.json").read_text(encoding="utf-8"))
training_metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
files = {}
for path in sorted(out.iterdir()):
    if path.name == "model-manifest.json" or not path.is_file():
        continue
    files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
qualified = (
    judge.get("passed") is True
    and metrics.get("valid_json", 0) >= 0.99
    and metrics.get("action_accuracy", 0) >= 0.90
    and metrics.get("joint_accuracy", 0) >= 0.88
    and metrics.get("finish_precision", 0) == 1.0
)
manifest = {
    "product": "Oko goal lifecycle model",
    "contract": "oko-goal-lifecycle-v1",
    "format": "GGUF",
    "default_artifact": model_name,
    "base_model": "Qwen/Qwen3-4B",
    "base_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "required_quality_gate": "final-judge.json",
    "evaluation_surface": "served Q4_K_M GGUF through Oko's loopback chat contract, decoding constrained to lifecycle-output-schema.json",
    "qualified": qualified,
    "review_model": judge.get("review_model"),
    "metrics": {
        "valid_json": metrics.get("valid_json"),
        "action_accuracy": metrics.get("action_accuracy"),
        "goal_ref_accuracy": metrics.get("goal_ref_accuracy"),
        "evidence_accuracy": metrics.get("evidence_accuracy"),
        "joint_accuracy": metrics.get("joint_accuracy"),
        "finish_precision": metrics.get("finish_precision"),
    },
    "training_metrics": {
        "valid_json": training_metrics.get("valid_json"),
        "action_accuracy": training_metrics.get("action_accuracy"),
        "joint_accuracy": training_metrics.get("joint_accuracy"),
        "finish_precision": training_metrics.get("finish_precision"),
    },
    "files": files,
    "transport": {
        "kind": "ordered-parts",
        "parts": parts,
        "assembled_bytes": model.stat().st_size,
        "assembled_sha256": digest(model),
    },
}
(out / "model-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
if not qualified:
    raise SystemExit("lifecycle model did not satisfy the manifest quality thresholds")
PY

if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "lifecycle model candidate staged but rejected by final audit"
  exit "$AUDIT_EXIT"
fi
echo "qualified lifecycle model staged in $OUT"
