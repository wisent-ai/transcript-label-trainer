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
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
if [ ! -s "$WORK/metrics-gguf.json" ]; then
  "$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
    lifecycle-evaluate-gguf \
    --model "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" \
    --dataset "$WORK/reviewed-eval.jsonl" \
    --predictions "$WORK/predictions-gguf.jsonl" \
    --metrics "$WORK/metrics-gguf.json" \
    --server "$LLAMA_CPP/build/bin/llama-server" \
    --server-log "$WORK/llama-server-eval.log" \
    --output-schema "$ROOT/training/lifecycle-model/lifecycle-output-schema.json" \
    --parallel "${LIFECYCLE_EVAL_PARALLEL:?lifecycle-model passes --eval-parallel as LIFECYCLE_EVAL_PARALLEL}" \
    --slot-context "${LIFECYCLE_EVAL_SLOT_CONTEXT:?lifecycle-model passes --eval-slot-context as LIFECYCLE_EVAL_SLOT_CONTEXT}" \
    --gpu-layers "${LIFECYCLE_EVAL_GPU_LAYERS:?lifecycle-model passes --eval-gpu-layers as LIFECYCLE_EVAL_GPU_LAYERS}"
fi

set +e
"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  lifecycle-audit "$WORK/predictions-gguf.jsonl" \
  --output "$WORK/final-judge.json" \
  --brama-model "${LIFECYCLE_AUDIT_MODEL:-best}"
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

"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  model-manifest --model lifecycle --output-dir "$OUT" --artifact "$MODEL"

if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "lifecycle model candidate staged but rejected by final audit"
  exit "$AUDIT_EXIT"
fi
echo "qualified lifecycle model staged in $OUT"
