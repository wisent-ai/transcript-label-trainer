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

# Ster trains the model on the chat it is served: lifecycle-examples writes
# each reviewed row as the lifecycle system prompt, the row's user envelope
# and the reviewed decision; ster tune sft fits an adapter on them with the
# settings lifecycle-model was given (--ster-options), and ster tune merge
# folds it into a standalone checkpoint for the GGUF export below. Ster's
# report of the run is kept as the training metrics; the gate below measures
# the served GGUF, never the trainer. A host without Ster stops here with the
# shell's 'ster: command not found' (stado product install ster --surface cli).
: "${LIFECYCLE_STER_OPTIONS:?lifecycle-model passes --ster-options as LIFECYCLE_STER_OPTIONS}"
read -r -a STER_OPTIONS <<<"$LIFECYCLE_STER_OPTIONS"
ster --version
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
STUDENT_MODEL="Qwen/Qwen3-4B"
STUDENT_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
if [ ! -s "$WORK/student/model.safetensors" ]; then
  "$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
    lifecycle-examples --rows "$WORK/reviewed-train.jsonl" --output "$WORK/examples.json"
  ster tune sft --model "$STUDENT_MODEL" --revision "$STUDENT_REVISION" --device cuda \
    --examples "$WORK/examples.json" --output "$WORK/adapter.safetensors" \
    "${STER_OPTIONS[@]}" > "$WORK/metrics.json"
  ster tune merge --model "$STUDENT_MODEL" --revision "$STUDENT_REVISION" --device cuda \
    --adapter "$WORK/adapter.safetensors" --output "$WORK/student"
fi

LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
# llama.cpp's own converter is the one step that runs its upstream Python,
# installed from llama.cpp's declared requirements.
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"
if [ ! -s "$WORK/python-requirements.lock" ]; then
  "$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"
fi
if [ ! -s "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" ]; then
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
    --outfile "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" --outtype f16
fi
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-quantize llama-server --parallel
if [ ! -s "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" ]; then
  "$LLAMA_CPP/build/bin/llama-quantize" \
    "$WORK/oko-lifecycle-qwen3-4b-f16.gguf" \
    "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" Q4_K_M
fi

# The gate must measure what production serves: the quantized GGUF answering
# Oko's loopback chat contract with decoding constrained to the checked-in
# decision schema. Ster's training report is a different surface and cannot
# stand in for it.
if [ ! -s "$WORK/metrics-gguf.json" ]; then
  "$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
    lifecycle-evaluate-gguf \
    --model "$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf" \
    --dataset "$WORK/reviewed-eval.jsonl" \
    --predictions "$WORK/predictions-gguf.jsonl" \
    --metrics "$WORK/metrics-gguf.json" \
    --server "$LLAMA_CPP/build/bin/llama-server" \
    --server-log "$WORK/llama-server-eval.log" \
    --output-schema "$ROOT/training/lifecycle-model/lifecycle-output-schema.json"
fi

set +e
"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  lifecycle-audit "$WORK/predictions-gguf.jsonl" \
  --output "$WORK/final-judge.json" \
  --brama-model "${LIFECYCLE_AUDIT_MODEL:-best}" \
  --max-wrong-share "${LIFECYCLE_AUDIT_MAX_WRONG_SHARE:?Set LIFECYCLE_AUDIT_MAX_WRONG_SHARE to the largest share of decisions the audit may call wrong}"
AUDIT_EXIT=$?
set -e
[ -s "$WORK/final-judge.json" ]

MODEL="$WORK/oko-lifecycle-qwen3-4b-q4_k_m.gguf"
MODEL_NAME="$(basename "$MODEL")"
rm -f "$OUT/$MODEL_NAME".part-*
split -b "${LIFECYCLE_MODEL_PART_BYTES:-128M}" -d -a 3 \
  "$MODEL" "$OUT/$MODEL_NAME.part-"
cp "$WORK/metrics.json" \
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
