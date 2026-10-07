#!/usr/bin/env bash
# Stado GPU job: train, independently audit, export, and stage the Jeden goal model.
set -euo pipefail

: "${GOAL_AUDIT_WORKERS:?Set GOAL_AUDIT_WORKERS to the parallel Brama calls goal-model was given for the audit}"
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
if [ ! -s "$WORK/student/config.json" ] \
  || [ ! -s "$WORK/predictions.jsonl" ] \
  || [ ! -s "$WORK/metrics.json" ]; then
  "$VENV/bin/python" "$ROOT/training/goal-model/train.py"
fi

if [ ! -s "$WORK/python-requirements.lock" ]; then
  "$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"
fi

LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
if [ ! -s "$WORK/jeden-goal-qwen3-4b-f16.gguf" ]; then
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
    --outfile "$WORK/jeden-goal-qwen3-4b-f16.gguf" --outtype f16
fi
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-quantize \
  -j "${GOAL_LLAMA_BUILD_JOBS:-8}"
if [ ! -s "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" ]; then
  "$LLAMA_CPP/build/bin/llama-quantize" \
    "$WORK/jeden-goal-qwen3-4b-f16.gguf" \
    "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" Q4_K_M
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
set +e
"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  goal-audit "$WORK/predictions.jsonl" \
  --output "$WORK/final-judge.json" \
  --best --workers "$GOAL_AUDIT_WORKERS"
AUDIT_EXIT=$?
set -e
[ -s "$WORK/final-judge.json" ]

MODEL="$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf"
MODEL_NAME="$(basename "$MODEL")"
rm -f "$OUT/$MODEL_NAME".part-*
split -b "${GOAL_MODEL_PART_BYTES:-128M}" -d -a 3 \
  "$MODEL" "$OUT/$MODEL_NAME.part-"
cp "$WORK/metrics.json" "$WORK/predictions.jsonl" \
   "$WORK/python-requirements.lock" "$WORK/final-judge.json" "$OUT/"
cp "$ROOT/training/goal-model/goal-system-prompt.txt" "$OUT/goal-system-prompt.md"

"$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release -- \
  model-manifest --model goal --output-dir "$OUT" --artifact "$MODEL"

if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "goal model candidate staged but rejected by final audit"
  exit "$AUDIT_EXIT"
fi
echo "qualified goal model staged in $OUT"
