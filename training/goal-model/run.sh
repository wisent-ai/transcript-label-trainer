#!/usr/bin/env bash
# Stado GPU job: train, independently audit, export, and stage the Jeden goal model.
set -euo pipefail

: "${GOAL_STER_OPTIONS:?goal-model passes --ster-options as GOAL_STER_OPTIONS}"
: "${GOAL_MODEL_WORK_DIR:?goal-model passes the work directory of its job as GOAL_MODEL_WORK_DIR}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET="${1:?usage: run.sh REVIEWED_GOALS_JSONL}"
WORK="$GOAL_MODEL_WORK_DIR"
OUT="/tmp/wc-${WC_JOB_ID:?WC_JOB_ID is required}/output"
VENV="$WORK/venv"
mkdir -p "$WORK" "$OUT"
cp "$DATASET" "$WORK/reviewed-goals.jsonl"
cd "$WORK"

# Ster trains the model on the chat Jeden serves: goal-examples writes each
# reviewed teacher row as the goal system prompt, the <user>…</user> message
# and the reviewed <goal> answer (gold rows held out); ster tune sft fits an
# adapter with exactly the settings goal-model was given, and ster tune merge
# folds it into a standalone checkpoint for the GGUF export below. Ster's
# report of the run is kept as the training metrics. A host without Ster
# stops here with the shell's 'ster: command not found'
# (stado product install ster --surface cli).
read -r -a STER_OPTIONS <<<"$GOAL_STER_OPTIONS"
ster --version
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
TRAINER=("$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release --)
STUDENT_MODEL="Qwen/Qwen3-4B"
STUDENT_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
if [ ! -s "$WORK/student/model.safetensors" ]; then
  "${TRAINER[@]}" goal-examples --rows "$WORK/reviewed-goals.jsonl" --output "$WORK/examples.json"
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
if [ ! -s "$WORK/jeden-goal-qwen3-4b-f16.gguf" ]; then
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/student" \
    --outfile "$WORK/jeden-goal-qwen3-4b-f16.gguf" --outtype f16
fi
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" \
  -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-quantize llama-server --parallel
if [ ! -s "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" ]; then
  "$LLAMA_CPP/build/bin/llama-quantize" \
    "$WORK/jeden-goal-qwen3-4b-f16.gguf" \
    "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" Q4_K_M
fi

# The audit judges what Jeden runs: the quantized GGUF answering the goal
# chat, decoding constrained to <goal/> or one <goal>…</goal> line.
if [ ! -s "$WORK/predictions.jsonl" ]; then
  "${TRAINER[@]}" goal-evaluate-gguf \
    --model "$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf" \
    --dataset "$WORK/reviewed-goals.jsonl" \
    --predictions "$WORK/predictions.jsonl" \
    --metrics "$WORK/metrics-gguf.json" \
    --server "$LLAMA_CPP/build/bin/llama-server" \
    --server-log "$WORK/llama-server-eval.log"
fi

set +e
"${TRAINER[@]}" goal-audit "$WORK/predictions.jsonl" \
  --output "$WORK/final-judge.json" \
  --best
AUDIT_EXIT=$?
set -e
[ -s "$WORK/final-judge.json" ]

MODEL="$WORK/jeden-goal-qwen3-4b-q4_k_m.gguf"
MODEL_NAME="$(basename "$MODEL")"
rm -f "$OUT/$MODEL_NAME".part-*
split -b "${GOAL_MODEL_PART_BYTES:-128M}" -d -a 3 \
  "$MODEL" "$OUT/$MODEL_NAME.part-"
cp "$WORK/metrics.json" "$WORK/predictions.jsonl" "$WORK/metrics-gguf.json" \
   "$WORK/python-requirements.lock" "$WORK/final-judge.json" "$OUT/"
cp "$ROOT/training/goal-model/goal-system-prompt.txt" "$OUT/goal-system-prompt.md"

"${TRAINER[@]}" model-manifest --model goal --output-dir "$OUT" --artifact "$MODEL"

if [ "$AUDIT_EXIT" -ne 0 ]; then
  echo "goal model candidate staged but rejected by final audit"
  exit "$AUDIT_EXIT"
fi
echo "qualified goal model staged in $OUT"
