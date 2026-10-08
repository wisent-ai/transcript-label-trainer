#!/usr/bin/env bash
# Stado GPU job: curate, train, audit, and privately publish Echo's humanizer.
set -euo pipefail

: "${HUMANIZER_HF_REPO:?Set HUMANIZER_HF_REPO to the private Hugging Face destination}"
: "${HUMANIZER_WORKERS:?Set HUMANIZER_WORKERS to the parallel Brama calls humanizer-model was given}"
: "${HUMANIZER_ATTEMPTS:?Set HUMANIZER_ATTEMPTS to the Brama attempts humanizer-model was given}"
: "${HUMANIZER_MIN_SEMANTIC_FIDELITY:?Set HUMANIZER_MIN_SEMANTIC_FIDELITY to the lowest semantic fidelity the adapter may score}"
: "${HUMANIZER_MIN_VOICE_MATCH:?Set HUMANIZER_MIN_VOICE_MATCH to the lowest voice match the adapter may score}"
: "${HUMANIZER_MIN_PASS_RATE:?Set HUMANIZER_MIN_PASS_RATE to the lowest share of cases the judge must pass}"
: "${HUMANIZER_MAX_BOILERPLATE_RATE:?Set HUMANIZER_MAX_BOILERPLATE_RATE to the highest share the judge may call AI boilerplate}"
: "${HUMANIZER_MIN_VOICE_GAIN:?Set HUMANIZER_MIN_VOICE_GAIN to how much closer to your voice than the base the adapter must be}"
: "${HUMANIZER_MIN_SEMANTIC_DELTA:?Set HUMANIZER_MIN_SEMANTIC_DELTA to the lowest semantic fidelity change against the base}"
: "${HUMANIZER_STER_OPTIONS:?humanizer-model passes --ster-options as HUMANIZER_STER_OPTIONS}"
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

# The vendor tools that stay Python run from their own declared requirements:
# the Hugging Face CLI humanizer-publish uploads with, and (below) llama.cpp's
# converter. Training is Ster's.
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet 'huggingface-hub>=0.36,<1'

export HUMANIZER_TARGETS="$WORK/targets.jsonl"
export HUMANIZER_TRAIN_DATASET="$WORK/train.jsonl"
export HUMANIZER_TEST_DATASET="$WORK/test.jsonl"
export HUMANIZER_PREDICTIONS="$WORK/predictions.jsonl"
export HUMANIZER_AUDIT_OUTPUT="$WORK/audit.json"
export HUMANIZER_MODEL_DIR="$WORK/student"
export HUMANIZER_METRICS="$WORK/metrics.json"
export HUMANIZER_PREPARATION="$WORK/preparation.json"
export HUMANIZER_BASE_MODEL="TheDrummer/Cydonia-24B-v4.3"
export HUMANIZER_BASE_REVISION="db0426d39d4bd4a6d34fdc71db97569da68f55e1"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
TRAINER=("$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release --)

"${TRAINER[@]}" humanizer-prepare "$HUMANIZER_TARGETS" --output-dir "$WORK" \
  --workers "$HUMANIZER_WORKERS" --attempts "$HUMANIZER_ATTEMPTS"
# Ster trains the adapter on the prepared train split, exports it in the
# vLLM/PEFT layout humanizer-publish publishes, and merges it into a
# checkpoint so the evaluation asks the student exactly as served. A host
# without Ster stops at the shell's 'ster: command not found'
# (stado product install ster --surface cli).
read -r -a STER_OPTIONS <<<"$HUMANIZER_STER_OPTIONS"
ster --version
BASE=(--model "$HUMANIZER_BASE_MODEL" --revision "$HUMANIZER_BASE_REVISION" --device cuda)
if [ ! -s "$WORK/adapter.safetensors" ]; then
  "${TRAINER[@]}" humanizer-examples --rows "$HUMANIZER_TRAIN_DATASET" --output "$WORK/examples.json"
  ster tune sft "${BASE[@]}" --examples "$WORK/examples.json" --output "$WORK/adapter.safetensors" \
    "${STER_OPTIONS[@]}" > "$WORK/training.json"
fi
ster tune export --model "$HUMANIZER_BASE_MODEL" --revision "$HUMANIZER_BASE_REVISION" \
  --adapter "$WORK/adapter.safetensors" --format peft --output "$HUMANIZER_MODEL_DIR"
if [ ! -s "$WORK/merged/model.safetensors" ] && [ ! -s "$WORK/merged/consolidated.safetensors" ]; then
  ster tune merge --model "$HUMANIZER_BASE_MODEL" --revision "$HUMANIZER_BASE_REVISION" \
    --adapter "$WORK/adapter.safetensors" --output "$WORK/merged"
fi

# Both models are evaluated at full precision, so the student's answers are
# the base's with the published adapter attached and nothing a quantizer did.
LLAMA_CPP="$WORK/llama.cpp"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
"$VENV/bin/python" -m pip install --quiet -r "$LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt"
if [ ! -s "$WORK/base-f16.gguf" ]; then
  # hf download answers the local directory of the pinned revision.
  BASE_DIR="$("$VENV/bin/hf" download "$HUMANIZER_BASE_MODEL" --revision "$HUMANIZER_BASE_REVISION")"
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$BASE_DIR" --outfile "$WORK/base-f16.gguf" --outtype f16
fi
if [ ! -s "$WORK/student-f16.gguf" ]; then
  "$VENV/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/merged" --outfile "$WORK/student-f16.gguf" --outtype f16
fi
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --target llama-server
if [ ! -s "$HUMANIZER_PREDICTIONS" ]; then
  "${TRAINER[@]}" humanizer-evaluate-gguf \
    --base "$WORK/base-f16.gguf" --student "$WORK/student-f16.gguf" \
    --dataset "$HUMANIZER_TEST_DATASET" --predictions "$HUMANIZER_PREDICTIONS" --metrics "$HUMANIZER_METRICS" \
    --server "$LLAMA_CPP/build/bin/llama-server" --server-log "$WORK/llama-server-eval.log" \
    --base-model "$HUMANIZER_BASE_MODEL" --base-revision "$HUMANIZER_BASE_REVISION"
fi
"${TRAINER[@]}" humanizer-audit "$HUMANIZER_PREDICTIONS" --output "$HUMANIZER_AUDIT_OUTPUT" \
  --workers "$HUMANIZER_WORKERS" --attempts "$HUMANIZER_ATTEMPTS" \
  --min-semantic-fidelity "$HUMANIZER_MIN_SEMANTIC_FIDELITY" --min-voice-match "$HUMANIZER_MIN_VOICE_MATCH" \
  --min-pass-rate "$HUMANIZER_MIN_PASS_RATE" --max-boilerplate-rate "$HUMANIZER_MAX_BOILERPLATE_RATE" \
  --min-voice-gain "$HUMANIZER_MIN_VOICE_GAIN" --min-semantic-delta "$HUMANIZER_MIN_SEMANTIC_DELTA"
HF_BIN="$VENV/bin/hf" "${TRAINER[@]}" humanizer-publish "$HUMANIZER_MODEL_DIR" \
  --repo "$HUMANIZER_HF_REPO" --metrics "$HUMANIZER_METRICS" \
  --audit "$HUMANIZER_AUDIT_OUTPUT" --preparation "$HUMANIZER_PREPARATION" \
  --output "$WORK/publication.json"
"$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"

cp "$WORK/preparation.json" "$WORK/metrics.json" "$WORK/training.json" "$WORK/audit.json" \
   "$WORK/predictions.jsonl" "$WORK/publication.json" \
   "$WORK/python-requirements.lock" "$OUT/"
cp -R "$WORK/student" "$OUT/adapter"

"${TRAINER[@]}" model-manifest --model humanizer --output-dir "$OUT"

echo "qualified private humanizer model and evidence staged in $OUT"
