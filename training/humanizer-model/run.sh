#!/usr/bin/env bash
# Stado GPU job: curate, train, audit, and privately publish Echo's humanizer.
set -euo pipefail

: "${HUMANIZER_HF_REPO:?Set HUMANIZER_HF_REPO to the private Hugging Face destination}"
: "${HUMANIZER_WORKERS:?Set HUMANIZER_WORKERS to the parallel Brama calls humanizer-model was given}"
: "${HUMANIZER_ATTEMPTS:?Set HUMANIZER_ATTEMPTS to the Brama attempts humanizer-model was given}"
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
  'peft>=0.17,<1' 'bitsandbytes>=0.46,<1' 'huggingface-hub>=0.36,<1'

export HUMANIZER_TARGETS="$WORK/targets.jsonl"
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

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/cargo-target}"
TRAINER=("$HOME/.cargo/bin/cargo" run --manifest-path "$ROOT/Cargo.toml" --locked --release --)

"${TRAINER[@]}" humanizer-prepare "$HUMANIZER_TARGETS" --output-dir "$WORK" \
  --workers "$HUMANIZER_WORKERS" --attempts "$HUMANIZER_ATTEMPTS"
"$VENV/bin/python" "$ROOT/training/humanizer-model/train.py"
"${TRAINER[@]}" humanizer-audit "$HUMANIZER_PREDICTIONS" --output "$HUMANIZER_AUDIT_OUTPUT" \
  --workers "$HUMANIZER_WORKERS" --attempts "$HUMANIZER_ATTEMPTS"
HF_BIN="$VENV/bin/hf" "${TRAINER[@]}" humanizer-publish "$HUMANIZER_MODEL_DIR" \
  --repo "$HUMANIZER_HF_REPO" --metrics "$HUMANIZER_METRICS" \
  --audit "$HUMANIZER_AUDIT_OUTPUT" --preparation "$HUMANIZER_PREPARATION" \
  --output "$WORK/publication.json"
"$VENV/bin/python" -m pip freeze > "$WORK/python-requirements.lock"

cp "$WORK/preparation.json" "$WORK/metrics.json" "$WORK/audit.json" \
   "$WORK/predictions.jsonl" "$WORK/publication.json" \
   "$WORK/python-requirements.lock" "$OUT/"
cp -R "$WORK/student" "$OUT/adapter"

"${TRAINER[@]}" model-manifest --model humanizer --output-dir "$OUT"

echo "qualified private humanizer model and evidence staged in $OUT"
