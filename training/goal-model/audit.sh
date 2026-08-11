#!/usr/bin/env bash
# Stado CPU job: audit one staged goal-model candidate after its GPU slot is free.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_URI="${1:?usage: audit.sh MODEL_ARTIFACT_URI}"
WORK="${GOAL_AUDIT_WORK_DIR:-/tmp/jeden-goal-audit-${WC_JOB_ID:?WC_JOB_ID is required}}"
OUT="/tmp/wc-${WC_JOB_ID}/output"
STADO="${STADO_BIN:-$HOME/.stado/bin/stado}"
mkdir -p "$WORK" "$OUT"

"$STADO" storage get "$MODEL_URI/predictions.jsonl" "$WORK/predictions.jsonl"
"$STADO" storage get "$MODEL_URI/model-manifest.json" "$WORK/model-manifest.json"

audit_args=(
  goal-audit "$WORK/predictions.jsonl"
  --output "$WORK/final-judge.json"
)
if [ -n "${TLT_BRAMA_MODEL:-}" ]; then
  audit_args+=(--brama-model "$TLT_BRAMA_MODEL")
else
  audit_args+=(--best)
fi

set +e
cargo run --manifest-path "$ROOT/Cargo.toml" --release -- "${audit_args[@]}"
audit_exit=$?
set -e
[ -s "$WORK/final-judge.json" ]
cp "$WORK/final-judge.json" "$OUT/"
printf '%s\n' "$audit_exit" > "$OUT/audit-exit-code"

MANIFEST="$WORK/model-manifest.json" JUDGE="$WORK/final-judge.json" OUT="$OUT" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8"))
judge_path = Path(os.environ["JUDGE"])
judge = json.loads(judge_path.read_text(encoding="utf-8"))
manifest["qualified"] = judge.get("passed") is True
manifest["review_model"] = judge.get("review_model")
manifest.setdefault("files", {})["final-judge.json"] = {
    "bytes": judge_path.stat().st_size,
    "sha256": hashlib.sha256(judge_path.read_bytes()).hexdigest(),
}
Path(os.environ["OUT"], "model-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

if [ "$audit_exit" -eq 0 ]; then
  echo "goal model qualified"
else
  echo "goal model rejected by final audit"
fi
