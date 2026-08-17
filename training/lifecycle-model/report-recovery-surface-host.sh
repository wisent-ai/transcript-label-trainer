#!/bin/sh
set -eu

job_id=b5de55bd
for work in \
  "/mnt/wd16tb/stado/jobs/oko-lifecycle-model-$job_id" \
  "/mnt/wd16tb/wisent-staging/oko-lifecycle-model-$job_id"
do
  [ -d "$work" ] || continue
  printf 'work=%s\n' "$work"
  for relative in \
    predictions.jsonl metrics.json final-judge.json \
    cargo-target/release/transcript-label-trainer \
    oko-lifecycle-qwen3-4b-q4_k_m.gguf
  do
    path="$work/$relative"
    if [ -s "$path" ]; then
      printf '%s %s bytes executable=%s\n' "$relative" "$(/usr/bin/stat --format '%s' "$path")" "$([ -x "$path" ] && printf yes || printf no)"
    else
      printf 'missing %s\n' "$relative"
    fi
  done
  for path in "$work"/*-audit-status.json "$work"/retrain-status.json "$work"/final-judge-*.json; do
    [ ! -s "$path" ] || {
      printf '=== %s summary ===\n' "$(/usr/bin/basename "$path")"
      /usr/bin/jq '{
        passed,
        review_model,
        state,
        exit_code,
        output_bytes,
        counts,
        thresholds,
        source_revision,
        old_model_sha256,
        new_model_sha256,
        error,
        top_level_keys: keys
      }' "$path"
    }
  done
  for path in "$work"/*-audit.log; do
    [ ! -s "$path" ] || {
      printf '=== %s start ===\n' "$(/usr/bin/basename "$path")"
      /usr/bin/sed -n '1,30p' "$path" | /usr/bin/cut -c 1-1000
    }
  done
  if [ -s "$work/retrain.log" ]; then
    printf '%s\n' '=== retrain.log end ==='
    /usr/bin/tac "$work/retrain.log" \
      | /usr/bin/sed -n '1,20p' \
      | /usr/bin/tac \
      | /usr/bin/cut -c 1-1000
  fi
done

for root in "/tmp/wc-$job_id" "$HOME/.stado/jobs/$job_id"; do
  [ -d "$root" ] || continue
  printf 'job_root=%s\n' "$root"
  for candidate in "$root/Cargo.toml" "$root/repo/Cargo.toml" "$root/work/Cargo.toml"; do
    [ ! -s "$candidate" ] || printf 'source=%s\n' "$candidate"
  done
done
