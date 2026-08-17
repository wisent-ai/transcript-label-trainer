#!/bin/sh
set -eu

work=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
"$HOME/.stado/bin/oko-lifecycle-prepare-split" \
  "$work/reviewed-train.jsonl" \
  "$work/reviewed-eval.jsonl" \
  2026-06-12 \
  "$work/reviewed-train-day12.jsonl" \
  "$work/reviewed-eval-day12.jsonl"
"$HOME/.stado/bin/oko-lifecycle-generate-curriculum" \
  "$work/reviewed-train-day12.jsonl" \
  "$work/curriculum-train-raw.jsonl" \
  --per-family 64 \
  --seed 31
"$HOME/.stado/bin/oko-lifecycle-generate-curriculum" \
  "$work/reviewed-eval-day12.jsonl" \
  "$work/curriculum-eval-raw.jsonl" \
  --per-family 24 \
  --seed 97
