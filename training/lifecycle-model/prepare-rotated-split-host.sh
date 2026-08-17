#!/bin/sh
set -eu
work=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
exec "$work/venv/bin/python" \
  "$work/audit-source/training/lifecycle-model/prepare-retraining-data.py" \
  "$work/reviewed-train.jsonl" \
  "$work/reviewed-eval.jsonl" \
  2026-06-11 \
  "$work/reviewed-train-rotated.jsonl" \
  "$work/reviewed-eval-rotated.jsonl"
