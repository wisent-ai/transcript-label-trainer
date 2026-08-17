#!/bin/sh
set -eu

work=/mnt/wisent-staging/oko-lifecycle-model-b5de55bd
exec "$HOME/.stado/bin/oko-lifecycle-assemble-curriculum" \
  "$work/rereviewed-train-day12.jsonl" \
  "$work/rereviewed-eval-day12.jsonl" \
  "$work/curriculum-train-reviewed.jsonl" \
  "$work/curriculum-eval-reviewed.jsonl" \
  "$work/reviewed-train-curriculum.jsonl" \
  "$work/reviewed-eval-curriculum.jsonl" \
  --minimum-eval-curriculum-per-action 16
