#!/bin/sh
set -eu
work=/mnt/wisent-staging/oko-lifecycle-model-b5de55bd
exec "$HOME/.stado/bin/oko-lifecycle-prepare-split" \
  "$work/rereviewed-train-source.jsonl" \
  "$work/rereviewed-eval-source.jsonl" \
  2026-06-12 \
  "$work/rereviewed-train-day12.jsonl" \
  "$work/rereviewed-eval-day12.jsonl"
