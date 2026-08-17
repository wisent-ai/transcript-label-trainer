#!/bin/sh
set -eu
work=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
[ -f "$work/curriculum-review-status.json" ] && /usr/bin/cat "$work/curriculum-review-status.json"
for file in curriculum-train-reviewed.jsonl curriculum-eval-reviewed.jsonl; do
  if [ -f "$work/$file" ]; then
    printf '%s ' "$file"
    /usr/bin/wc -l < "$work/$file"
  fi
done
[ -f "$work/curriculum-review.log" ] && /usr/bin/tail -n 8 "$work/curriculum-review.log"
