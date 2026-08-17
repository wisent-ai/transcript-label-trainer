#!/bin/sh
set -eu
work=/mnt/wisent-staging/oko-lifecycle-model-b5de55bd
[ -f "$work/source-rereview-status.json" ] && /usr/bin/cat "$work/source-rereview-status.json"
for file in rereviewed-train-source.jsonl rereviewed-eval-source.jsonl; do
  if [ -f "$work/$file" ]; then
    printf '%s ' "$file"
    /usr/bin/wc -l < "$work/$file"
  fi
done
[ -f "$work/source-rereview.log" ] && /usr/bin/tail -n 8 "$work/source-rereview.log"
