#!/bin/sh
set -eu

job_id=${1:-b5de55bd}
work="/mnt/wisent-staging/oko-lifecycle-model-$job_id"
output="/tmp/wc-$job_id/output"

printf 'job_id=%s\n' "$job_id"
printf '%s\n' '=== matching processes ==='
/usr/bin/ps ax -o pid=,ppid=,etime=,stat=,command= \
  | /usr/bin/awk -v job="$job_id" 'index($0, job) || index($0, "lifecycle-audit") || index($0, "llama-quantize") { print }'

printf '%s\n' '=== work products ==='
for name in final-judge.json metrics.json predictions.jsonl python-requirements.lock oko-lifecycle-qwen3-4b-f16.gguf oko-lifecycle-qwen3-4b-q4_k_m.gguf; do
  path="$work/$name"
  if [ -s "$path" ]; then
    printf '%s %s\n' "$name" "$(/usr/bin/stat --format '%s' "$path")"
    case "$name" in
      final-judge.json|metrics.json) /bin/cat "$path";;
    esac
  else
    printf 'missing %s\n' "$name"
  fi
done

printf '%s\n' '=== staged output ==='
if [ -d "$output" ]; then
  /usr/bin/find "$output" -maxdepth 1 -type f -printf '%f %s\n' | /usr/bin/sort
  [ ! -s "$output/model-manifest.json" ] || /bin/cat "$output/model-manifest.json"
else
  printf 'missing %s\n' "$output"
fi
