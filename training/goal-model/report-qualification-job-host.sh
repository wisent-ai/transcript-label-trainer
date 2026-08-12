#!/bin/sh
set -eu

job_id=${1:?usage: report-qualification-job-host JOB_ID}
work="/mnt/wd16tb/stado/jobs/jeden-goal-568ebd79663775c9"
printf 'job_id=%s\n' "$job_id"
printf '%s\n' '=== matching processes ==='
/usr/bin/ps ax -o pid=,ppid=,etime=,stat=,command= \
  | /usr/bin/awk -v job="$job_id" 'index($0, job) || index($0, "goal-audit") || index($0, "llama-quantize") { print }'
printf '%s\n' '=== work products ==='
for name in final-judge.json metrics.json predictions.jsonl; do
 path="$work/$name"
 if [ -s "$path" ]; then
  printf '%s %s\n' "$name" "$(/usr/bin/stat --format '%s' "$path")"
  [ "$name" != final-judge.json ] || /bin/cat "$path"
 else
  printf 'missing %s\n' "$name"
 fi
done
printf '%s\n' '=== staged output ==='
for output in /tmp/wc-*/output; do
 [ -d "$output" ] || continue
 manifest="$output/model-manifest.json"
 [ -s "$manifest" ] || continue
 /bin/cat "$manifest"
 /usr/bin/find "$output" -maxdepth 1 -type f -printf '%f %s\n' | /usr/bin/sort
done
