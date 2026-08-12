#!/bin/sh
set -eu

work=$(
  /usr/bin/find /mnt/wd16tb/stado/jobs -maxdepth 1 -type d -name 'jeden-goal-*' -printf '%T@ %p\n' \
    | /usr/bin/sort -nr \
    | /usr/bin/awk 'NR == 1 { print $2 }'
)
[ -n "$work" ] || {
  printf '%s\n' 'no Jeden goal-model work directory found' >&2
  exit 1
}
printf 'work=%s\n' "$work"
for name in metrics.json final-judge.json
do
  path="$work/$name"
  if [ -s "$path" ]; then
    printf '=== %s ===\n' "$name"
    /bin/cat "$path"
  else
    printf 'missing=%s\n' "$path"
  fi
done

printf '%s\n' '=== staged outputs ==='
found=false
for output in /tmp/wc-*/output
do
  [ -d "$output" ] || continue
  manifest="$output/model-manifest.json"
  [ -s "$manifest" ] || continue
  found=true
  printf 'output=%s\n' "$output"
  /bin/cat "$manifest"
  /usr/bin/find "$output" -maxdepth 1 -type f -printf '%f %s\n' | /usr/bin/sort
 done
"$found" || printf '%s\n' 'no staged model output found'
