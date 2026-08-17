#!/bin/sh
set -eu

work=/mnt/wisent-staging/oko-lifecycle-model-b5de55bd
printf '%s\n' '=== status ==='
/bin/cat "$work/retrain-status.json"
printf '%s\n' '=== log end ==='
/usr/bin/tac "$work/retrain.log" | /usr/bin/sed -n '1,16p' | /usr/bin/tac | /usr/bin/cut -c 1-1000
printf '%s\n' '=== gpu ==='
/usr/bin/nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '=== retrain process ==='
/usr/bin/ps ax -o pid=,etime=,stat=,command= \
  | /usr/bin/awk 'index($0, "train.py") || index($0, "oko-lifecycle-retrain") { print }' \
  | /usr/bin/cut -c 1-1000
