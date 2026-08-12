#!/bin/sh
set -eu
output=/tmp/jeden-goal-qualified-v6
rm -rf "$output"
mkdir -p "$output"
OUTPUT_DIR="$output" /root/.stado/bin/stage-qualified-host
for path in "$output"/*
do
  name=${path##*/}
  /root/.stado/bin/stado storage put "stado://probierz/artifacts/models/jeden/goal-qwen3-4b/v6/$name" "$path"
done
/root/.stado/bin/stado storage stat 'stado://probierz/artifacts/models/jeden/goal-qwen3-4b/v6/model-manifest.json'
