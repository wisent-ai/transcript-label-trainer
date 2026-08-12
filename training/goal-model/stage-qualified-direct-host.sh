#!/bin/sh
set -eu
output=/tmp/jeden-goal-qualified-v6
rm -rf "$output"
mkdir -p "$output"
OUTPUT_DIR="$output" /root/.stado/bin/stage-qualified-host
/root/.stado/bin/stado storage put-tree 'stado://probierz/artifacts/models/jeden/goal-qwen3-4b/v6/' "$output"
/root/.stado/bin/stado storage inspect 'stado://probierz/artifacts/models/jeden/goal-qwen3-4b/v6/model-manifest.json'
