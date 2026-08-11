#!/bin/sh
set -eu

work=$(
  /usr/bin/find /tmp -maxdepth 1 -type d -name 'wc-*' -printf '%T@ %p\n' \
    | /usr/bin/sort -nr \
    | /usr/bin/awk 'NR == 1 { print $2 }'
)
[ -n "$work" ]
model=$(/usr/bin/find "$work/output" -maxdepth 1 -type f -name '*.gguf' | /usr/bin/awk 'NR == 1')
[ -n "$model" ]
name=${model##*/}
size=$(/usr/bin/stat --format '%s' "$model")
sha256=$(/usr/bin/sha256sum "$model" | /usr/bin/cut -d ' ' -f 1)
parts="$work/output/large-output"
/bin/rm -rf "$parts"
/bin/mkdir -p "$parts"
/usr/bin/split -b 32M -d -a 3 "$model" "$parts/$name.part-"
count=$(/usr/bin/find "$parts" -maxdepth 1 -type f -name "$name.part-*" | /usr/bin/wc -l)
/usr/bin/python3 - "$parts/manifest.json" "$name" "$size" "$sha256" "$count" <<'PY'
import json
import sys
from pathlib import Path

path, name, size, sha256, count = sys.argv[1:]
Path(path).write_text(json.dumps({
    "filename": name,
    "size_bytes": int(size),
    "sha256": sha256,
    "part_bytes": 32 * 1024 * 1024,
    "part_count": int(count),
}, indent=2) + "\n", encoding="utf-8")
PY
/bin/sync "$parts"
/bin/rm "$model"
printf 'chunked %s bytes=%s sha256=%s parts=%s\n' "$name" "$size" "$sha256" "$count"
