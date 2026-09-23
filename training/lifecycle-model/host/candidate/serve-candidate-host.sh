#!/bin/sh
# Expose one converted lifecycle candidate on this host's loopback for a bounded
# window, so the machine that will serve the model can fetch the exact bytes
# through Stado's encrypted forwarding channel:
#
#   stado host run-helper <target> oko-lifecycle-serve.sh --uuid <JOB_UUID>
#   stado host forward-remote <target> oko-lifecycle-fetch \
#       --remote-port 18790 --local-port 18790
#
# The release channel is not used for this hop: `release_api.publishers`
# declares the `oko/` publisher only on the control-plane machine, and a burst
# training host has no business holding a release-publishing credential.
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
port=18790
window=3600

[ -d "$work" ] || { echo "no such job work directory: $work" >&2; exit 1; }
cd "$work"
[ -s oko-lifecycle-qwen3-4b-q4_k_m.gguf ] || { echo "no converted candidate present" >&2; exit 1; }

if [ -f serve.pid ] && kill -0 "$(cat serve.pid)" 2>/dev/null; then
    kill "$(cat serve.pid)" 2>/dev/null || true
    sleep 1
fi

setsid nohup timeout "$window" /usr/bin/python3 -m http.server "$port" \
    --bind 127.0.0.1 --directory "$work" >serve.log 2>&1 &
echo $! >serve.pid
sleep 2
kill -0 "$(cat serve.pid)" 2>/dev/null || { echo "server failed to start" >&2; cat serve.log >&2; exit 1; }

/usr/bin/python3 - <<'PY'
import hashlib
import json
from pathlib import Path

name = "oko-lifecycle-qwen3-4b-q4_k_m.gguf"
value = hashlib.sha256()
with Path(name).open("rb") as source:
    while chunk := source.read(8 * 1024 * 1024):
        value.update(chunk)
print(
    json.dumps(
        {
            "filename": name,
            "bytes": Path(name).stat().st_size,
            "sha256": value.hexdigest(),
            "port": 18790,
            "window_seconds": 3600,
            "files": sorted(
                path.name
                for path in Path().iterdir()
                if path.is_file() and path.suffix in {".gguf", ".jsonl", ".json", ".lock"}
            ),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
