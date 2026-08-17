#!/bin/sh
# Report one lifecycle training job's live state: process, accelerator, log
# tail, and the held-out metrics once the trainer has written them.
#
#   stado host run-helper <target> oko-lifecycle-train-report.sh --uuid <JOB_UUID>
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
[ -d "$work" ] || { echo "no such job work directory: $work" >&2; exit 1; }
cd "$work"

if [ -f train.pid ] && kill -0 "$(cat train.pid)" 2>/dev/null; then
    printf 'state=running pid=%s\n' "$(cat train.pid)"
else
    printf 'state=not-running\n'
fi

printf 'work=%s\n' "$work"
du -sh . 2>/dev/null || true

printf '\n== accelerator ==\n'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

printf '\n== log tail ==\n'
tail -n 40 train.log 2>/dev/null || echo "no train.log yet"

if [ -f metrics.json ]; then
    printf '\n== metrics ==\n'
    /usr/bin/python3 - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
report.pop("log_history", None)
print(json.dumps(report, indent=2, sort_keys=True))
PY
fi
