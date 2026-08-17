#!/bin/sh
# Report one lifecycle training job's live state: process, accelerator, the
# informative log lines, and the held-out metrics once the trainer writes them.
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

printf '\n== log ==\n'
# Progress bars rewrite one line thousands of times; keep only the lines that
# carry information: the dataset summary, loss records, generation counters,
# the newest step, and anything that looks like a failure.
if [ -f train.log ]; then
    lines=$(mktemp)
    tr '\r' '\n' <train.log | grep -Ev '^[[:space:]]*$' >"$lines"
    grep -E "reviewed train rows|'loss'|'eval_loss'|held-out predictions|written|Error|Traceback|CUDA out of memory" \
        "$lines" | tail -n 20 || true
    printf -- '-- newest line --\n'
    tail -n 1 "$lines"
    rm -f "$lines"
else
    echo "no train.log yet"
fi

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
