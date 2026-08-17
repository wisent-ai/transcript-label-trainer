#!/bin/sh
# Report one lifecycle training job's live state: the unit, its checkpoints,
# the accelerator, the informative log lines, and the held-out metrics once the
# trainer writes them.
#
#   stado host run-helper <target> oko-lifecycle-train-report.sh --uuid <JOB_UUID>
set -eu

job="${1:?job uuid required}"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
unit="oko-lifecycle-train.service"
[ -d "$work" ] || { echo "no such job work directory: $work" >&2; exit 1; }
cd "$work"

printf 'unit=%s state=%s enabled=%s\n' \
    "$unit" \
    "$(systemctl is-active "$unit" 2>/dev/null || echo unknown)" \
    "$(systemctl is-enabled "$unit" 2>/dev/null || echo unknown)"
printf 'work=%s\n' "$work"
du -sh . 2>/dev/null || true

printf '\n== checkpoints ==\n'
if [ -d student-checkpoints ]; then
    ls -1dt student-checkpoints/checkpoint-* 2>/dev/null | head -n 3 || echo "none yet"
else
    echo "none yet"
fi

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
