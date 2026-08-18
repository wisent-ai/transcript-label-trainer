#!/bin/sh
# Launch lifecycle label-model training on this host's accelerator, as a unit
# that survives a power cut: on 2026-08-17 a tripped breaker killed a detached
# run that kept no checkpoint, and roughly two hundred optimizer steps were
# lost. The trainer now checkpoints and resumes, and this unit restarts it on
# failure and on boot.
#
# Installed and started through Stado:
#   stado host install-helper <target> training/lifecycle-model/train.py oko-lifecycle-train.py
#   stado host install-helper <target> training/lifecycle-model/lifecycle-system-prompt.txt lifecycle-system-prompt.txt
#   stado host install-helper <target> training/lifecycle-model/launch-training-host.sh oko-lifecycle-train-launch.sh
#   stado host run-helper <target> oko-lifecycle-train-launch.sh --uuid <JOB_UUID>
#
# The single argument is the job UUID: run-helper refuses operator words, so
# every other input is either baked in here or read from immutable Stado
# release objects, which are bearer-free to read.
set -eu

job="${1:?job uuid required}"
stado="$HOME/.stado/bin/stado"
trainer="$HOME/.stado/bin/oko-lifecycle-train.py"
curriculum="stado://releases/oko/lifecycle-curriculum/20260818"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"
unit="oko-lifecycle-train.service"
unit_path="/etc/systemd/system/$unit"

[ -x "$stado" ] || { echo "stado CLI absent at $stado" >&2; exit 1; }
[ -f "$trainer" ] || { echo "trainer absent at $trainer" >&2; exit 1; }
case "$work" in
    /mnt/wisent-training/*) ;;
    *) echo "refusing work root outside the declared training mount" >&2; exit 1 ;;
esac
mountpoint -q /mnt/wisent-training || {
    echo "/mnt/wisent-training is not a mount point; refusing to fill the root volume" >&2
    exit 1
}

mkdir -p "$work"
cd "$work"

if [ ! -s reviewed-train.jsonl ]; then
    "$stado" storage get "$curriculum/reviewed-train.jsonl" reviewed-train.jsonl
fi
if [ ! -s reviewed-eval.jsonl ]; then
    "$stado" storage get "$curriculum/reviewed-eval.jsonl" reviewed-eval.jsonl
fi

cat >"$unit_path" <<UNIT
[Unit]
Description=Oko goal-lifecycle label-model training ($job)
After=network-online.target mnt-wisent\\x2dtraining.mount
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$work
Environment=HF_HOME=/mnt/wisent-staging/hf-cache
Environment=CUDA_VISIBLE_DEVICES=0
Environment=LIFECYCLE_STUDENT_MODEL=Qwen/Qwen3-4B
Environment=LIFECYCLE_STUDENT_EPOCHS=3
Environment=LIFECYCLE_STUDENT_LR=1e-5
Environment=LIFECYCLE_STUDENT_SAVE_STEPS=50
Environment=LIFECYCLE_TRAIN_DATASET=$work/reviewed-train.jsonl
Environment=LIFECYCLE_EVAL_DATASET=$work/reviewed-eval.jsonl
ExecStart=/usr/bin/python3 $trainer
StandardOutput=append:$work/train.log
StandardError=append:$work/train.log
Restart=on-failure
RestartSec=30
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
UNIT

mkdir -p /mnt/wisent-staging/hf-cache
systemctl daemon-reload
systemctl enable "$unit" >/dev/null 2>&1
systemctl restart "$unit"
sleep 3
systemctl is-active "$unit" >/dev/null || {
    echo "unit failed to start" >&2
    systemctl status "$unit" --no-pager --lines 20 >&2
    exit 1
}

printf 'unit=%s state=%s work=%s log=%s\n' \
    "$unit" "$(systemctl is-active "$unit")" "$work" "$work/train.log"
ls -d "$work/student-checkpoints/checkpoint-"* 2>/dev/null || echo "no checkpoint yet"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
