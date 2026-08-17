#!/bin/sh
# Launch lifecycle label-model training on this host's accelerator.
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
curriculum="stado://releases/oko/lifecycle-curriculum/20260817"
work="/mnt/wisent-training/stado/training/oko-lifecycle-$job"

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

if [ -f train.pid ] && kill -0 "$(cat train.pid)" 2>/dev/null; then
    echo "training already running pid=$(cat train.pid) work=$work"
    exit 0
fi

HF_HOME=/mnt/wisent-staging/hf-cache
export HF_HOME
mkdir -p "$HF_HOME"
CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES
LIFECYCLE_STUDENT_MODEL=Qwen/Qwen3-4B
export LIFECYCLE_STUDENT_MODEL
LIFECYCLE_STUDENT_EPOCHS=3
export LIFECYCLE_STUDENT_EPOCHS
LIFECYCLE_STUDENT_LR=1e-5
export LIFECYCLE_STUDENT_LR
LIFECYCLE_TRAIN_DATASET="$work/reviewed-train.jsonl"
export LIFECYCLE_TRAIN_DATASET
LIFECYCLE_EVAL_DATASET="$work/reviewed-eval.jsonl"
export LIFECYCLE_EVAL_DATASET

setsid nohup /usr/bin/python3 "$trainer" >>train.log 2>&1 &
echo $! >train.pid
sleep 2
printf 'started pid=%s work=%s log=%s\n' "$(cat train.pid)" "$work" "$work/train.log"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
