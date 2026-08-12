#!/bin/sh
set -eu

source_work=/mnt/wd16tb/stado/jobs/jeden-goal-568ebd79663775c9
work=/mnt/wd16tb/stado/jobs/jeden-goal-prompt-v3
venv="$source_work/venv"
train=/root/.stado/files/goal-train.py
prepare=/root/.stado/files/goal-prepare-retraining-data.py
[ -x "$venv/bin/python" ]
[ -s "$source_work/reviewed-goals.jsonl" ]
[ -s "$train" ]
[ -s "$prepare" ]
mkdir -p "$work"
"$venv/bin/python" "$prepare" \
  "$source_work/reviewed-goals.jsonl" "$work/reviewed-goals.jsonl"
cd "$work"
export GOAL_DATASET="$work/reviewed-goals.jsonl"
export GOAL_STUDENT_MODEL=Qwen/Qwen3-4B
export GOAL_STUDENT_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
export GOAL_STUDENT_EPOCHS=3
exec "$venv/bin/python" "$train"
