#!/usr/bin/env python3
"""Print held-out goal predictions rejected by the final judge."""

import json
from pathlib import Path

roots = sorted(
    Path("/mnt/wd16tb/stado/jobs").glob("jeden-goal-*"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not roots:
    raise SystemExit("no Jeden goal-model work directory found")
work = roots[0]
judge_path = work / "final-judge-prompt-v2.json"
predictions_path = work / "predictions-prompt-v2.jsonl"
if not judge_path.is_file():
    judge_path = work / "final-judge.json"
    predictions_path = work / "predictions.jsonl"
with judge_path.open(encoding="utf-8") as source:
    judge = json.load(source)
with predictions_path.open(encoding="utf-8") as source:
    predictions = {
        record["session_id"]: record
        for line in source
        if (record := json.loads(line))
    }
for audit in judge.get("records", []):
    if audit.get("verdict") == "both-sensible":
        continue
    record = predictions.get(audit.get("session_id"), {})
    print(json.dumps({"verdict": audit.get("verdict"), **record}, sort_keys=True))
