#!/usr/bin/env python3
"""Summarize goal candidate training curves and repeated audit outcomes."""

import json
from pathlib import Path

work = Path("/mnt/wd16tb/stado/jobs/jeden-goal-568ebd79663775c9")
metrics = json.loads((work / "metrics.json").read_text())
print(json.dumps({
    "base_model": metrics.get("base_model"),
    "epochs": metrics.get("epochs"),
    "learning_rate": metrics.get("learning_rate"),
    "exact_match": metrics.get("exact_match"),
    "evaluations": [
        item for item in metrics.get("log_history", [])
        if "eval_loss" in item
    ],
}, sort_keys=True))
for output in sorted(Path("/tmp").glob("wc-*/output"), key=lambda path: path.stat().st_mtime):
    judge_path = output / "final-judge.json"
    if not judge_path.is_file():
        continue
    judge = json.loads(judge_path.read_text())
    print(json.dumps({
        "output": str(output),
        "mtime": output.stat().st_mtime,
        "passed": judge.get("passed"),
        "counts": judge.get("counts"),
        "created_at": judge.get("created_at"),
    }, sort_keys=True))
