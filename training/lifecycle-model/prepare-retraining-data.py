#!/usr/bin/env python3
"""Build a fresh day-held-out lifecycle split from reviewed rows."""

import json
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) != 6:
    raise SystemExit(
        "usage: prepare-retraining-data.py OLD_TRAIN OLD_EVAL EVAL_DAY NEW_TRAIN NEW_EVAL"
    )

old_train, old_eval, eval_day, new_train, new_eval = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    Path(sys.argv[4]),
    Path(sys.argv[5]),
)

rows_by_id = {}
for source in (old_train, old_eval):
    for raw in source.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        previous = rows_by_id.setdefault(row["id"], row)
        if previous != row:
            raise SystemExit(f"conflicting reviewed rows for {row['id']}")

splits = {"train": [], "eval": []}
counts = {"train": Counter(), "eval": Counter()}
for row in sorted(rows_by_id.values(), key=lambda value: value["id"]):
    split = "eval" if row["split_day"] == eval_day else "train"
    row["split"] = split
    target = json.loads(
        next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    )
    splits[split].append(row)
    counts[split][target["action"]] += 1

required_actions = {"startGoal", "continueCurrent", "finishGoal", "ignore"}
for split in ("train", "eval"):
    missing = required_actions - counts[split].keys()
    if missing:
        raise SystemExit(f"{split} split is missing actions: {', '.join(sorted(missing))}")

for path, split in ((new_train, "train"), (new_eval, "eval")):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits[split]),
        encoding="utf-8",
    )

print(json.dumps({
    "eval_day": eval_day,
    "train_rows": len(splits["train"]),
    "eval_rows": len(splits["eval"]),
    "train_actions": dict(sorted(counts["train"].items())),
    "eval_actions": dict(sorted(counts["eval"].items())),
}, sort_keys=True))
