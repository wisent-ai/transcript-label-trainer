#!/usr/bin/env python3
"""Report reviewed lifecycle rows by day and action without row content."""

import json
from collections import Counter, defaultdict
from pathlib import Path

work = Path("/mnt/wisent-staging/oko-lifecycle-model-b5de55bd")
counts = defaultdict(Counter)
review_models = Counter()
ids = set()
for name in ("reviewed-train.jsonl", "reviewed-eval.jsonl"):
    path = work / name
    print(json.dumps({"file": name, "bytes": path.stat().st_size if path.exists() else 0}))
    if not path.exists():
        continue
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row["id"] in ids:
                continue
            ids.add(row["id"])
            target = json.loads(
                next(message["content"] for message in row["messages"] if message["role"] == "assistant")
            )
            counts[row["split_day"]][target["action"]] += 1
            review_models[row.get("metadata", {}).get("review_model", "missing")] += 1
for day in sorted(counts):
    print(json.dumps({"day": day, "rows": sum(counts[day].values()), "actions": counts[day]}, sort_keys=True))
print(json.dumps({"review_models": review_models}, sort_keys=True))

metrics = json.loads((work / "metrics.json").read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "train_hard_example_counts": metrics.get("train_hard_example_counts"),
            "train_action_counts": metrics.get("train_action_counts"),
            "effective_train_action_counts": metrics.get("effective_train_action_counts"),
        },
        sort_keys=True,
    )
)
print(
    json.dumps(
        {
            key: metrics.get(key)
            for key in (
                "epochs",
                "learning_rate",
                "train_rows",
                "unique_train_rows",
                "train_action_counts",
                "effective_train_action_counts",
                "train_hard_example_counts",
                "eval_action_counts",
                "valid_json",
                "action_accuracy",
                "goal_ref_accuracy",
                "evidence_accuracy",
                "joint_accuracy",
                "finish_precision",
                "by_action",
            )
        },
        sort_keys=True,
    )
)
