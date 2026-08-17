#!/usr/bin/env python3
"""Measure reviewed lifecycle evidence against candidate status invariants."""

import json
from collections import Counter
from pathlib import Path

work = Path("/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd")
for name in ("reviewed-train.jsonl", "reviewed-eval.jsonl"):
    evidence_status = Counter()
    finish_status = Counter()
    start_titles = Counter()
    with (work / name).open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            target = json.loads(
                next(item["content"] for item in row["messages"] if item["role"] == "assistant")
            )
            envelope = json.loads(
                next(item["content"] for item in row["messages"] if item["role"] == "user")
            )
            candidate = next(
                (item for item in envelope["candidates"] if item["ref"] == target["goal_ref"]),
                {},
            )
            status = candidate.get("status", "missing")
            if target["lifecycle_evidence"] == "explicit_open":
                evidence_status[status] += 1
            if target["action"] == "finishGoal":
                finish_status[status] += 1
            if target["action"] == "startGoal":
                start_titles[len(target["title"].split())] += 1
    print(
        json.dumps(
            {
                "file": name,
                "explicit_open_candidate_status": evidence_status,
                "finish_candidate_status": finish_status,
                "start_title_word_counts": start_titles,
            },
            sort_keys=True,
        )
    )
