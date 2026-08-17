#!/usr/bin/env python3
"""Print compact evidence for every rejected lifecycle prediction."""

from collections import Counter
import json
from pathlib import Path

work = Path("/mnt/wisent-staging/oko-lifecycle-model-b5de55bd")
judge = json.loads((work / "final-judge-local.json").read_text(encoding="utf-8"))
predictions = {}
with (work / "predictions.jsonl").open(encoding="utf-8") as handle:
    for raw in handle:
        if raw.strip():
            row = json.loads(raw)
            predictions[row["id"]] = row

wrong_ids = {
    record["id"]
    for record in judge["records"]
    if (record.get("decision") or {}).get("verdict") != "student-sensible"
    or (record.get("decision") or {}).get("dangerous_finish")
}
confusion = Counter()
error_dimensions = Counter()
wrong_by_target = Counter()
for row_id in sorted(wrong_ids):
    row = predictions[row_id]
    target = row["target"]
    prediction = row["prediction"] or {}
    wrong_by_target[target["action"]] += 1
    confusion[(target["action"], prediction.get("action", "invalid"))] += 1
    dimensions = tuple(
        name
        for name, field in (
            ("action", "action"),
            ("goal_ref", "goal_ref"),
            ("evidence", "lifecycle_evidence"),
            ("title", "title"),
        )
        if prediction.get(field) != target[field]
    )
    error_dimensions[dimensions or ("semantic_only",)] += 1

print(
    json.dumps(
        {
            "counts": judge["counts"],
            "thresholds": judge["thresholds"],
            "wrong_by_target": dict(sorted(wrong_by_target.items())),
            "confusion": {
                f"{target}->{prediction}": count
                for (target, prediction), count in sorted(confusion.items())
            },
            "error_dimensions": {
                "+".join(dimensions): count
                for dimensions, count in sorted(error_dimensions.items())
            },
        },
        sort_keys=True,
    )
)
for record in judge["records"]:
    decision = record.get("decision") or {}
    if decision.get("verdict") == "student-sensible" and not decision.get("dangerous_finish"):
        continue
    row = predictions[record["id"]]
    envelope = row["input"]
    selected_refs = {row["target"]["goal_ref"], (row["prediction"] or {}).get("goal_ref")}
    print(
        json.dumps(
            {
                "id": row["id"],
                "text": (envelope.get("text") or "")[:240],
                "target": row["target"],
                "prediction": row["prediction"],
                "selected_candidates": [
                    {
                        "ref": candidate.get("ref"),
                        "title": candidate.get("title"),
                        "same_session": candidate.get("same_session"),
                    }
                    for candidate in envelope.get("candidates", [])
                    if candidate.get("ref") in selected_refs
                ],
                "judge": decision,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
