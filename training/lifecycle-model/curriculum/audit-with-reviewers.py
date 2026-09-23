#!/usr/bin/env python3
"""Judge held-out lifecycle decisions with independent reviewers, not the student.

`lifecycle-audit` routes this through Brama. When no operator-approved Brama
route answers, the judgement still has to come from somewhere that is not the
model under test, and parallel reviewer agents are that somewhere: each one is
handed the masked input, the reference decision and the student's decision, and
returns one verdict per row.

    split    predictions-gguf.jsonl -> batch files a reviewer can be handed
    verdict  reviewer outputs       -> final-judge.json with the same gate

The gate is the one the release publisher already enforces: more than two
percent semantically wrong student decisions fails, and a wrong decision that
declares an objective finished when it is not is reported separately because it
is the error that costs a user their goal state.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MAX_WRONG_RATE = 0.02
VERDICTS = {"right", "wrong"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split(arguments: argparse.Namespace) -> None:
    rows = read_rows(Path(arguments.predictions))
    directory = Path(arguments.out)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("batch-*.jsonl"):
        stale.unlink()
    size = max(1, arguments.per_batch)
    written = []
    for index in range(0, len(rows), size):
        chunk = rows[index : index + size]
        path = directory / f"batch-{index // size:02d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in chunk:
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "input": row["input"],
                            "reference": row["target"],
                            "student": row["prediction"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        written.append({"path": str(path), "rows": len(chunk)})
    print(json.dumps({"batches": written, "rows": len(rows)}, indent=2))


def verdict(arguments: argparse.Namespace) -> None:
    predictions = {row["id"]: row for row in read_rows(Path(arguments.predictions))}
    records: list[dict] = []
    seen: set[str] = set()
    for path in sorted(Path(arguments.reviews).glob("*.jsonl")):
        for row in read_rows(path):
            identifier = row["id"]
            if identifier not in predictions:
                raise SystemExit(f"reviewer judged an unknown row: {identifier}")
            if identifier in seen:
                raise SystemExit(f"row judged twice: {identifier}")
            if row.get("verdict") not in VERDICTS:
                raise SystemExit(f"{identifier} carries no usable verdict")
            seen.add(identifier)
            prediction = predictions[identifier]
            student = prediction.get("prediction") or {}
            records.append(
                {
                    "id": identifier,
                    "verdict": row["verdict"],
                    "reason": row.get("reason", ""),
                    "student_action": student.get("action"),
                    "reference_action": (prediction.get("target") or {}).get("action"),
                    "dangerous_finish": row["verdict"] == "wrong"
                    and student.get("action") == "finishGoal",
                }
            )
    missing = sorted(set(predictions) - seen)
    if missing:
        raise SystemExit(f"{len(missing)} rows were never judged, first {missing[0]}")

    total = len(records)
    wrong = sum(1 for record in records if record["verdict"] == "wrong")
    dangerous = sum(1 for record in records if record["dangerous_finish"])
    rate = wrong / total if total else 1.0
    report = {
        "contract": "oko-goal-lifecycle-v1",
        "review_model": arguments.review_model,
        "rows": total,
        "wrong": wrong,
        "wrong_rate": rate,
        "dangerous_finish": dangerous,
        "maximum_wrong_rate": MAX_WRONG_RATE,
        "passed": rate <= MAX_WRONG_RATE and dangerous == 0,
        "wrong_by_student_action": dict(
            Counter(
                record["student_action"] for record in records if record["verdict"] == "wrong"
            )
        ),
        "records": records,
    }
    Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    splitter = subparsers.add_parser("split", help="write reviewer batches")
    splitter.add_argument("predictions")
    splitter.add_argument("--out", default="/tmp/oko-audit-batches")
    splitter.add_argument("--per-batch", type=int, default=60)
    splitter.set_defaults(handler=split)

    judge = subparsers.add_parser("verdict", help="aggregate reviewer verdicts")
    judge.add_argument("predictions")
    judge.add_argument("--reviews", default="/tmp/oko-audit-out")
    judge.add_argument("--output", required=True)
    judge.add_argument(
        "--review-model",
        required=True,
        help="exact reviewer identity recorded in the verdict",
    )
    judge.set_defaults(handler=verdict)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
