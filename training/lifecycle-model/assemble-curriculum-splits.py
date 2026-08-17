#!/usr/bin/env python3
"""Assemble real and independently reviewed lifecycle curriculum splits."""

import argparse
import json
from collections import Counter
from pathlib import Path


ACTIONS = {"startGoal", "continueCurrent", "finishGoal", "ignore"}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        target_for(row)
    return rows


def target_for(row):
    assistants = [message for message in row["messages"] if message["role"] == "assistant"]
    if len(assistants) != 1:
        raise ValueError(f"{row['id']} must contain one reviewed assistant decision")
    target = json.loads(assistants[0]["content"])
    if target.get("action") not in ACTIONS:
        raise ValueError(f"{row['id']} has invalid reviewed action")
    target["title"] = ""
    assistants[0]["content"] = json.dumps(
        target, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return target


def accepted_curriculum(rows):
    accepted = []
    rejected = Counter()
    for row in rows:
        intended = row.get("metadata", {}).get("intended_action")
        reviewed = target_for(row)["action"]
        if intended == reviewed:
            accepted.append(row)
        else:
            rejected[f"{intended}->{reviewed}"] += 1
    return accepted, rejected


def write_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)


def action_counts(rows):
    return Counter(target_for(row)["action"] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_train")
    parser.add_argument("base_eval")
    parser.add_argument("reviewed_train_curriculum")
    parser.add_argument("reviewed_eval_curriculum")
    parser.add_argument("output_train")
    parser.add_argument("output_eval")
    parser.add_argument("--minimum-eval-curriculum-per-action", type=int, default=16)
    args = parser.parse_args()

    base_train = read_rows(args.base_train)
    base_eval = read_rows(args.base_eval)
    train_curriculum, train_rejected = accepted_curriculum(read_rows(args.reviewed_train_curriculum))
    eval_curriculum, eval_rejected = accepted_curriculum(read_rows(args.reviewed_eval_curriculum))

    eval_curriculum_counts = action_counts(eval_curriculum)
    missing = {
        action: args.minimum_eval_curriculum_per_action - eval_curriculum_counts[action]
        for action in sorted(ACTIONS)
        if eval_curriculum_counts[action] < args.minimum_eval_curriculum_per_action
    }
    if missing:
        raise SystemExit(f"reviewed evaluation curriculum is underrepresented: {missing}")

    train_rows = base_train + train_curriculum
    eval_rows = base_eval + eval_curriculum
    train_ids = [row["id"] for row in train_rows]
    eval_ids = [row["id"] for row in eval_rows]
    if len(train_ids) != len(set(train_ids)):
        raise SystemExit("assembled training split contains duplicate ids")
    if len(eval_ids) != len(set(eval_ids)):
        raise SystemExit("assembled evaluation split contains duplicate ids")
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise SystemExit(f"assembled splits overlap at {next(iter(overlap))}")

    write_rows(args.output_train, train_rows)
    write_rows(args.output_eval, eval_rows)
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_actions": action_counts(train_rows),
                "eval_actions": action_counts(eval_rows),
                "accepted_train_curriculum": len(train_curriculum),
                "accepted_eval_curriculum": len(eval_curriculum),
                "rejected_train_curriculum": train_rejected,
                "rejected_eval_curriculum": eval_rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
