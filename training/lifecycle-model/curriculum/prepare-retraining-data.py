#!/usr/bin/env python3
"""Build a fresh day-held-out lifecycle split from reviewed rows."""

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

USAGE = (
    "usage: prepare-retraining-data.py OLD_TRAIN OLD_EVAL EVAL_DAY "
    "NEW_TRAIN NEW_EVAL [--goal-dataset REVIEWED_GOALS_JSONL]"
)
GOAL_AUGMENTATION_LIMIT = 256

if len(sys.argv) not in {6, 8} or (len(sys.argv) == 8 and sys.argv[6] != "--goal-dataset"):
    raise SystemExit(USAGE)

old_train, old_eval, eval_day, new_train, new_eval = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    Path(sys.argv[4]),
    Path(sys.argv[5]),
)
goal_dataset = Path(sys.argv[7]) if len(sys.argv) == 8 else None


def target_for(row):
    return json.loads(
        next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    )


def envelope_for(row):
    return json.loads(
        next(message["content"] for message in row["messages"] if message["role"] == "user")
    )


def normalized_message(value):
    return " ".join(value.casefold().split())


def lifecycle_features(message):
    words = message.split()
    lowered = message.casefold()
    return {
        "turn_index": 0,
        "is_first_turn_in_session": True,
        "word_count": len(words),
        "is_short_prompt": len(words) <= 12,
        "is_question": "?" in message,
        "has_new_objective_language": True,
        "has_status_or_meta_language": any(
            token in lowered for token in ("status", "progress", "what happened")
        ),
        "has_correction_or_rejection_language": any(
            token in lowered for token in ("wrong", "instead", "do not", "don't")
        ),
        "has_completion_or_ack_language": any(
            token in lowered for token in ("done", "finished", "thanks", "thank you")
        ),
    }


rows_by_id = {}
for source in (old_train, old_eval):
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
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
    splits[split].append(row)
    counts[split][target_for(row)["action"]] += 1

augmented_start_goals = 0
if goal_dataset is not None:
    start_template = next(
        row for row in splits["train"] if target_for(row)["action"] == "startGoal"
    )
    template_envelope = envelope_for(start_template)
    distractor = next(
        candidate
        for row in splits["train"]
        for candidate in envelope_for(row)["candidates"]
        if candidate["ref"] != "NEW_GOAL"
    )
    known_messages = {
        normalized_message(envelope_for(row)["text"])
        for rows in splits.values()
        for row in rows
    }
    reviewed_goals = []
    with goal_dataset.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            reviewed = json.loads(raw)
            message = " ".join(str(reviewed.get("message", "")).split())
            title = " ".join(str(reviewed.get("goal", "")).split())
            title_words = title.split()
            if (
                not message
                or not 3 <= len(title_words) <= 7
                or len(title) > 100
                or normalized_message(message) in known_messages
            ):
                continue
            known_messages.add(normalized_message(message))
            reviewed_goals.append((message, title, reviewed))
    reviewed_goals.sort(
        key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest()
    )
    for index, (message, title, reviewed) in enumerate(
        reviewed_goals[:GOAL_AUGMENTATION_LIMIT], 1
    ):
        prompt_id = f"goal-augmentation-{index:04d}"
        envelope = copy.deepcopy(template_envelope)
        envelope.update(
            {
                "prompt_id": prompt_id,
                "member_id": "curated-goal-model",
                "provider": reviewed.get("runtime", "unknown"),
                "session_id": reviewed.get("session_id", prompt_id),
                "turn_index": 0,
                "timestamp": "",
                "local_day": "curated-goal-model",
                "text": message,
                "lifecycle_features": lifecycle_features(message),
                "recent_session_prompts": [],
                "recent_member_prompts": [],
                "candidates": [
                    {
                        **copy.deepcopy(distractor),
                        "ref": "C1",
                        "same_session": False,
                        "is_last_member_goal": False,
                        "score": 0.0,
                    },
                    {"ref": "NEW_GOAL", "title": "Create a new goal"},
                ],
            }
        )
        target = {
            "action": "startGoal",
            "goal_ref": "NEW_GOAL",
            "title": title,
            "lifecycle_evidence": "none",
        }
        row = {
            "id": prompt_id,
            "split_day": "curated-goal-model",
            "split": "train",
            "messages": [
                copy.deepcopy(start_template["messages"][0]),
                {"role": "user", "content": json.dumps(envelope, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ],
            "metadata": {
                "source": "reviewed-goal-model",
                "source_session_id": reviewed.get("session_id"),
                "reviewed_by": reviewed.get("reviewed_by"),
            },
        }
        splits["train"].append(row)
        counts["train"]["startGoal"] += 1
        augmented_start_goals += 1

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

print(
    json.dumps(
        {
            "eval_day": eval_day,
            "goal_dataset": str(goal_dataset) if goal_dataset else None,
            "augmented_start_goals": augmented_start_goals,
            "train_rows": len(splits["train"]),
            "eval_rows": len(splits["eval"]),
            "train_actions": dict(sorted(counts["train"].items())),
            "eval_actions": dict(sorted(counts["eval"].items())),
        },
        sort_keys=True,
    )
)
