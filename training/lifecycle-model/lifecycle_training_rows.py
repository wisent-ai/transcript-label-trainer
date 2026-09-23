"""Lifecycle trainer settings and the rows it trains on, shared with train.py."""

import json
import os
import random
from collections import Counter, defaultdict

SEED = 29
BASE_MODEL = os.environ.get("LIFECYCLE_STUDENT_MODEL", "Qwen/Qwen3-4B")
BASE_REVISION = os.environ.get(
    "LIFECYCLE_STUDENT_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"
)
EPOCHS = float(os.environ.get("LIFECYCLE_STUDENT_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("LIFECYCLE_STUDENT_LR", "2e-5"))
OPTIMIZER = os.environ.get("LIFECYCLE_STUDENT_OPTIM", "adamw_torch")
MAX_LENGTH = int(os.environ.get("LIFECYCLE_STUDENT_MAX_LENGTH", "3072"))
SAVE_STEPS = int(os.environ.get("LIFECYCLE_STUDENT_SAVE_STEPS", "50"))
MIN_TRAIN_ROWS_BY_ACTION = {
    "continueCurrent": 0,
    "finishGoal": int(os.environ.get("LIFECYCLE_MIN_FINISH_ROWS", "384")),
    "ignore": int(os.environ.get("LIFECYCLE_MIN_IGNORE_ROWS", "768")),
    "startGoal": int(os.environ.get("LIFECYCLE_MIN_START_ROWS", "512")),
}
MIN_EXPLICIT_OPEN_ROWS = int(os.environ.get("LIFECYCLE_MIN_EXPLICIT_OPEN_ROWS", "384"))
MIN_COMPLETION_NEGATIVE_ROWS = int(
    os.environ.get("LIFECYCLE_MIN_COMPLETION_NEGATIVE_ROWS", "512")
)
MIN_OPEN_EVIDENCE_NEGATIVE_ROWS = int(
    os.environ.get("LIFECYCLE_MIN_OPEN_EVIDENCE_NEGATIVE_ROWS", "512")
)


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(decision):
    return json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def decision_obeys_contract(value, allow_legacy_start_title=False):
    if not isinstance(value, dict):
        return False
    required = {"action", "goal_ref", "title", "lifecycle_evidence"}
    if set(value) != required or not all(isinstance(value[key], str) for key in required):
        return False
    action = value["action"]
    evidence = value["lifecycle_evidence"]
    if action not in {"startGoal", "continueCurrent", "finishGoal", "ignore"}:
        return False
    if evidence not in {"none", "explicit_open", "explicit_completion"}:
        return False
    if action == "startGoal":
        if value["goal_ref"] != "NEW_GOAL":
            return False
        title_words = value["title"].split()
        if title_words and (not allow_legacy_start_title or not 3 <= len(title_words) <= 7):
            return False
    elif value["goal_ref"] == "NEW_GOAL" or value["title"]:
        return False
    if (action == "finishGoal") != (evidence == "explicit_completion"):
        return False
    return True


def parse_decision(text, allow_legacy_start_title=False):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not decision_obeys_contract(value, allow_legacy_start_title):
        return None
    return value


def target_for(row):
    assistants = [item for item in row["messages"] if item["role"] == "assistant"]
    if len(assistants) != 1:
        raise ValueError(f"{row['id']} must contain one assistant target")
    value = parse_decision(assistants[0]["content"], allow_legacy_start_title=True)
    if value is None:
        raise ValueError(f"{row['id']} contains an invalid assistant target")
    value["title"] = ""
    return value


def input_for(row):
    users = [item for item in row["messages"] if item["role"] == "user"]
    if len(users) != 1:
        raise ValueError(f"{row['id']} must contain one user input")
    return json.loads(users[0]["content"])


def is_completion_negative(row):
    if target_for(row)["action"] == "finishGoal":
        return False
    text = input_for(row)["text"].casefold()
    return any(
        token in text
        for token in (
            "commit",
            "complete",
            "done",
            "finished",
            "installed",
            "subagent_notification",
            "success",
            "working",
            "good",
            "great",
            "no limit",
            "looks good",
        )
    )

def is_open_evidence_negative(row):
    target = target_for(row)
    if target["lifecycle_evidence"] != "none":
        return False
    text = input_for(row)["text"].casefold()
    return any(
        token in text
        for token in (
            "failed",
            "improve",
            "open",
            "pending",
            "research",
            "retry",
            "status",
            "still",
            "wrong",
        )
    )



def augment_train_rows(rows):
    action_buckets = defaultdict(list)
    for row in rows:
        action_buckets[target_for(row)["action"]].append(row)
    required = set(MIN_TRAIN_ROWS_BY_ACTION)
    missing = required - action_buckets.keys()
    if missing:
        raise ValueError(f"training split is missing actions: {', '.join(sorted(missing))}")
    augmented = list(rows)
    for action in sorted(required):
        bucket = action_buckets[action]
        minimum = MIN_TRAIN_ROWS_BY_ACTION[action]
        for index in range(max(0, minimum - len(bucket))):
            augmented.append(bucket[index % len(bucket)])
    hard_buckets = {
        "completion_negative": [row for row in rows if is_completion_negative(row)],
        "open_evidence_negative": [
            row for row in rows if is_open_evidence_negative(row)
        ],
        "explicit_open": [
            row for row in rows if target_for(row)["lifecycle_evidence"] == "explicit_open"
        ],
    }
    hard_minimums = {
        "completion_negative": MIN_COMPLETION_NEGATIVE_ROWS,
        "open_evidence_negative": MIN_OPEN_EVIDENCE_NEGATIVE_ROWS,
        "explicit_open": MIN_EXPLICIT_OPEN_ROWS,
    }
    for name, bucket in hard_buckets.items():
        minimum = hard_minimums[name]
        for index in range(max(0, minimum - len(bucket))):
            augmented.append(bucket[index % len(bucket)])
    raw_counts = {action: len(action_buckets[action]) for action in sorted(required)}
    hard_counts = {name: len(bucket) for name, bucket in sorted(hard_buckets.items())}
    effective_counts = dict(Counter(target_for(row)["action"] for row in augmented))
    return augmented, raw_counts, effective_counts, hard_counts

