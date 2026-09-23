"""Reading lifecycle evaluation rows and the decisions a model writes for them."""

import json

ACTIONS = {"startGoal", "continueCurrent", "finishGoal", "ignore"}
EVIDENCE = {"none", "explicit_open", "explicit_completion"}


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    required = {"action", "goal_ref", "title", "lifecycle_evidence"}
    if not isinstance(value, dict) or set(value) != required:
        return None
    if not all(isinstance(value[key], str) for key in required):
        return None
    action = value["action"]
    evidence = value["lifecycle_evidence"]
    if action not in ACTIONS or evidence not in EVIDENCE:
        return None
    if action == "startGoal":
        if value["goal_ref"] != "NEW_GOAL":
            return None
        title_words = value["title"].split()
        if title_words and (not allow_legacy_start_title or not 3 <= len(title_words) <= 7):
            return None
    elif value["goal_ref"] == "NEW_GOAL" or value["title"]:
        return None
    if (action == "finishGoal") != (evidence == "explicit_completion"):
        return None
    return value


def target_for(row):
    content = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    value = parse_decision(content, allow_legacy_start_title=True)
    if value is None:
        raise RuntimeError(f"invalid evaluation target for {row['id']}")
    value["title"] = ""
    return value


def input_for(row):
    content = next(message["content"] for message in row["messages"] if message["role"] == "user")
    return json.loads(content)
