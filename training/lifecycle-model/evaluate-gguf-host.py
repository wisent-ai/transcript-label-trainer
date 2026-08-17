#!/usr/bin/env python3
"""Evaluate the quantized lifecycle model through Oko's serving protocol."""

import copy
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORK = Path(
    os.environ.get(
        "LIFECYCLE_EVAL_WORK",
        "/mnt/wisent-staging/oko-lifecycle-model-b5de55bd",
    )
)
MODEL = Path(
    os.environ.get(
        "LIFECYCLE_EVAL_MODEL",
        str(WORK / "oko-lifecycle-qwen3-8b-q4_k_m.gguf"),
    )
)
DATASET = Path(
    os.environ.get("LIFECYCLE_EVAL_DATASET", str(WORK / "reviewed-eval-curriculum.jsonl"))
)
PREDICTIONS = Path(
    os.environ.get("LIFECYCLE_EVAL_PREDICTIONS", str(WORK / "predictions-gguf.jsonl"))
)
METRICS = Path(os.environ.get("LIFECYCLE_EVAL_METRICS", str(WORK / "metrics-gguf.json")))
SERVER_LOG = Path(
    os.environ.get("LIFECYCLE_EVAL_SERVER_LOG", str(WORK / "llama-server-eval.log"))
)
SERVER = Path(
    os.environ.get(
        "LIFECYCLE_EVAL_SERVER",
        str(WORK / "llama.cpp/build/bin/llama-server"),
    )
)
PORT = int(os.environ.get("LIFECYCLE_EVAL_PORT", "11440"))
ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PARALLEL = int(os.environ.get("LIFECYCLE_EVAL_PARALLEL", "4"))
SLOT_CONTEXT = int(os.environ.get("LIFECYCLE_EVAL_SLOT_CONTEXT", "4096"))
ACTIONS = {"startGoal", "continueCurrent", "finishGoal", "ignore"}
SYSTEM_PROMPT = Path(
    os.environ.get(
        "LIFECYCLE_EVAL_SYSTEM_PROMPT",
        str(WORK / "audit-source/training/lifecycle-model/lifecycle-system-prompt.txt"),
    )
).read_text(encoding="utf-8").strip()
EVIDENCE = {"none", "explicit_open", "explicit_completion"}
OUTPUT_SCHEMA = json.loads(
    Path(
        os.environ.get(
            "LIFECYCLE_EVAL_OUTPUT_SCHEMA",
            str(WORK / "audit-source/training/lifecycle-model/lifecycle-output-schema.json"),
        )
    ).read_text(encoding="utf-8")
)


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

def response_format(row):
    """Narrow the one checked-in decision contract to this request's candidates.

    The contract declares goal_ref as a pattern because it cannot know the
    live references; serving replaces that pattern with the exact enumeration
    and constrains decoding to the result. Nothing else is redeclared here.
    """
    existing_refs = [
        candidate["ref"]
        for candidate in input_for(row)["candidates"]
        if candidate["ref"] != "NEW_GOAL"
    ]
    variants = []
    for variant in OUTPUT_SCHEMA["oneOf"]:
        variant = copy.deepcopy(variant)
        goal_ref = variant["properties"]["goal_ref"]
        if "pattern" in goal_ref:
            del goal_ref["pattern"]
            goal_ref["enum"] = existing_refs
        variants.append(variant)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "oko_goal_lifecycle",
            "strict": True,
            "schema": {"oneOf": variants},
        },
    }


def wait_for_server(process):
    health = f"http://127.0.0.1:{PORT}/health"
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited {process.returncode} before readiness")
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise RuntimeError("llama-server did not become ready")


def classify(row):
    user_message = next(message for message in row["messages"] if message["role"] == "user")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message["content"]},
    ]
    payload = json.dumps(
        {
            "model": "oko-goal-lifecycle-v1",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 96,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": response_format(row),
        }
    ).encode()
    last_error = None
    for attempt in range(3):
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            raw = body["choices"][0]["message"]["content"].strip()
            return raw, parse_decision(raw)
        except Exception as error:
            last_error = error
            time.sleep(1 << attempt)
    raise RuntimeError(f"{row['id']} inference failed: {last_error}")


def main():
    for required in (MODEL, DATASET, SERVER):
        if not required.is_file():
            raise RuntimeError(f"required GGUF evaluation input is missing: {required}")
    rows = read_rows(DATASET)
    with SERVER_LOG.open("wb") as server_log:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--model",
                str(MODEL),
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--alias",
                "oko-goal-lifecycle-v1",
                "--ctx-size",
                str(SLOT_CONTEXT * PARALLEL),
                "--gpu-layers",
                "99",
                "--parallel",
                str(PARALLEL),
                "--no-webui",
            ],
            cwd=WORK,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_server(process)
            results = [None] * len(rows)
            with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
                futures = {executor.submit(classify, row): index for index, row in enumerate(rows)}
                for completed, future in enumerate(as_completed(futures), 1):
                    index = futures[future]
                    raw, prediction = future.result()
                    row = rows[index]
                    target = target_for(row)
                    results[index] = {
                        "id": row["id"],
                        "target": target,
                        "input": input_for(row),
                        "prediction": prediction,
                        "raw": raw,
                        "valid": prediction is not None,
                        "action_correct": prediction is not None and prediction["action"] == target["action"],
                        "goal_ref_correct": prediction is not None and prediction["goal_ref"] == target["goal_ref"],
                        "evidence_correct": prediction is not None
                        and prediction["lifecycle_evidence"] == target["lifecycle_evidence"],
                        "metadata": row.get("metadata", {}),
                    }
                    if completed % 25 == 0 or completed == len(rows):
                        print(f"quantized lifecycle predictions {completed}/{len(rows)}", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    metrics = Counter()
    by_action = defaultdict(Counter)
    with PREDICTIONS.open("w", encoding="utf-8") as handle:
        for result in results:
            prediction = result["prediction"]
            target = result["target"]
            action_correct = result["action_correct"]
            ref_correct = result["goal_ref_correct"]
            evidence_correct = result["evidence_correct"]
            joint = action_correct and ref_correct and evidence_correct
            result["joint_correct"] = joint
            metrics.update(
                {
                    "valid_json": int(result["valid"]),
                    "action_correct": int(action_correct),
                    "goal_ref_correct": int(ref_correct),
                    "evidence_correct": int(evidence_correct),
                    "joint_correct": int(joint),
                }
            )
            bucket = by_action[target["action"]]
            bucket["rows"] += 1
            bucket["action_correct"] += int(action_correct)
            bucket["joint_correct"] += int(joint)
            if prediction and prediction["action"] == "finishGoal":
                metrics["predicted_finish"] += 1
                metrics["correct_finish"] += int(target["action"] == "finishGoal")
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    total = len(results)
    report = {
        "rows": total,
        "model": str(MODEL),
        "valid_json": metrics["valid_json"] / total,
        "action_accuracy": metrics["action_correct"] / total,
        "goal_ref_accuracy": metrics["goal_ref_correct"] / total,
        "evidence_accuracy": metrics["evidence_correct"] / total,
        "joint_accuracy": metrics["joint_correct"] / total,
        "finish_precision": metrics["correct_finish"] / metrics["predicted_finish"]
        if metrics["predicted_finish"]
        else 0.0,
        "by_action": {action: dict(counts) for action, counts in sorted(by_action.items())},
    }
    METRICS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
