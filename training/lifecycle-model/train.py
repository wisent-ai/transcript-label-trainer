#!/usr/bin/env python3
"""Fine-tune Oko's contextual goal-lifecycle model from reviewed JSONL."""

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 29
BASE_MODEL = os.environ.get("LIFECYCLE_STUDENT_MODEL", "Qwen/Qwen3-4B")
BASE_REVISION = os.environ.get(
    "LIFECYCLE_STUDENT_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"
)
EPOCHS = float(os.environ.get("LIFECYCLE_STUDENT_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LIFECYCLE_STUDENT_LR", "1e-5"))
MAX_LENGTH = int(os.environ.get("LIFECYCLE_STUDENT_MAX_LENGTH", "3072"))
MIN_TRAIN_ROWS_PER_ACTION = int(
    os.environ.get("LIFECYCLE_MIN_TRAIN_ROWS_PER_ACTION", "256")
)
SYSTEM_PROMPT = (Path(__file__).resolve().parent / "lifecycle-system-prompt.txt").read_text(
    encoding="utf-8"
).strip()


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(decision):
    return json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_decision(text):
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
    return value


def target_for(row):
    assistants = [item for item in row["messages"] if item["role"] == "assistant"]
    if len(assistants) != 1:
        raise ValueError(f"{row['id']} must contain one assistant target")
    value = parse_decision(assistants[0]["content"])
    if value is None:
        raise ValueError(f"{row['id']} contains an invalid assistant target")
    return value


def augment_train_rows(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[target_for(row)["action"]].append(row)
    required = {"startGoal", "continueCurrent", "finishGoal", "ignore"}
    missing = required - buckets.keys()
    if missing:
        raise ValueError(f"training split is missing actions: {', '.join(sorted(missing))}")
    augmented = list(rows)
    for action in sorted(required):
        bucket = buckets[action]
        for index in range(max(0, MIN_TRAIN_ROWS_PER_ACTION - len(bucket))):
            augmented.append(bucket[index % len(bucket)])
    return augmented, {action: len(buckets[action]) for action in sorted(required)}




def prompt_messages(row):
    messages = [item for item in row["messages"] if item["role"] != "assistant"]
    if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"{row['id']} has an invalid prompt envelope")
    messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    return messages


def digest(paths):
    value = hashlib.sha256()
    for path in paths:
        value.update(Path(path).read_bytes())
    return value.hexdigest()


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_path = Path(os.environ.get("LIFECYCLE_TRAIN_DATASET", "reviewed-train.jsonl"))
    eval_path = Path(os.environ.get("LIFECYCLE_EVAL_DATASET", "reviewed-eval.jsonl"))
    unique_train_rows = read_rows(train_path)
    eval_rows = read_rows(eval_path)
    if len(unique_train_rows) < 500:
        raise SystemExit(
            f"need at least 500 reviewed train rows, found {len(unique_train_rows)}"
        )
    if len(eval_rows) < 100:
        raise SystemExit(f"need at least 100 reviewed eval rows, found {len(eval_rows)}")
    train_rows, train_action_counts = augment_train_rows(unique_train_rows)
    eval_action_counts = Counter(target_for(row)["action"] for row in eval_rows)
    missing_eval_actions = {
        "startGoal", "continueCurrent", "finishGoal", "ignore"
    } - eval_action_counts.keys()
    if missing_eval_actions:
        raise SystemExit(
            "held-out split is missing actions: "
            + ", ".join(sorted(missing_eval_actions))
        )
    random.Random(SEED).shuffle(train_rows)
    print(
        f"reviewed train rows: {len(unique_train_rows)}; "
        f"effective balanced rows: {len(train_rows)}; "
        f"held-out rows: {len(eval_rows)}; "
        f"train actions: {train_action_counts}; "
        f"held-out actions: {dict(sorted(eval_action_counts.items()))}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            prompt_messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = canonical(target_for(row))
        full = prompt + answer + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        labels = list(encoded["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        encoded["labels"] = labels
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0]))
    eval_dataset = Dataset.from_list(eval_rows).map(encode, remove_columns=list(eval_rows[0]))

    def collate(features):
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    args = TrainingArguments(
        output_dir="student-checkpoints",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        seed=SEED,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.train()
    model.config.use_cache = True
    trainer.save_model("student")
    tokenizer.save_pretrained("student")

    model.eval()
    model.to("cuda")
    predictions = []
    metrics = Counter()
    by_action = defaultdict(Counter)
    for index, row in enumerate(eval_rows, 1):
        target = target_for(row)
        prompt = tokenizer.apply_chat_template(
            prompt_messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        prediction = parse_decision(raw)
        valid = prediction is not None
        action_correct = valid and prediction["action"] == target["action"]
        ref_correct = valid and prediction["goal_ref"] == target["goal_ref"]
        evidence_correct = valid and prediction["lifecycle_evidence"] == target["lifecycle_evidence"]
        joint = action_correct and ref_correct and evidence_correct
        exact = valid and canonical(prediction) == canonical(target)
        metrics.update(
            {
                "valid_json": int(valid),
                "action_correct": int(action_correct),
                "goal_ref_correct": int(ref_correct),
                "evidence_correct": int(evidence_correct),
                "joint_correct": int(joint),
                "exact_json": int(exact),
            }
        )
        bucket = by_action[target["action"]]
        bucket["rows"] += 1
        bucket["action_correct"] += int(action_correct)
        bucket["joint_correct"] += int(joint)
        if valid and prediction["action"] == "finishGoal":
            metrics["predicted_finish"] += 1
            metrics["correct_finish"] += int(target["action"] == "finishGoal")
        predictions.append(
            {
                "id": row["id"],
                "target": target,
                "input": json.loads(prompt_messages(row)[1]["content"]),
                "prediction": prediction,
                "raw": raw,
                "valid": valid,
                "action_correct": action_correct,
                "goal_ref_correct": ref_correct,
                "evidence_correct": evidence_correct,
                "joint_correct": joint,
                "metadata": row.get("metadata", {}),
            }
        )
        if index % 25 == 0 or index == len(eval_rows):
            print(f"generated {index}/{len(eval_rows)} held-out predictions", flush=True)

    with open("predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(eval_rows)
    report = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset_sha256": digest([train_path, eval_path]),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_rows": len(train_rows),
        "unique_train_rows": len(unique_train_rows),
        "train_action_counts": train_action_counts,
        "eval_action_counts": dict(sorted(eval_action_counts.items())),
        "eval_rows": total,
        "valid_json": metrics["valid_json"] / total,
        "action_accuracy": metrics["action_correct"] / total,
        "goal_ref_accuracy": metrics["goal_ref_correct"] / total,
        "evidence_accuracy": metrics["evidence_correct"] / total,
        "joint_accuracy": metrics["joint_correct"] / total,
        "exact_json": metrics["exact_json"] / total,
        "finish_precision": (
            metrics["correct_finish"] / metrics["predicted_finish"]
            if metrics["predicted_finish"]
            else 0.0
        ),
        "by_action": {key: dict(value) for key, value in sorted(by_action.items())},
        "log_history": trainer.state.log_history,
    }
    Path("metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("student and held-out predictions written", flush=True)


if __name__ == "__main__":
    main()
