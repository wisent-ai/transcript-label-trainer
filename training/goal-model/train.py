#!/usr/bin/env python3
"""Fine-tune the Jeden goal student from a Brama-reviewed JSONL dataset."""

import hashlib
import json
import os
import random
from pathlib import Path

SEED = 17
BASE_MODEL = os.environ.get("GOAL_STUDENT_MODEL", "Qwen/Qwen3-4B")
BASE_REVISION = os.environ.get(
    "GOAL_STUDENT_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"
)
EPOCHS = float(os.environ.get("GOAL_STUDENT_EPOCHS", "4"))
LEARNING_RATE = float(os.environ.get("GOAL_STUDENT_LR", "1e-5"))
MAX_LENGTH = int(os.environ.get("GOAL_STUDENT_MAX_LENGTH", "2048"))
HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = (HERE / "goal-system-prompt.md").read_text(encoding="utf-8").strip()


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def messages(message):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<user>{message}</user>"},
    ]


def main():
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    dataset_path = Path(os.environ.get("GOAL_DATASET", "reviewed-goals.jsonl"))
    rows = read_rows(dataset_path)
    gold = [row for row in rows if row.get("gold")]
    train_rows = [row for row in rows if not row.get("gold")]
    if len(train_rows) < 100:
        raise SystemExit(f"need at least 100 reviewed teacher rows, found {len(train_rows)}")
    if len(gold) < 20:
        raise SystemExit(f"need at least 20 reviewed gold rows, found {len(gold)}")
    random.Random(SEED).shuffle(train_rows)
    random.Random(SEED).shuffle(gold)
    print(f"reviewed train rows: {len(train_rows)}; held-out gold rows: {len(gold)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            messages(row["message"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        answer = f"<goal>{row['goal']}</goal>" if row.get("goal") else "<goal/>"
        full = prompt + answer + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        encoded["labels"] = labels
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0]))
    eval_dataset = Dataset.from_list(gold).map(encode, remove_columns=list(gold[0]))

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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
    exact = 0
    for index, row in enumerate(gold, 1):
        prompt = tokenizer.apply_chat_template(
            messages(row["message"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        student = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        expected = f"<goal>{row['goal']}</goal>" if row.get("goal") else "<goal/>"
        exact += int(student == expected)
        predictions.append({
            "session_id": row["session_id"],
            "message": row["message"],
            "goal": row.get("goal") or "",
            "student": student,
        })
        if index % 25 == 0 or index == len(gold):
            print(f"generated {index}/{len(gold)} held-out predictions", flush=True)

    with open("predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    metrics = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset_sha256": dataset_sha256,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_rows": len(train_rows),
        "gold_rows": len(gold),
        "exact_match": exact / len(gold),
        "log_history": trainer.state.log_history,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("student and held-out predictions written", flush=True)


if __name__ == "__main__":
    main()
