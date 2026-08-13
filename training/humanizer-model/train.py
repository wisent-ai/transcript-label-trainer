#!/usr/bin/env python3
"""Fine-tune and compare Echo's personal-voice humanizer on frozen splits."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SEED = 47
BASE_MODEL = os.environ.get("HUMANIZER_BASE_MODEL", "TheDrummer/Cydonia-24B-v4.3")
BASE_REVISION = os.environ.get(
    "HUMANIZER_BASE_REVISION", "db0426d39d4bd4a6d34fdc71db97569da68f55e1"
)
EPOCHS = float(os.environ.get("HUMANIZER_EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("HUMANIZER_LR", "2e-4"))
MAX_LENGTH = int(os.environ.get("HUMANIZER_MAX_LENGTH", "1024"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def ngrams(value: str, size: int = 3) -> Counter:
    normalized = " ".join(value.lower().split())
    if len(normalized) < size:
        return Counter([normalized])
    return Counter(normalized[index : index + size] for index in range(len(normalized) - size + 1))


def chrf(reference: str, candidate: str) -> float:
    left, right = ngrams(reference), ngrams(candidate)
    overlap = sum((left & right).values())
    precision = overlap / max(1, sum(right.values()))
    recall = overlap / max(1, sum(left.values()))
    return 2 * precision * recall / max(1e-12, precision + recall)


def score(source: str, target: str, candidate: str) -> dict[str, float]:
    return {
        "target_chrf": chrf(target, candidate),
        "target_sequence": SequenceMatcher(None, target.lower(), candidate.lower()).ratio(),
        "source_chrf": chrf(source, candidate),
        "length_ratio": len(candidate) / max(1, len(target)),
    }


def mean(values: list[dict], key: str) -> float:
    return sum(row[key] for row in values) / max(1, len(values))


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    train_path = Path(os.environ.get("HUMANIZER_TRAIN_DATASET", "train.jsonl"))
    validation_path = Path(os.environ.get("HUMANIZER_VALIDATION_DATASET", "validation.jsonl"))
    test_path = Path(os.environ.get("HUMANIZER_TEST_DATASET", "test.jsonl"))
    train_rows = read_rows(train_path)
    validation_rows = read_rows(validation_path)
    test_rows = read_rows(test_path)
    if len(train_rows) < 700 or len(validation_rows) < 70 or len(test_rows) < 70:
        raise SystemExit(
            f"insufficient frozen splits: train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)}"
        )
    random.Random(SEED).shuffle(train_rows)
    print(
        f"train rows: {len(train_rows)}; validation rows: {len(validation_rows)}; test rows: {len(test_rows)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prompt(row: dict) -> str:
        return tokenizer.apply_chat_template(
            row["messages"][:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def target(row: dict) -> str:
        return row["messages"][2]["content"].strip()

    def encode(row: dict) -> dict:
        prefix = prompt(row)
        answer = target(row)
        full = prefix + answer + tokenizer.eos_token
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
        labels = list(encoded["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        encoded["labels"] = labels
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0]))
    validation_dataset = Dataset.from_list(validation_rows).map(
        encode, remove_columns=list(validation_rows[0])
    )

    def collate(features: list[dict]) -> dict:
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

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = True

    def generate(candidate_model, row: dict) -> str:
        encoded = tokenizer(prompt(row), return_tensors="pt").to("cuda")
        target_tokens = len(tokenizer(target(row), add_special_tokens=False)["input_ids"])
        with torch.inference_mode():
            generated = candidate_model.generate(
                **encoded,
                max_new_tokens=min(512, max(64, int(target_tokens * 1.6) + 24)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    base_outputs = []
    for index, row in enumerate(test_rows, 1):
        output = generate(model, row)
        base_outputs.append(output)
        if index % 25 == 0 or index == len(test_rows):
            print(f"base predictions {index}/{len(test_rows)}", flush=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            base_model_name_or_path=BASE_MODEL,
            revision=BASE_REVISION,
            task_type="CAUSAL_LM",
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    arguments = TrainingArguments(
        output_dir="student-checkpoints",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        seed=SEED,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate,
    )
    trainer.train()
    model.config.use_cache = True
    trainer.save_model("student")
    tokenizer.save_pretrained("student")

    predictions = []
    base_scores, student_scores = [], []
    for index, (row, base_output) in enumerate(zip(test_rows, base_outputs), 1):
        student_output = generate(model, row)
        source = row["messages"][1]["content"].strip()
        expected = target(row)
        base_score = score(source, expected, base_output)
        student_score = score(source, expected, student_output)
        base_scores.append(base_score)
        student_scores.append(student_score)
        predictions.append(
            {
                "id": row["id"],
                "source": source,
                "target": expected,
                "base": base_output,
                "student": student_output,
                "base_metrics": base_score,
                "student_metrics": student_score,
                "metadata": row.get("metadata", {}),
            }
        )
        if index % 25 == 0 or index == len(test_rows):
            print(f"student predictions {index}/{len(test_rows)}", flush=True)

    with Path("predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    metrics = {
        "schema_version": 1,
        "contract": "echo-lukasz-humanizer-v1",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dataset_sha256": digest([train_path, validation_path, test_path]),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "base": {key: mean(base_scores, key) for key in base_scores[0]},
        "student": {key: mean(student_scores, key) for key in student_scores[0]},
        "log_history": trainer.state.log_history,
    }
    metrics["target_chrf_gain"] = metrics["student"]["target_chrf"] - metrics["base"]["target_chrf"]
    Path("metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
