#!/usr/bin/env python3
"""Regenerate held-out goals with runtime contract correction."""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path("/mnt/wd16tb/stado/jobs/jeden-goal-prompt-v5")
PROMPT = Path.home().joinpath(".stado/files/goal-system-prompt.md").read_text().strip()
SOURCE = WORK / "predictions.jsonl"
OUTPUT = WORK / "predictions-corrected.jsonl"
POLISH = {
    "aby", "ale", "czy", "co", "dlaczego", "gdzie", "go", "ile", "jak", "jakie",
    "ktore", "ktory", "ma", "mamy", "moj", "na", "nasz", "naszego", "naszych",
    "nie", "o", "przed", "prosze", "sie", "to", "w", "z", "zamknelo", "znalezc",
}
ENGLISH = {
    "and", "check", "continue", "find", "identify", "in", "of", "on", "resolve",
    "the", "to", "update", "with",
}

def words(text):
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)

def is_polish(text):
    tokens = words(text)
    return sum(token in POLISH for token in tokens) >= 2 or any(
        character in text.lower() for character in "ąćęłńóśźż"
    )

def violations(message, answer):
    issues = []
    if is_polish(message) and not is_polish(answer):
        issues.append("answer in Polish")
    identifiers = set(re.findall(r"\b[A-Z][A-Z0-9.+-]{1,}\b", message))
    missing = sorted(identifier for identifier in identifiers if identifier not in answer)
    if missing:
        issues.append("preserve identifiers exactly: " + ", ".join(missing))
    return issues

rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
tokenizer = AutoTokenizer.from_pretrained(WORK / "student")
model = AutoModelForCausalLM.from_pretrained(
    WORK / "student", torch_dtype=torch.bfloat16
).to("cuda").eval()

def generate(message, system):
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<user>{message}</user>"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

with OUTPUT.open("w", encoding="utf-8") as destination:
    for index, row in enumerate(rows, 1):
        answer = row["student"]
        issues = violations(row["message"], answer)
        if issues:
            correction = (
                PROMPT + "\n\nThe previous draft was invalid. Correct these issues: "
                + "; ".join(issues) + "."
            )
            answer = generate(row["message"], correction)
        row["student"] = answer
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        if issues:
            print(json.dumps({"session_id": row["session_id"], "issues": issues, "student": answer}, ensure_ascii=False))
print(OUTPUT)
