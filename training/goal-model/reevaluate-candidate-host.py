#!/usr/bin/env python3
"""Regenerate held-out predictions from the staged student with the current prompt."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path("/mnt/wd16tb/stado/jobs/jeden-goal-568ebd79663775c9")
PROMPT = Path.home().joinpath(".stado/files/goal-system-prompt.md").read_text().strip()
SOURCE = WORK / "predictions.jsonl"
OUTPUT = WORK / "predictions-prompt-v2.jsonl"
MODEL = WORK / "student"

rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
with OUTPUT.open("w", encoding="utf-8") as destination:
    for index, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"<user>{row['message']}</user>"},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        row["student"] = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        if index % 25 == 0 or index == len(rows):
            print(f"generated {index}/{len(rows)}", flush=True)
print(OUTPUT)
