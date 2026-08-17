#!/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd/venv/bin/python
"""Measure whether lifecycle answer labels survive training truncation."""

import json
from pathlib import Path

from transformers import AutoTokenizer

work = Path("/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd")
tokenizer = AutoTokenizer.from_pretrained(work / "student")
limit = 3072
judge = json.loads((work / "final-judge-local.json").read_text(encoding="utf-8"))
rejected = {
    record["id"]
    for record in judge["records"]
    if (record.get("decision") or {}).get("verdict") != "student-sensible"
    or (record.get("decision") or {}).get("dangerous_finish")
}

for split in ("reviewed-train.jsonl", "reviewed-eval.jsonl"):
    lengths = []
    rejected_lengths = []
    answer_missing = 0
    answer_partial = 0
    with (work / split).open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            messages = [item for item in row["messages"] if item["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            answer = next(item["content"] for item in row["messages"] if item["role"] == "assistant")
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            answer_tokens = len(tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
            lengths.append(prompt_tokens)
            if prompt_tokens >= limit:
                answer_missing += 1
            elif prompt_tokens + answer_tokens > limit:
                answer_partial += 1
            if row["id"] in rejected:
                rejected_lengths.append(prompt_tokens)
    ordered = sorted(lengths)
    print(
        json.dumps(
            {
                "split": split,
                "rows": len(lengths),
                "prompt_p50": ordered[len(ordered) // 2],
                "prompt_p90": ordered[len(ordered) * 9 // 10],
                "prompt_p99": ordered[len(ordered) * 99 // 100],
                "prompt_max": ordered[-1],
                "answer_missing": answer_missing,
                "answer_partial": answer_partial,
                "rejected_rows": len(rejected_lengths),
                "rejected_prompt_min": min(rejected_lengths) if rejected_lengths else None,
                "rejected_prompt_max": max(rejected_lengths) if rejected_lengths else None,
            },
            sort_keys=True,
        )
    )
