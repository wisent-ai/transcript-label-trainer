#!/usr/bin/env python3
"""Independent Brama audit of base and trained humanizer outputs."""

from __future__ import annotations

import concurrent.futures
import json
import os
from collections import defaultdict
from pathlib import Path

from prepare import Brama, retry

JUDGE_PROMPT = (
    "Compare a base model and a trained personal-voice model on one held-out style-transfer case. "
    "The source is generic AI prose. The reference is a real message by the target author. Score each "
    "candidate independently from 0 to 1 for semantic_fidelity to the source and voice_match to the "
    "reference author's cadence, directness, register, and phrasing without requiring exact wording. "
    "ai_boilerplate is true when canned AI phrasing remains. passed is true only when semantic_fidelity "
    "is at least 0.95, voice_match is at least 0.75, and ai_boilerplate is false. Return exactly JSON: "
    '{"base":{"semantic_fidelity":0.0,"voice_match":0.0,"ai_boilerplate":true,"passed":false},'
    '"student":{"semantic_fidelity":0.0,"voice_match":0.0,"ai_boilerplate":true,"passed":false}}.'
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge output is not JSON")
    value = json.loads(raw[start : end + 1])
    if set(value) != {"base", "student"}:
        raise ValueError("judge output has invalid candidates")
    for name in ("base", "student"):
        result = value[name]
        if set(result) != {"semantic_fidelity", "voice_match", "ai_boilerplate", "passed"}:
            raise ValueError(f"judge output has invalid {name} shape")
        for score in ("semantic_fidelity", "voice_match"):
            if not isinstance(result[score], (int, float)) or not 0 <= result[score] <= 1:
                raise ValueError(f"judge output has invalid {name}.{score}")
        if not isinstance(result["ai_boilerplate"], bool) or not isinstance(result["passed"], bool):
            raise ValueError(f"judge output has invalid {name} verdict")
    return value


def judge(row: dict, client: Brama, model: str) -> dict:
    payload = json.dumps(
        {key: row[key] for key in ("source", "target", "base", "student")},
        ensure_ascii=False,
    )
    verdict = retry(lambda: parse(client.chat(model, JUDGE_PROMPT, payload, 256)))
    return {"id": row["id"], "verdict": verdict}


def aggregate(records: list[dict], candidate: str) -> dict:
    values = [record["verdict"][candidate] for record in records]
    return {
        "semantic_fidelity": sum(item["semantic_fidelity"] for item in values) / len(values),
        "voice_match": sum(item["voice_match"] for item in values) / len(values),
        "ai_boilerplate_rate": sum(item["ai_boilerplate"] for item in values) / len(values),
        "pass_rate": sum(item["passed"] for item in values) / len(values),
    }


def main() -> None:
    predictions_path = Path(os.environ.get("HUMANIZER_PREDICTIONS", "predictions.jsonl"))
    output_path = Path(os.environ.get("HUMANIZER_AUDIT_OUTPUT", "audit.json"))
    model = os.environ.get("HUMANIZER_AUDIT_MODEL", "-best")
    workers = int(os.environ.get("HUMANIZER_AUDIT_WORKERS", "8"))
    input_rows = read_rows(predictions_path)
    client = Brama()
    records = []
    failures = defaultdict(int)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge, row, client, model): row["id"] for row in input_rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as error:
                failures[f"{type(error).__name__}:{str(error)[:160]}"] += 1
            if index % 25 == 0 or index == len(futures):
                print(f"audited {index}/{len(futures)}", flush=True)
    if failures or len(records) != len(input_rows):
        raise SystemExit(f"audit incomplete: {dict(failures)}")
    records.sort(key=lambda value: value["id"])
    base = aggregate(records, "base")
    student = aggregate(records, "student")
    voice_gain = student["voice_match"] - base["voice_match"]
    semantic_delta = student["semantic_fidelity"] - base["semantic_fidelity"]
    passed = (
        student["semantic_fidelity"] >= 0.95
        and student["voice_match"] >= 0.80
        and student["pass_rate"] >= 0.90
        and student["ai_boilerplate_rate"] <= 0.08
        and voice_gain >= 0.15
        and semantic_delta >= -0.02
    )
    report = {
        "schema_version": 1,
        "contract": "echo-lukasz-humanizer-v1",
        "review_model": model,
        "rows": len(records),
        "base": base,
        "student": student,
        "voice_match_gain": voice_gain,
        "semantic_fidelity_delta": semantic_delta,
        "passed": passed,
        "records": records,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False))
    if not passed:
        raise SystemExit("trained humanizer did not satisfy the held-out quality gate")


if __name__ == "__main__":
    main()
