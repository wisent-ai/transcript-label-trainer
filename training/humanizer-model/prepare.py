#!/usr/bin/env python3
"""Create inverse style-transfer pairs from masked Łukasz-authored messages."""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "Faithfully paraphrase the supplied text. Preserve its complete meaning, intent, level of "
    "certainty, facts, names, product and company names, technical terms, numbers, dates, amounts, "
    "URLs, email addresses, quotations, code, commands, meaningful formatting, emotional force, "
    "and calls to action. Keep the source language and register. Do not add claims, examples, "
    "greetings, conclusions, or context. Return only the rewritten text, with no analysis, label, "
    "preface, or notes."
)
TEACHER_PROMPT = (
    "Create the source side of an inverse style-transfer example. Rewrite the authored message as "
    "generic polished AI-assistant prose in the same language. Preserve every fact, name, technical "
    "term, number, date, amount, URL, email address, quotation, command, intent, uncertainty, emotional "
    "force, and call to action. Do not summarize, answer, explain, censor, or add information. Make the "
    "wording and rhythm substantially more generic and AI-like. Return only the rewritten message."
)
REVIEW_PROMPT = (
    "Judge one proposed inverse style-transfer pair. The source must be generic polished AI prose; "
    "the target must preserve the same complete meaning while retaining the author's natural voice. "
    "Return exactly JSON with booleans faithful, generic_ai, same_language, and usable. usable may be "
    "true only when all other fields are true and neither side adds or drops any fact, name, number, "
    "technical term, request, question, uncertainty, or emotional force."
)
ANCHOR = re.compile(r"https?://\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b\d[\d.,:/%-]*\b")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_url() -> str:
    for name in ("BRAMA_URL", "JEDEN_BRAMA_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value.rstrip("/")
    return "https://brama.wisent.com"


class Brama:
    def __init__(self) -> None:
        self.url = resolve_url()
        self.agent_id = os.environ.get("WISENT_APP_AGENT_ID", "wisent-app").strip()
        self.secret = os.environ["WISENT_APP_AGENT_AUTH_SECRET"].strip()
        self.token = os.environ["BRAMA_TOKEN"].strip()
        if not self.secret or not self.token:
            raise RuntimeError("Brama credentials are empty")

    def chat(self, model: str, system: str, user: str, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        timestamp = str(int(time.time()))
        digest = hashlib.sha256(body.encode()).hexdigest()
        signature = hmac.new(
            self.secret.encode(),
            f"{self.agent_id}:{timestamp}:{digest}".encode(),
            hashlib.sha256,
        ).hexdigest()
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            data=body.encode(),
            headers={
                "authorization": f"Bearer {self.token}",
                "content-type": "application/json",
                "x-agent-id": self.agent_id,
                "x-agent-timestamp": timestamp,
                "x-agent-body-sha256": digest,
                "x-agent-signature": signature,
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Brama HTTP {response.status_code}: {response.text[:300]}")
        content = response.json()["choices"][0]["message"]["content"].strip()
        if not content:
            raise RuntimeError("Brama returned empty content")
        return content


def retry(operation):
    last = None
    for attempt in range(4):
        try:
            return operation()
        except Exception as error:  # retained in the rejected-row record, never hides a failed corpus
            last = error
            time.sleep(2**attempt)
    raise last


def protected_anchors(text: str) -> set[str]:
    return {match.group(0) for match in ANCHOR.finditer(text)}


def valid_source(target: str, source: str) -> bool:
    target_clean = target.strip()
    source_clean = source.strip()
    if not source_clean or source_clean == target_clean:
        return False
    ratio = len(source_clean) / max(1, len(target_clean))
    if ratio < 0.65 or ratio > 2.5:
        return False
    return protected_anchors(target_clean).issubset(protected_anchors(source_clean))


def parse_review(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("review is not JSON")
    value = json.loads(raw[start : end + 1])
    required = {"faithful", "generic_ai", "same_language", "usable"}
    if set(value) != required or any(not isinstance(value[key], bool) for key in required):
        raise ValueError("review has invalid shape")
    return value


def prepare(row: dict, client: Brama, teacher: str, reviewer: str) -> tuple[dict | None, str | None]:
    target = row["target"].strip()
    try:
        source = retry(
            lambda: client.chat(
                teacher,
                TEACHER_PROMPT,
                target,
                min(1024, max(192, len(target) * 2)),
            )
        )
        if not valid_source(target, source):
            return None, "source_contract"
        review_user = json.dumps({"source": source, "target": target}, ensure_ascii=False)
        review = retry(
            lambda: parse_review(client.chat(reviewer, REVIEW_PROMPT, review_user, 96))
        )
        if not review["usable"]:
            return None, "review_rejected"
        split_bucket = int(hashlib.sha256(row["session_id"].encode()).hexdigest()[:8], 16) % 10
        split = "test" if split_bucket == 0 else "validation" if split_bucket == 1 else "train"
        return (
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "session_id": row["session_id"],
                    "runtime": row["runtime"],
                    "split": split,
                    "teacher_model": teacher,
                    "review_model": reviewer,
                    "review": review,
                },
            },
            None,
        )
    except Exception as error:
        return None, f"error:{type(error).__name__}:{str(error)[:160]}"


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    input_path = Path(os.environ.get("HUMANIZER_TARGETS", "targets.jsonl"))
    output_dir = Path(os.environ.get("HUMANIZER_DATASET_DIR", "."))
    teacher = os.environ.get("HUMANIZER_TEACHER_MODEL", "codex/gpt-5.6-sol")
    reviewer = os.environ.get("HUMANIZER_REVIEW_MODEL", "-best")
    workers = int(os.environ.get("HUMANIZER_PREP_WORKERS", "16"))
    targets = rows(input_path)
    client = Brama()
    accepted: list[dict] = []
    rejected = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare, row, client, teacher, reviewer) for row in targets]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            value, reason = future.result()
            if value is not None:
                accepted.append(value)
            else:
                rejected[reason or "unknown"] += 1
            if index % 50 == 0 or index == len(futures):
                print(
                    f"prepared {index}/{len(futures)}; accepted {len(accepted)}; "
                    f"rejected {dict(sorted(rejected.items()))}",
                    flush=True,
                )
    accepted.sort(key=lambda row: row["id"])
    split_rows = {
        name: [row for row in accepted if row["metadata"]["split"] == name]
        for name in ("train", "validation", "test")
    }
    minimums = {"train": 700, "validation": 70, "test": 70}
    for name, minimum in minimums.items():
        if len(split_rows[name]) < minimum:
            raise SystemExit(f"{name} has {len(split_rows[name])} accepted rows; need {minimum}")
        write_jsonl(output_dir / f"{name}.jsonl", split_rows[name])
    report = {
        "schema_version": 1,
        "source": "transcript-lake:masked-user-events",
        "target_rows": len(targets),
        "accepted_rows": len(accepted),
        "splits": {name: len(values) for name, values in split_rows.items()},
        "rejected": dict(sorted(rejected.items())),
        "teacher_model": teacher,
        "review_model": reviewer,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
