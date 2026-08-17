#!/usr/bin/env python3
"""Qualify a staged lifecycle candidate on the machine that will serve it.

The training host converts and stages the candidate under
``stado://releases/oko/candidates/lifecycle-qwen3-4b/<digest>``. Qualification
belongs where Oko runs, because the surface that must be measured is the served
Q4_K_M GGUF answering Oko's loopback chat contract under constrained decoding —
not the trainer's own in-process generation.

Steps, in order, each one refusing to continue on failure:

1. fetch the ordered parts, assemble them, verify the assembled digest;
2. evaluate the served model with the checked-in decision contract enforced;
3. judge every held-out decision independently through Brama;
4. publish ``stado://releases/oko/models/...`` only when the gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CANDIDATE_FAMILY = "lifecycle-qwen3-4b"
MODEL_NAME = "oko-lifecycle-qwen3-4b-q4_k_m.gguf"
STADO = Path(os.environ.get("STADO_BIN", Path.home() / ".stado/bin/stado"))
AUDIT_WRAPPER = Path(
    os.environ.get("LIFECYCLE_AUDIT_WRAPPER", "/tmp/oko-lifecycle-review-local.sh")
)
LLAMA_SERVER = Path(os.environ.get("LIFECYCLE_LLAMA_SERVER", "/opt/homebrew/bin/llama-server"))


def run(*args: str, env: dict[str, str] | None = None) -> None:
    process = subprocess.run(args, env=env, check=False)
    if process.returncode != 0:
        raise SystemExit(f"failed ({process.returncode}): {' '.join(args)}")


def stado(*args: str) -> str:
    process = subprocess.run(
        [str(STADO), *args], check=False, capture_output=True, text=True
    )
    if process.returncode != 0:
        raise SystemExit(f"stado {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout


def digest_of(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def fetch_candidate(base: str, work: Path, expected_digest: str) -> Path:
    manifest_path = work / "candidate-manifest.json"
    if not manifest_path.exists():
        stado("storage", "get", f"{base}/candidate-manifest.json", str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["sha256"] != expected_digest:
        raise SystemExit("candidate manifest digest does not match the requested digest")

    model = work / manifest["filename"]
    if model.exists() and digest_of(model) == expected_digest:
        return model

    parts_dir = work / "parts"
    parts_dir.mkdir(exist_ok=True)
    parts = []
    for name in manifest["parts"]:
        part = parts_dir / name
        if not part.exists():
            stado("storage", "get", f"{base}/large-output/{name}", str(part))
        parts.append(part)
    with model.open("wb") as target:
        for part in parts:
            target.write(part.read_bytes())
    assembled = digest_of(model)
    if assembled != expected_digest:
        raise SystemExit(f"assembled digest {assembled} != published {expected_digest}")
    for name in ("metrics.json", "predictions.jsonl", "python-requirements.lock"):
        destination = work / f"training-{name}" if name != "python-requirements.lock" else work / name
        if not destination.exists():
            stado("storage", "get", f"{base}/{name}", str(destination))
    return model


def evaluate(work: Path, model: Path, dataset: Path) -> dict:
    metrics_path = work / "metrics-gguf.json"
    if not metrics_path.exists():
        environment = dict(os.environ)
        environment.update(
            {
                "LIFECYCLE_EVAL_WORK": str(work),
                "LIFECYCLE_EVAL_MODEL": str(model),
                "LIFECYCLE_EVAL_DATASET": str(dataset),
                "LIFECYCLE_EVAL_PREDICTIONS": str(work / "predictions-gguf.jsonl"),
                "LIFECYCLE_EVAL_METRICS": str(metrics_path),
                "LIFECYCLE_EVAL_SERVER_LOG": str(work / "llama-server-eval.log"),
                "LIFECYCLE_EVAL_SERVER": str(LLAMA_SERVER),
                "LIFECYCLE_EVAL_SYSTEM_PROMPT": str(
                    REPOSITORY / "training/lifecycle-model/lifecycle-system-prompt.txt"
                ),
                "LIFECYCLE_EVAL_OUTPUT_SCHEMA": str(
                    REPOSITORY / "training/lifecycle-model/lifecycle-output-schema.json"
                ),
            }
        )
        run(
            sys.executable,
            str(REPOSITORY / "training/lifecycle-model/evaluate-gguf-host.py"),
            env=environment,
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def judge(work: Path) -> dict:
    judge_path = work / "final-judge-gguf.json"
    if not judge_path.exists():
        run(
            "sh",
            str(AUDIT_WRAPPER),
            "lifecycle-audit",
            str(work / "predictions-gguf.jsonl"),
            "--output",
            str(judge_path),
        )
    return json.loads(judge_path.read_text(encoding="utf-8"))


def publish(work: Path, model: Path, digest: str) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "LIFECYCLE_PUBLISH_JOB_ID": digest[:8],
            "LIFECYCLE_PUBLISH_WORK": str(work),
            "LIFECYCLE_PUBLISH_SOURCE": str(REPOSITORY),
            "LIFECYCLE_PUBLISH_MODEL_NAME": model.name,
            "LIFECYCLE_PUBLISH_MODEL": str(model),
            "LIFECYCLE_PUBLISH_JUDGE": str(work / "final-judge-gguf.json"),
            "LIFECYCLE_PUBLISH_BASE_MODEL": "Qwen/Qwen3-4B",
            "LIFECYCLE_PUBLISH_DESTINATION_FAMILY": CANDIDATE_FAMILY,
            "STADO_BIN": str(STADO),
        }
    )
    run(
        sys.executable,
        str(REPOSITORY / "training/lifecycle-model/publish-recovered-candidate-host.py"),
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("digest", help="assembled sha256 of the staged candidate GGUF")
    parser.add_argument(
        "--dataset",
        default=str(Path.home() / ".transcript-label-trainer/goal-model/reviewed-eval.jsonl"),
        help="held-out curriculum the served model is measured on",
    )
    parser.add_argument(
        "--work",
        default=None,
        help="local qualification directory; defaults to a digest-named directory",
    )
    arguments = parser.parse_args()

    work = Path(
        arguments.work
        or Path.home() / f".transcript-label-trainer/qualification/{arguments.digest[:12]}"
    )
    work.mkdir(parents=True, exist_ok=True)
    dataset = Path(arguments.dataset)
    if not dataset.is_file():
        raise SystemExit(f"held-out curriculum missing: {dataset}")
    if not LLAMA_SERVER.is_file():
        raise SystemExit(f"llama-server missing: {LLAMA_SERVER}")

    base = f"stado://releases/oko/candidates/{CANDIDATE_FAMILY}/{arguments.digest}"
    model = fetch_candidate(base, work, arguments.digest)
    metrics = evaluate(work, model, dataset)
    print(json.dumps({"served_metrics": metrics}, indent=2, sort_keys=True), flush=True)
    verdict = judge(work)
    print(
        json.dumps(
            {"judge": {k: v for k, v in verdict.items() if k != "records"}},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    publish(work, model, arguments.digest)


if __name__ == "__main__":
    main()
