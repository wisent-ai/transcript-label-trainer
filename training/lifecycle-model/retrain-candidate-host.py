#!/usr/bin/env python3
"""Retrain and independently audit the existing lifecycle candidate in place."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REVISION = "11a04eb07c31fa7bb80c6406c2a121b789f84f6c"
JOB_ID = "b5de55bd"
WORK = Path(f"/mnt/wd16tb/wisent-staging/oko-lifecycle-model-{JOB_ID}")
SOURCE = WORK / "audit-source"
VENV = WORK / "venv/bin/python"
STATUS = WORK / "retrain-status.json"
TRAIN_LOG = WORK / "retrain.log"
AUDIT_LOG = WORK / "local-audit.log"
BINARY = WORK / "cargo-target/release/transcript-label-trainer"
GRANT_ENV = Path("/root/.stado/files/stado-agent-grant.env")
BRAMA_URL = "https://charless-mac-mini.tail6443b3.ts.net"
MODEL = "wisent-backend/chat/primary"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_status(**values: object) -> None:
    payload = {"job_id": JOB_ID, "source_revision": REVISION, **values}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=WORK, prefix=".retrain-", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(STATUS)


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key] = value
    return values


def acquire(values: dict[str, str], item: str, field: str) -> str:
    url = values["WC_AGENT_SKARBIEC_URL"].rstrip("/") + "/v1/items/read"
    token = Path(values["WC_AGENT_SKARBIEC_TOKEN_FILE"]).read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        url,
        data=json.dumps({"id": item, "field": field}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Consumer": values["WC_AGENT_SKARBIEC_CONSUMER"],
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.load(response).get("value")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Skarbiec returned no string for {item}/{field}")
    return value


def run(command: list[str], log: Path, environment: dict[str, str]) -> None:
    with log.open("ab") as output:
        result = subprocess.run(
            command,
            cwd=WORK,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {command[0]}")


def main() -> None:
    started_at = now()
    write_status(state="preparing", started_at=started_at)
    subprocess.run(["/usr/bin/git", "-C", str(SOURCE), "fetch", "origin", REVISION], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(SOURCE), "checkout", "--detach", REVISION], check=True)

    backup = WORK / "before-retrain"
    backup.mkdir(exist_ok=True)
    for name in (
        "metrics.json",
        "predictions.jsonl",
        "final-judge.json",
        "final-judge-best.json",
        "final-judge-codex.json",
        "final-judge-local.json",
    ):
        source = WORK / name
        if source.is_file() and not (backup / name).exists():
            shutil.copy2(source, backup / name)
    old_model = WORK / "oko-lifecycle-qwen3-4b-q4_k_m.gguf"
    old_model_sha256 = digest(old_model)

    for path in (WORK / "student", WORK / "student-checkpoints"):
        if path.exists():
            shutil.rmtree(path)
    TRAIN_LOG.write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "LIFECYCLE_TRAIN_DATASET": str(WORK / "reviewed-train-curriculum.jsonl"),
            "LIFECYCLE_EVAL_DATASET": str(WORK / "reviewed-eval-curriculum.jsonl"),
            "LIFECYCLE_STUDENT_MODEL": "Qwen/Qwen3-8B",
            "LIFECYCLE_STUDENT_REVISION": "b968826d9c46dd6066d109eabc6255188de91218",
            "LIFECYCLE_STUDENT_EPOCHS": "5",
            "LIFECYCLE_STUDENT_LR": "2e-5",
            "LIFECYCLE_STUDENT_OPTIM": "adafactor",
            "HF_HOME": str(WORK / "hf-cache"),
        }
    )
    write_status(
        state="training",
        started_at=started_at,
        old_model_sha256=old_model_sha256,
    )
    print(f"lifecycle retraining started at {started_at}", flush=True)
    run([str(VENV), str(SOURCE / "training/lifecycle-model/train.py")], TRAIN_LOG, environment)

    f16 = WORK / "oko-lifecycle-qwen3-8b-f16.gguf"
    q4 = WORK / "oko-lifecycle-qwen3-8b-q4_k_m.gguf"
    f16.unlink(missing_ok=True)
    q4.unlink(missing_ok=True)
    write_status(state="converting", started_at=started_at, old_model_sha256=old_model_sha256)
    run(
        [
            str(VENV),
            str(WORK / "llama.cpp/convert_hf_to_gguf.py"),
            str(WORK / "student"),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ],
        TRAIN_LOG,
        environment,
    )
    run(
        [
            str(WORK / "llama.cpp/build/bin/llama-quantize"),
            str(f16),
            str(q4),
            "Q4_K_M",
        ],
        TRAIN_LOG,
        environment,
    )

    grant = read_env(GRANT_ENV)
    environment.update(
        {
            "BRAMA_URL": BRAMA_URL,
            "BRAMA_TOKEN": acquire(grant, "jeden-model-router", "token"),
            "WISENT_APP_AGENT_AUTH_SECRET": acquire(
                grant, "jeden-agent-auth", "agent_auth_secret"
            ),
        }
    )
    judge = WORK / "final-judge-local.json"
    judge.unlink(missing_ok=True)
    write_status(
        state="auditing",
        started_at=started_at,
        old_model_sha256=old_model_sha256,
        new_model_sha256=digest(q4),
    )
    AUDIT_LOG.write_text("", encoding="utf-8")
    run(
        [
            str(BINARY),
            "lifecycle-audit",
            str(WORK / "predictions.jsonl"),
            "--output",
            str(judge),
            "--brama-model",
            MODEL,
        ],
        AUDIT_LOG,
        environment,
    )
    environment["BRAMA_TOKEN"] = ""
    environment["WISENT_APP_AGENT_AUTH_SECRET"] = ""
    write_status(
        state="qualified",
        started_at=started_at,
        finished_at=now(),
        old_model_sha256=old_model_sha256,
        new_model_sha256=digest(q4),
        review_model=MODEL,
    )
    print("lifecycle retraining and audit qualified", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status(state="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        raise
