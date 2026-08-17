#!/usr/bin/env python3
"""Qualify the served GGUF lifecycle model with an independent Brama judge."""

import json
import os
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/mnt/wisent-staging/oko-lifecycle-model-b5de55bd")
PYTHON = WORK / "venv/bin/python"
EVALUATOR = Path("/root/.stado/bin/oko-lifecycle-evaluate-gguf")
BINARY = WORK / "cargo-target/release/transcript-label-trainer"
PREDICTIONS = WORK / "predictions-gguf.jsonl"
JUDGE = WORK / "final-judge-gguf.json"
STATUS = WORK / "gguf-qualification-status.json"
LOG = WORK / "gguf-qualification.log"
GRANT_ENV = Path("/root/.stado/files/stado-agent-grant.env")
MODEL = "wisent-backend/chat/primary"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_env(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key] = value
    return values


def acquire(values, item, field):
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


def write_status(state, **extra):
    payload = {"state": state, "review_model": MODEL, "updated_at": now(), **extra}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=WORK, prefix=".gguf-qualification-", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(STATUS)


def run(command, environment):
    with LOG.open("ab") as log:
        result = subprocess.run(
            command,
            cwd=WORK,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {command[0]}")


def main():
    for required in (PYTHON, EVALUATOR, BINARY, GRANT_ENV):
        if not required.is_file():
            raise RuntimeError(f"required GGUF qualification input is missing: {required}")
    LOG.write_text("", encoding="utf-8")
    write_status("evaluating", started_at=now())
    environment = os.environ.copy()
    run([str(PYTHON), str(EVALUATOR)], environment)

    grant = read_env(GRANT_ENV)
    environment.update(
        {
            "BRAMA_URL": "https://charless-mac-mini.tail6443b3.ts.net",
            "BRAMA_TOKEN": acquire(grant, "jeden-model-router", "token"),
            "WISENT_APP_AGENT_AUTH_SECRET": acquire(
                grant, "jeden-agent-auth", "agent_auth_secret"
            ),
        }
    )
    JUDGE.unlink(missing_ok=True)
    write_status("auditing", started_at=now())
    run(
        [
            str(BINARY),
            "lifecycle-audit",
            str(PREDICTIONS),
            "--output",
            str(JUDGE),
            "--brama-model",
            MODEL,
        ],
        environment,
    )
    environment["BRAMA_TOKEN"] = ""
    environment["WISENT_APP_AGENT_AUTH_SECRET"] = ""
    report = json.loads(JUDGE.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"quantized lifecycle audit rejected candidate: {report.get('counts')}")
    write_status(
        "qualified",
        finished_at=now(),
        counts=report["counts"],
        thresholds=report["thresholds"],
    )
    print(json.dumps({"state": "qualified", "counts": report["counts"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        raise
