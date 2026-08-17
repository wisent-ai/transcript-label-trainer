#!/usr/bin/env python3
"""Recover the completed lifecycle candidate with an explicit Brama judge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = "b5de55bd"
WORK = Path(f"/mnt/wisent-staging/oko-lifecycle-model-{JOB_ID}")
BINARY = WORK / "cargo-target/release/transcript-label-trainer"
PREDICTIONS = WORK / "predictions.jsonl"
MODEL = os.environ.get("LIFECYCLE_AUDIT_MODEL", "-best").strip()
LABEL = os.environ.get("LIFECYCLE_AUDIT_LABEL", "best").strip()
if not LABEL or not all(character.isalnum() or character in "-_" for character in LABEL):
    raise RuntimeError("LIFECYCLE_AUDIT_LABEL must be a safe filename component")
OUTPUT = WORK / f"final-judge-{LABEL}.json"
STATUS = WORK / f"{LABEL}-audit-status.json"
LOG = WORK / f"{LABEL}-audit.log"
GRANT_ENV = Path("/root/.stado/files/stado-agent-grant.env")
BRAMA_URL = "https://charless-mac-mini.tail6443b3.ts.net"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_status(value: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=WORK, prefix=".codex-audit-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(STATUS)


def main() -> int:
    started_at = now()
    for required in (BINARY, PREDICTIONS, GRANT_ENV):
        if not required.is_file():
            raise RuntimeError(f"required audit input is missing: {required}")

    grant = read_env(GRANT_ENV)
    environment = os.environ.copy()
    environment.update(
        {
            "BRAMA_URL": BRAMA_URL,
            "BRAMA_TOKEN": acquire(grant, "jeden-model-router", "token"),
            "WISENT_APP_AGENT_AUTH_SECRET": acquire(
                grant, "jeden-agent-auth", "agent_auth_secret"
            ),
        }
    )
    OUTPUT.unlink(missing_ok=True)
    write_status(
        {
            "state": "running",
            "job_id": JOB_ID,
            "review_model": MODEL,
            "started_at": started_at,
        }
    )
    print(f"lifecycle {LABEL} audit started at {started_at}", flush=True)
    command = [
        str(BINARY),
        "lifecycle-audit",
        str(PREDICTIONS),
        "--output",
        str(OUTPUT),
    ]
    command.extend(["--best"] if MODEL == "-best" else ["--brama-model", MODEL])
    with LOG.open("wb") as log:
        result = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    environment["BRAMA_TOKEN"] = ""
    environment["WISENT_APP_AGENT_AUTH_SECRET"] = ""
    output_bytes = OUTPUT.stat().st_size if OUTPUT.is_file() else 0
    state = "completed" if result.returncode == 0 and output_bytes else "failed"
    write_status(
        {
            "state": state,
            "job_id": JOB_ID,
            "review_model": MODEL,
            "started_at": started_at,
            "finished_at": now(),
            "exit_code": result.returncode,
            "output_bytes": output_bytes,
        }
    )
    print(
        f"lifecycle {LABEL} audit {state} exit={result.returncode} bytes={output_bytes}",
        flush=True,
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status(
            {
                "state": "failed",
                "job_id": JOB_ID,
                "review_model": MODEL,
                "finished_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        print(
            f"lifecycle {LABEL} audit failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
