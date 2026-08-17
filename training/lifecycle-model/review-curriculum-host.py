#!/usr/bin/env python3
"""Review deterministic lifecycle curriculum through the authenticated Brama route."""

import json
import os
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd")
BINARY = WORK / "cargo-target/release/transcript-label-trainer"
GRANT_ENV = Path("/root/.stado/files/stado-agent-grant.env")
MODEL = os.environ.get("LIFECYCLE_REVIEW_MODEL", "wisent-backend/chat/primary")
STATUS = WORK / "curriculum-review-status.json"
LOG = WORK / "curriculum-review.log"
JOBS = (
    (WORK / "curriculum-train-raw.jsonl", WORK / "curriculum-train-reviewed.jsonl", "train"),
    (WORK / "curriculum-eval-raw.jsonl", WORK / "curriculum-eval-reviewed.jsonl", "eval"),
)


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


def row_count(path):
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_status(state, **extra):
    payload = {
        "state": state,
        "review_model": MODEL,
        "updated_at": now(),
        "train_rows": row_count(JOBS[0][1]),
        "eval_rows": row_count(JOBS[1][1]),
        **extra,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=WORK, prefix=".curriculum-review-", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(STATUS)


def main():
    for required in (BINARY, GRANT_ENV, JOBS[0][0], JOBS[1][0]):
        if not required.is_file():
            raise RuntimeError(f"required curriculum review input is missing: {required}")
    grant = read_env(GRANT_ENV)
    environment = os.environ.copy()
    environment.update(
        {
            "BRAMA_URL": "https://charless-mac-mini.tail6443b3.ts.net",
            "BRAMA_TOKEN": acquire(grant, "jeden-model-router", "token"),
            "WISENT_APP_AGENT_AUTH_SECRET": acquire(
                grant, "jeden-agent-auth", "agent_auth_secret"
            ),
            "LIFECYCLE_REVIEW_WORKERS": "2",
        }
    )
    write_status("running", started_at=now())
    with LOG.open("ab") as log:
        for source, output, split in JOBS:
            command = [
                str(BINARY),
                "lifecycle-review",
                str(source),
                "--output",
                str(output),
                "--split",
                split,
                "--brama-model",
                MODEL,
            ]
            result = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            write_status(
                "running" if result.returncode == 0 else "failed",
                completed_split=split,
                exit_code=result.returncode,
            )
            if result.returncode:
                return result.returncode
    environment["BRAMA_TOKEN"] = ""
    environment["WISENT_APP_AGENT_AUTH_SECRET"] = ""
    write_status("completed", finished_at=now(), exit_code=0)
    print(
        json.dumps(
            {
                "state": "completed",
                "train_rows": row_count(JOBS[0][1]),
                "eval_rows": row_count(JOBS[1][1]),
                "review_model": MODEL,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status("failed", error=f"{type(error).__name__}: {error}")
        raise
