#!/usr/bin/env python3
"""Re-review legacy lifecycle labels against the current serving contract."""

import json
import os
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/mnt/wisent-staging/oko-lifecycle-model-b5de55bd")
BINARY = WORK / "cargo-target/release/transcript-label-trainer"
GRANT_ENV = Path("/root/.stado/files/stado-agent-grant.env")
MODEL = "wisent-backend/chat/primary"
STATUS = WORK / "source-rereview-status.json"
LOG = WORK / "source-rereview.log"
JOBS = (
    (WORK / "reviewed-train.jsonl", WORK / "rereviewed-train-source.jsonl", "train"),
    (WORK / "reviewed-eval.jsonl", WORK / "rereviewed-eval-source.jsonl", "eval"),
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
        mode="w", encoding="utf-8", dir=WORK, prefix=".source-rereview-", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(STATUS)


def main():
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
            result = subprocess.run(
                [
                    str(BINARY),
                    "lifecycle-review",
                    str(source),
                    "--output",
                    str(output),
                    "--split",
                    split,
                    "--brama-model",
                    MODEL,
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode:
                write_status("failed", failed_split=split, exit_code=result.returncode)
                return result.returncode
            write_status("running", completed_split=split, exit_code=0)
    environment["BRAMA_TOKEN"] = ""
    environment["WISENT_APP_AGENT_AUTH_SECRET"] = ""
    write_status("completed", finished_at=now(), exit_code=0)
    print(json.dumps({"state": "completed", "train_rows": row_count(JOBS[0][1]), "eval_rows": row_count(JOBS[1][1])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status("failed", error=f"{type(error).__name__}: {error}")
        raise
