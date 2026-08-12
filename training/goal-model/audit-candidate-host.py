#!/usr/bin/env python3
"""Audit the staged candidate using worker-scoped Skarbiec credentials."""

import json
import os
from pathlib import Path
import urllib.request

WORK = Path(os.environ.get(
    "GOAL_AUDIT_WORK",
    "/mnt/wd16tb/stado/jobs/jeden-goal-prompt-v5",
))
BINARY = Path("/mnt/wd16tb/stado/jobs/jeden-goal-568ebd79663775c9/cargo-target/release/transcript-label-trainer")
GRANT = Path("/root/.stado/files/stado-agent-grant.env")
values = {}
for raw in GRANT.read_text().splitlines():
    key, separator, value = raw.partition("=")
    if separator:
        values[key] = value

url = values["WC_AGENT_SKARBIEC_URL"].rstrip("/") + "/v1/items/read"
token = Path(values["WC_AGENT_SKARBIEC_TOKEN_FILE"]).read_text().strip()

def read_secret(item, field):
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
    with urllib.request.urlopen(request, timeout=15) as response:
        value = json.load(response).get("value")
    if not isinstance(value, str) or not value:
        raise SystemExit(f"empty Skarbiec field: {item}#{field}")
    return value

environment = os.environ.copy()
environment["BRAMA_TOKEN"] = read_secret("jeden-model-router", "token")
environment["WISENT_APP_AGENT_AUTH_SECRET"] = read_secret(
    "jeden-agent-auth", "agent_auth_secret"
)
environment["BRAMA_URL"] = "http://127.0.0.1:17601"
environment["TLT_STADO_BIN"] = "/root/.stado/bin/stado"
os.execve(
    BINARY,
    [
        str(BINARY),
        "goal-audit",
        str(WORK / "predictions-corrected-v3.jsonl"),
        "--output",
        str(WORK / "final-judge-corrected-v3.json"),
        "--best",
    ],
    environment,
)
