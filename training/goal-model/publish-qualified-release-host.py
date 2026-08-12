#!/usr/bin/env python3
"""Publish a qualified staged goal model under its immutable digest coordinate."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE = Path("/tmp/jeden-goal-qualified-v6")
manifest = json.loads((SOURCE / "model-manifest.json").read_text(encoding="utf-8"))
if manifest.get("qualified") is not True or manifest.get("required_quality_gate") != "final-judge.json":
    raise SystemExit("model manifest is not qualified")
judge = json.loads((SOURCE / "final-judge.json").read_text(encoding="utf-8"))
if judge.get("passed") is not True or judge.get("complete") is not True:
    raise SystemExit("final judge did not pass")
model = manifest["default_artifact"]
digest = manifest["transport"]["assembled_sha256"]
parts = manifest["transport"]["parts"]
chunk_manifest = {
    "filename": model,
    "sha256": digest,
    "bytes": manifest["transport"]["assembled_bytes"],
    "part_count": len(parts),
    "parts": parts,
}
chunk_path = SOURCE / "large-output-manifest.json"
chunk_path.write_text(json.dumps(chunk_manifest, indent=2) + "\n", encoding="utf-8")
transport_digest = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
base = f"stado://releases/jeden-desktop/models/goal-qwen3-4b/{digest}/transports/{transport_digest}"
stado = "/root/.stado/bin/stado"
environment = os.environ.copy()
environment["STADO_API_TOKEN_FILE"] = "/root/.stado/jeden-desktop-release-publisher-token"
environment.pop("STADO_API_TOKEN", None)
environment["STADO_API_URL"] = "https://charless-mac-mini.tail6443b3.ts.net:8443"
def publish(name, source):
    uri = f"{base}/{name}"
    probe = subprocess.run([stado, "storage", "get", uri, "/tmp/stado-release-existing"], env=environment)
    if probe.returncode == 0:
        return
    subprocess.run([stado, "storage", "put", uri, str(source)], check=True, env=environment)
for name in ("model-manifest.json", "final-judge.json", "metrics.json", "predictions.jsonl", "goal-system-prompt.md", "python-requirements.lock"):
    publish(name, SOURCE / name)
publish("large-output/manifest.json", chunk_path)
for name in parts:
    publish(f"large-output/{name}", SOURCE / name)
print(json.dumps({"base": base, "digest": digest, "parts": len(parts)}, sort_keys=True))
