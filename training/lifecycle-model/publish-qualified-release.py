#!/usr/bin/env python3
"""Publish one qualified Oko lifecycle model from a Stado job artifact."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(stado, *args, capture=False):
    return subprocess.run(
        [stado, *args],
        check=True,
        capture_output=capture,
        text=capture,
    )


def fetch(stado, source, name, destination):
    run(stado, "storage", "get", f"{source}/{name}", str(destination))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f"sha256 mismatch for {path.name}: expected {expected}, found {actual}")


def publish(stado, destination, name, source):
    uri = f"{destination}/{name}"
    probe = run(stado, "storage", "stat", uri, "--json", capture=True)
    if json.loads(probe.stdout).get("state") != "present":
        run(stado, "storage", "put", uri, str(source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Stado job output URI")
    parser.add_argument("--stado", default=os.environ.get("STADO_BIN", "stado"))
    args = parser.parse_args()
    source = args.source.rstrip("/")

    with tempfile.TemporaryDirectory(prefix="oko-lifecycle-release-") as temporary:
        root = Path(temporary)
        manifest_path = root / "model-manifest.json"
        fetch(args.stado, source, "model-manifest.json", manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("qualified") is not True:
            raise SystemExit("lifecycle model manifest is not qualified")
        if manifest.get("contract") != "oko-goal-lifecycle-v1":
            raise SystemExit("lifecycle model manifest has the wrong contract")
        if manifest.get("required_quality_gate") != "final-judge.json":
            raise SystemExit("lifecycle model manifest has the wrong quality gate")

        files = manifest.get("files", {})
        metadata_names = (
            "final-judge.json",
            "metrics.json",
            "predictions.jsonl",
            "lifecycle-system-prompt.txt",
            "lifecycle-output-schema.json",
            "python-requirements.lock",
        )
        for name in metadata_names:
            path = root / name
            fetch(args.stado, source, name, path)
            verify(path, files[name]["sha256"])

        judge = json.loads((root / "final-judge.json").read_text(encoding="utf-8"))
        if judge.get("passed") is not True:
            raise SystemExit("lifecycle final judge did not pass")

        transport = manifest["transport"]
        model_name = manifest["default_artifact"]
        model_digest = transport["assembled_sha256"]
        destination = f"stado://releases/oko/models/lifecycle-qwen3-4b/{model_digest}"
        parts = transport["parts"]
        part_paths = []
        assembled = hashlib.sha256()
        assembled_bytes = 0
        for name in parts:
            path = root / name
            fetch(args.stado, source, name, path)
            verify(path, files[name]["sha256"])
            with path.open("rb") as source_part:
                while chunk := source_part.read(8 * 1024 * 1024):
                    assembled.update(chunk)
            assembled_bytes += path.stat().st_size
            part_paths.append(path)
        if assembled.hexdigest() != model_digest or assembled_bytes != transport["assembled_bytes"]:
            raise SystemExit("ordered model parts do not reconstruct the qualified artifact")

        for path in part_paths:
            publish(args.stado, destination, f"large-output/{path.name}", path)
        for name in metadata_names:
            publish(args.stado, destination, name, root / name)

        chunk_manifest = {
            "filename": model_name,
            "sha256": model_digest,
            "bytes": assembled_bytes,
            "part_count": len(parts),
            "parts": parts,
        }
        chunk_path = root / "large-output-manifest.json"
        chunk_path.write_text(json.dumps(chunk_manifest, indent=2) + "\n", encoding="utf-8")
        publish(args.stado, destination, "large-output/manifest-v2.json", chunk_path)
        publish(args.stado, destination, "model-manifest-v2.json", manifest_path)
        print(json.dumps({"base": destination, "digest": model_digest, "parts": len(parts)}, sort_keys=True))


if __name__ == "__main__":
    main()
