#!/usr/bin/env python3
"""Stage and publish the recovered qualified lifecycle candidate on its Stado host."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

JOB_ID = os.environ.get("LIFECYCLE_PUBLISH_JOB_ID", "b5de55bd")
SOURCE_REVISION = os.environ.get(
    "LIFECYCLE_PUBLISH_SOURCE_REVISION",
    "11a04eb07c31fa7bb80c6406c2a121b789f84f6c",
)
WORK = Path(
    os.environ.get(
        "LIFECYCLE_PUBLISH_WORK",
        f"/mnt/wisent-staging/oko-lifecycle-model-{JOB_ID}",
    )
)
SOURCE = Path(os.environ.get("LIFECYCLE_PUBLISH_SOURCE", str(WORK / "audit-source")))
OUT = WORK / "qualified-output"
MODEL_NAME = os.environ.get(
    "LIFECYCLE_PUBLISH_MODEL_NAME", "oko-lifecycle-qwen3-8b-q4_k_m.gguf"
)
MODEL = Path(os.environ.get("LIFECYCLE_PUBLISH_MODEL", str(WORK / MODEL_NAME)))
JUDGE = Path(
    os.environ.get("LIFECYCLE_PUBLISH_JUDGE", str(WORK / "final-judge-gguf.json"))
)
STADO = Path(os.environ.get("STADO_BIN", "/root/.stado/bin/stado"))
BASE_MODEL = os.environ.get("LIFECYCLE_PUBLISH_BASE_MODEL", "Qwen/Qwen3-8B")
BASE_REVISION = os.environ.get(
    "LIFECYCLE_PUBLISH_BASE_REVISION",
    "b968826d9c46dd6066d109eabc6255188de91218",
)
DESTINATION_FAMILY = os.environ.get(
    "LIFECYCLE_PUBLISH_DESTINATION_FAMILY", "lifecycle-qwen3-8b"
)
PART_BYTES = 128 * 1024 * 1024
# What the release serves. The first qualified Oko lifecycle model is MLX
# weights, not a GGUF: on 2026-08-18 the same fine-tune measured joint 0.9258
# through mlx_lm on Metal against 0.7381 for its Q4_K_M GGUF on the same
# machine and the same held-out split, so the GGUF path could not pass the gate
# at all. `format` and the companion file list are therefore inputs, not
# constants.
FORMAT = os.environ.get("LIFECYCLE_PUBLISH_FORMAT", "GGUF")
METRICS_NAME = os.environ.get("LIFECYCLE_PUBLISH_METRICS", "metrics-gguf.json")
PREDICTIONS_NAME = os.environ.get(
    "LIFECYCLE_PUBLISH_PREDICTIONS", "predictions-gguf.jsonl"
)
# Extra artefacts the runtime needs beside the weights, as NAME=PATH pairs.
COMPANIONS = [
    entry.split("=", 1)
    for entry in os.environ.get("LIFECYCLE_PUBLISH_COMPANIONS", "").split(",")
    if "=" in entry
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(STADO), *args],
        check=True,
        capture_output=capture,
        text=True,
    )


def publish(destination: str, name: str, source: Path) -> None:
    uri = f"{destination}/{name}"
    probe = run("storage", "stat", uri, "--json", capture=True)
    if json.loads(probe.stdout).get("state") != "present":
        run("storage", "put", uri, str(source))


def split_model() -> list[Path]:
    for stale in OUT.glob(f"{MODEL_NAME}.part-*"):
        stale.unlink()
    parts: list[Path] = []
    with MODEL.open("rb") as source:
        index = 0
        while chunk := source.read(PART_BYTES):
            part = OUT / f"{MODEL_NAME}.part-{index:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    return parts


def main() -> None:
    judge = json.loads(JUDGE.read_text(encoding="utf-8"))
    metrics = json.loads((WORK / METRICS_NAME).read_text(encoding="utf-8"))
    qualified = (
        judge.get("passed") is True
        and metrics.get("valid_json", 0) >= 0.99
        and metrics.get("action_accuracy", 0) >= 0.90
        and metrics.get("joint_accuracy", 0) >= 0.88
        and metrics.get("finish_precision", 0) == 1.0
    )
    if not qualified:
        raise SystemExit("recovered lifecycle candidate is not qualified")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    copies = {
        "final-judge.json": JUDGE,
        "metrics.json": WORK / METRICS_NAME,
        "predictions.jsonl": WORK / PREDICTIONS_NAME,
        "python-requirements.lock": WORK / "python-requirements.lock",
        "lifecycle-system-prompt.txt": SOURCE / "training/lifecycle-model/lifecycle-system-prompt.txt",
        "lifecycle-output-schema.json": SOURCE / "training/lifecycle-model/lifecycle-output-schema.json",
    }
    for name, source in copies.items():
        shutil.copy2(source, OUT / name)
    for name, source in COMPANIONS:
        shutil.copy2(Path(source), OUT / name)
    parts = split_model()

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(OUT.iterdir())
        if path.is_file()
    }
    model_digest = digest(MODEL)
    manifest = {
        "product": "Oko goal lifecycle model",
        "contract": "oko-goal-lifecycle-v1",
        "format": FORMAT,
        "default_artifact": MODEL_NAME,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_revision": SOURCE_REVISION,
        "required_quality_gate": "final-judge.json",
        "evaluation_surface": "served Q4_K_M GGUF through Oko's loopback chat contract",
        "qualified": True,
        "review_model": judge.get("review_model"),
        "metrics": {
            "valid_json": metrics.get("valid_json"),
            "action_accuracy": metrics.get("action_accuracy"),
            "goal_ref_accuracy": metrics.get("goal_ref_accuracy"),
            "evidence_accuracy": metrics.get("evidence_accuracy"),
            "joint_accuracy": metrics.get("joint_accuracy"),
            "finish_precision": metrics.get("finish_precision"),
        },
        "files": files,
        "transport": {
            "kind": "ordered-parts",
            "parts": [part.name for part in parts],
            "assembled_bytes": MODEL.stat().st_size,
            "assembled_sha256": model_digest,
        },
    }
    manifest_path = OUT / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    destination = f"stado://releases/oko/models/{DESTINATION_FAMILY}/{model_digest}"

    for part in parts:
        publish(destination, f"large-output/{part.name}", part)
    for name in copies:
        publish(destination, name, OUT / name)
    # The companions are what makes the weights loadable at all; a release that
    # lists them in its manifest and does not carry them is a release nobody can
    # install, which is exactly how the first MLX publish failed.
    for name, _ in COMPANIONS:
        publish(destination, name, OUT / name)
    chunk_manifest = {
        "filename": MODEL_NAME,
        "sha256": model_digest,
        "bytes": MODEL.stat().st_size,
        "part_count": len(parts),
        "parts": [part.name for part in parts],
    }
    chunk_path = OUT / "large-output-manifest.json"
    chunk_path.write_text(json.dumps(chunk_manifest, indent=2) + "\n", encoding="utf-8")
    publish(destination, "large-output/manifest-v2.json", chunk_path)
    publish(destination, "model-manifest-v2.json", manifest_path)
    print(
        json.dumps(
            {
                "base": destination,
                "digest": model_digest,
                "parts": len(parts),
                "review_model": judge.get("review_model"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
