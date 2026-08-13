#!/usr/bin/env python3
"""Publish one qualified personal-voice model to a private immutable HF revision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    token = os.environ["HF_TOKEN"].strip()
    if not token:
        raise SystemExit("HF_TOKEN is empty")
    repo_id = os.environ.get("HUMANIZER_HF_REPO", "lbartoszcze/lukasz-humanizer-qwen3-4b")
    model_dir = Path(os.environ.get("HUMANIZER_MODEL_DIR", "student"))
    metrics_path = Path(os.environ.get("HUMANIZER_METRICS", "metrics.json"))
    audit_path = Path(os.environ.get("HUMANIZER_AUDIT_OUTPUT", "audit.json"))
    preparation_path = Path(os.environ.get("HUMANIZER_PREPARATION", "preparation.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise SystemExit("refusing to publish an unqualified humanizer")
    if metrics.get("contract") != "echo-lukasz-humanizer-v1":
        raise SystemExit("metrics carry the wrong model contract")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    result = api.upload_folder(
        folder_path=model_dir,
        path_in_repo="",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish qualified Łukasz humanizer model",
    )
    for source, destination in (
        (metrics_path, "evaluation/metrics.json"),
        (audit_path, "evaluation/audit.json"),
        (preparation_path, "evaluation/preparation.json"),
    ):
        result = api.upload_file(
            path_or_fileobj=source,
            path_in_repo=destination,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {destination}",
        )
    api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)
    info = api.model_info(repo_id=repo_id, revision=result.oid, files_metadata=True)
    required = {"config.json", "tokenizer.json", "evaluation/metrics.json", "evaluation/audit.json"}
    files = {item.rfilename: item.size for item in info.siblings}
    missing = sorted(required - files.keys())
    if missing:
        raise SystemExit(f"published model is missing files: {missing}")
    publication = {
        "schema_version": 1,
        "contract": "echo-lukasz-humanizer-v1",
        "repo_id": repo_id,
        "revision": info.sha,
        "private": info.private,
        "files": files,
        "metrics_sha256": digest(metrics_path),
        "audit_sha256": digest(audit_path),
        "qualified": True,
    }
    Path("publication.json").write_text(
        json.dumps(publication, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(publication, ensure_ascii=False))


if __name__ == "__main__":
    main()
