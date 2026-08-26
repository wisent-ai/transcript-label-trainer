#!/usr/bin/env python3
"""Publish one qualified goal GGUF to its public Hugging Face model repository."""

import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HOME = Path.home()
CONFIG = HOME / ".stado" / "files" / "jeden-goal-huggingface-publish.json"
TOKEN_FILE = HOME / ".stado" / "huggingface-token"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def required_path(config: dict[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"publication config has no {key}")
    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"publication input is absent: {path}")
    return path

def retry_rate_limit(operation):
    for attempt in range(5):
        try:
            return operation()
        except HfHubHTTPError as error:
            if error.response.status_code != 429 or attempt == 4:
                raise
            time.sleep(min(30 * (2**attempt), 300))


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo_id = config.get("repo_id")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise SystemExit("publication config has no valid repo_id")

    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("Hugging Face token file is empty")
    api = HfApi(token=token)

    model = required_path(config, "model")
    card = Path(__file__).resolve().parents[2] / "README.md"
    prompt = required_path(config, "prompt")
    metrics = required_path(config, "metrics")
    license_file = Path(
        retry_rate_limit(
            lambda: hf_hub_download(
                repo_id="Qwen/Qwen3-4B",
                filename="LICENSE",
                repo_type="model",
                token=token,
            )
        )
    )
    expected_bytes = config.get("model_bytes")
    expected_sha256 = config.get("model_sha256")
    if model.stat().st_size != expected_bytes:
        raise SystemExit("model byte count does not match the qualified manifest")
    actual_sha256 = sha256(model)
    if actual_sha256 != expected_sha256:
        raise SystemExit("model SHA-256 does not match the qualified manifest")

    uploads = [
        (model, model.name),
        (prompt, "goal-system-prompt.md"),
        (metrics, "metrics.json"),
        (license_file, "LICENSE"),
        (card, "README.md"),
    ]
    revision = None
    for source, destination in uploads:
        result = retry_rate_limit(
            lambda: api.upload_file(
                path_or_fileobj=source,
                path_in_repo=destination,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Publish {destination}",
            )
        )
        revision = result.oid

    retry_rate_limit(
        lambda: api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
    )
    info = retry_rate_limit(
        lambda: api.model_info(repo_id=repo_id, files_metadata=True)
    )
    files = {item.rfilename: item.size for item in info.siblings}
    expected_files = {destination: source.stat().st_size for source, destination in uploads}
    if any(files.get(name) != size for name, size in expected_files.items()):
        raise SystemExit("published Hugging Face files do not match local inputs")

    print(
        json.dumps(
            {
                "repo": f"https://huggingface.co/{repo_id}",
                "revision": revision,
                "model_sha256": actual_sha256,
                "model_bytes": model.stat().st_size,
                "files": expected_files,
                "private": info.private,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
