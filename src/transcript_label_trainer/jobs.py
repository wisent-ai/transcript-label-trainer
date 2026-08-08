"""Declarative training jobs: load and validate a YAML job spec.

A job spec captures the four things an operator declares per training run:
WHO evaluated the transcripts (evaluator — the exact label-store source that
counts as ground truth), WHICH model to train (model), the SCOPE of training
data (scope), and the TASK (free text stored with the artifacts).

Every field is validated here; invalid specs fail with clear errors and no
silent defaults.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

# The lake labeler's source provenance grammar.
SOURCE_PATTERN = re.compile(r"^(manual|human|model|brama)(:[A-Za-z0-9._/-]+)?$")

# Job names become artifact directory names under $TLT_HOME/models/.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The one reserved model name: the existing sklearn backend. Anything else is
# a HuggingFace model id and selects the HF backend.
SKLEARN_MODEL = "tfidf-logreg"

_TOP_LEVEL_KEYS = {"name", "task", "evaluator", "model", "scope"}
_SCOPE_KEYS = {"aspect", "runtimes", "since", "values", "min_text_chars"}


class JobError(ValueError):
    """Raised when a job spec is invalid."""


def _require_string(spec: dict, key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JobError(f"'{key}' must be a non-empty string")
    return value.strip()


def _string_list(scope: dict, key: str) -> list[str] | None:
    value = scope.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise JobError(f"scope.{key} must be a non-empty list of strings")
    return [item.strip() for item in value]


def load(path: str) -> dict:
    """Load and fully validate a job spec file. Raises JobError."""
    spec_path = Path(path)
    if not spec_path.is_file():
        raise JobError(f"job file not found: {path}")
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JobError(f"job file {path} is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        raise JobError(f"job file {path} must contain a YAML mapping at the top level")

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise JobError(f"unknown job field(s): {', '.join(unknown)}")

    name = _require_string(raw, "name")
    if not NAME_PATTERN.match(name):
        raise JobError(
            f"'name' {name!r} must match {NAME_PATTERN.pattern} "
            "(it becomes the artifact directory name)"
        )

    task = _require_string(raw, "task")

    evaluator = _require_string(raw, "evaluator")
    if not SOURCE_PATTERN.match(evaluator):
        raise JobError(
            f"'evaluator' {evaluator!r} must match the label-store source "
            f"grammar {SOURCE_PATTERN.pattern} "
            "(manual, human, model or brama, with an optional :detail suffix)"
        )

    model = _require_string(raw, "model")

    scope = raw.get("scope")
    if not isinstance(scope, dict):
        raise JobError("'scope' must be a mapping with at least 'aspect'")
    unknown_scope = sorted(set(scope) - _SCOPE_KEYS)
    if unknown_scope:
        raise JobError(f"unknown scope field(s): {', '.join(unknown_scope)}")

    aspect = scope.get("aspect")
    if not isinstance(aspect, str) or not aspect.strip():
        raise JobError("scope.aspect is required and must be a non-empty string")

    since_raw = scope.get("since")
    since = None
    if since_raw is not None:
        if isinstance(since_raw, datetime):
            since = since_raw
        elif isinstance(since_raw, str):
            try:
                since = datetime.fromisoformat(since_raw.strip().replace("Z", "+00:00"))
            except ValueError:
                raise JobError(
                    f"scope.since {since_raw!r} is not an ISO date, e.g. \"2026-07-01\""
                )
        else:
            raise JobError(f"scope.since must be an ISO date string, got {since_raw!r}")
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    min_text_chars = scope.get("min_text_chars")
    if min_text_chars is not None:
        if not isinstance(min_text_chars, int) or isinstance(min_text_chars, bool) or min_text_chars < 0:
            raise JobError("scope.min_text_chars must be a non-negative integer")

    return {
        "name": name,
        "task": task,
        "evaluator": evaluator,
        "model": model,
        "scope": {
            "aspect": aspect.strip(),
            "runtimes": _string_list(scope, "runtimes"),
            "since": since,
            "values": _string_list(scope, "values"),
            "min_text_chars": min_text_chars,
        },
    }
