"""Declarative training jobs: load and validate a YAML job spec.

A job spec captures the four things an operator declares per training run:
WHO evaluated the transcripts (evaluator — the exact label-store source that
counts as ground truth), WHICH model to train (model), the SCOPE of training
data (scope), and the TASK (free text stored with the artifacts).

Two more sections govern how the run is judged, and both are ON unless the
spec turns them off: ``eval_split`` freezes a holdout of labeled sessions that
training never sees, and ``judge`` has a Brama-routed teacher rule on whether
the trained model's holdout predictions are acceptable.

Every field is validated here; invalid specs fail with clear errors and no
silent defaults.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import brama

# The lake labeler's source provenance grammar.
SOURCE_PATTERN = re.compile(r"^(manual|human|model|brama)(:[A-Za-z0-9._/-]+)?$")

# Job names become artifact directory names under <training root>/models/.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The one reserved model name: the existing sklearn backend. Anything else is
# a HuggingFace model id and selects the HF backend.
SKLEARN_MODEL = "tfidf-logreg"

# The frozen evaluation split is on by default: model comparisons over time
# have to run on the same untouched sessions, so the holdout is decided once
# and persisted next to the artifacts. The seed is fixed so a first run on the
# same labels always picks the same sessions.
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_EVAL_SEED = 20260808

# A fraction above this would starve training rather than measure it.
MAX_EVAL_FRACTION = 0.5

_TOP_LEVEL_KEYS = {"name", "task", "evaluator", "model", "scope", "eval_split", "judge"}
_SCOPE_KEYS = {"aspect", "runtimes", "since", "values", "min_text_chars"}
_EVAL_SPLIT_KEYS = {"fraction", "seed"}
_JUDGE_KEYS = {"model"}


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


def default_eval_split() -> dict:
    """The frozen split every run gets unless the spec says ``false``."""
    return {
        "enabled": True,
        "fraction": DEFAULT_EVAL_FRACTION,
        "seed": DEFAULT_EVAL_SEED,
    }


def default_judge() -> dict:
    """The Brama teacher verdict every run gets unless the spec says ``false``."""
    return {"enabled": True, "model": brama.DEFAULT_MODEL}


def _eval_split(raw: dict) -> dict:
    """Validate the eval_split section. Absent means the default, on."""
    value = raw.get("eval_split")
    if value is None or value is True:
        return default_eval_split()
    if value is False:
        return {"enabled": False, "fraction": None, "seed": None}
    if not isinstance(value, dict):
        raise JobError(
            "'eval_split' must be a mapping with 'fraction' and/or 'seed', "
            "true for the defaults, or false to train on every labeled session"
        )
    unknown = sorted(set(value) - _EVAL_SPLIT_KEYS)
    if unknown:
        raise JobError(f"unknown eval_split field(s): {', '.join(unknown)}")

    fraction = value.get("fraction")
    if fraction is None:
        fraction = DEFAULT_EVAL_FRACTION
    elif isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise JobError(
            f"eval_split.fraction must be a number, got {fraction!r}"
        )
    elif not 0 < float(fraction) <= MAX_EVAL_FRACTION:
        raise JobError(
            f"eval_split.fraction must be greater than 0 and at most "
            f"{MAX_EVAL_FRACTION}, got {fraction!r}"
        )

    seed = value.get("seed")
    if seed is None:
        seed = DEFAULT_EVAL_SEED
    elif isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise JobError(f"eval_split.seed must be a non-negative integer, got {seed!r}")

    return {"enabled": True, "fraction": float(fraction), "seed": int(seed)}


def _judge(raw: dict) -> dict:
    """Validate the judge section. Absent means the default teacher, on."""
    value = raw.get("judge")
    if value is None or value is True:
        return default_judge()
    if value is False:
        return {"enabled": False, "model": None}
    if not isinstance(value, dict):
        raise JobError(
            "'judge' must be a mapping with 'model', true for the default "
            f"teacher ({brama.DEFAULT_MODEL}), or false to skip the verdict"
        )
    unknown = sorted(set(value) - _JUDGE_KEYS)
    if unknown:
        raise JobError(f"unknown judge field(s): {', '.join(unknown)}")
    model = value.get("model")
    if model is None:
        model = brama.DEFAULT_MODEL
    elif not isinstance(model, str) or not model.strip():
        raise JobError("judge.model must be a non-empty Brama-routed model id")
    return {"enabled": True, "model": model.strip()}


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
        "eval_split": _eval_split(raw),
        "judge": _judge(raw),
    }
