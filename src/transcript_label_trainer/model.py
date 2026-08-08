"""Training, inference, and artifact inspection for aspect-label classifiers."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import __version__, lake

# Below this many labeled sessions a TF-IDF + logistic-regression model is not
# meaningful, so train refuses with an explicit message instead of fitting
# noise. This is a product floor, not a sklearn requirement.
MIN_LABELED_SESSIONS = 8

# Cross-validated accuracy is reported once every class can spare members for
# stratified folds; below that the metric would be noise, so it is omitted.
MIN_SESSIONS_FOR_CV = 10

_ASPECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def tlt_home() -> Path:
    return Path(os.environ.get("TLT_HOME", Path.home() / ".transcript-label-trainer"))


def models_dir() -> Path:
    return tlt_home() / "models"


def _aspect_dir(aspect: str) -> Path:
    if not _ASPECT_RE.match(aspect):
        raise ValueError(
            f"invalid aspect name {aspect!r}: use lowercase letters, digits, '-' and '_'"
        )
    return models_dir() / aspect


class NotEnoughData(Exception):
    """Raised when an aspect has too few labeled sessions to train."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _training_frame(aspect: str) -> tuple[list[str], list[str], dict[str, int]]:
    """Labeled sessions joined with their lake text.

    Returns (texts, values, counts_per_value). Raises NotEnoughData with the
    exact numbers when the aspect cannot be trained.
    """
    labels = lake.load_labels(aspect)
    n_labeled = len(labels)
    if n_labeled < MIN_LABELED_SESSIONS:
        raise NotEnoughData(
            f"aspect '{aspect}' has {n_labeled} labeled session(s); "
            f"at least {MIN_LABELED_SESSIONS} are required to train. "
            "Add labels with 'transcript-lake label add' and retry."
        )
    texts_by_id = lake.session_texts(list(labels))
    texts: list[str] = []
    values: list[str] = []
    skipped = 0
    for sid, record in labels.items():
        entry = texts_by_id.get(sid)
        if not entry or not entry["text"].strip():
            skipped += 1
            continue
        texts.append(entry["text"])
        values.append(str(record["value"]))
    if skipped:
        lake.warn(f"{skipped} labeled session(s) had no text in the lake and were skipped")
    distinct = sorted(set(values))
    if len(texts) < MIN_LABELED_SESSIONS or len(distinct) < 2:
        raise NotEnoughData(
            f"aspect '{aspect}' has {len(texts)} usable labeled session(s) "
            f"across {len(distinct)} distinct value(s); at least "
            f"{MIN_LABELED_SESSIONS} sessions and 2 distinct values are "
            "required to train. Add labels with 'transcript-lake label add' "
            "and retry."
        )
    counts = {value: values.count(value) for value in distinct}
    return texts, values, counts


def train(aspect: str) -> dict:
    texts, values, counts = _training_frame(aspect)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    min_class = min(counts.values())
    cv_accuracy = None
    cv_folds = 0
    if len(texts) >= MIN_SESSIONS_FOR_CV and min_class >= 2:
        cv_folds = min(5, min_class)
        scores = cross_val_score(
            pipeline, texts, values, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
        )
        cv_accuracy = round(float(scores.mean()), 4)

    pipeline.fit(texts, values)

    out_dir = _aspect_dir(aspect)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    joblib.dump(pipeline, model_path)
    metrics = {
        "aspect": aspect,
        "trained_at": _now(),
        "trainer_version": __version__,
        "model": "tfidf(1-2gram, sublinear) + logistic-regression",
        "n_sessions": len(texts),
        "classes": sorted(counts),
        "counts": counts,
        "cv_accuracy": cv_accuracy,
        "cv_folds": cv_folds,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _load_model(aspect: str) -> Pipeline:
    model_path = _aspect_dir(aspect) / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model for aspect '{aspect}' at {model_path}; "
            f"run 'transcript-label-trainer train --aspect {aspect}' first"
        )
    return joblib.load(model_path)


def infer(aspect: str, session: str | None = None, limit: int | None = None) -> list[dict]:
    pipeline = _load_model(aspect)

    if session:
        targets = [{"session_id": session}]
    else:
        labeled = set(lake.load_labels(aspect))
        targets = [s for s in lake.all_sessions() if s["session_id"] not in labeled]
        if limit is not None:
            targets = targets[:limit]

    texts_by_id = lake.session_texts([t["session_id"] for t in targets])
    suggestions: list[dict] = []
    for target in targets:
        sid = target["session_id"]
        entry = texts_by_id.get(sid)
        if not entry or not entry["text"].strip():
            continue
        proba = pipeline.predict_proba([entry["text"]])[0]
        best = int(proba.argmax())
        suggestions.append(
            {
                "ts": _now(),
                "session_id": sid,
                "runtime": entry.get("runtime") or target.get("runtime"),
                "aspect": aspect,
                "value": str(pipeline.classes_[best]),
                "note": f"confidence={proba[best]:.2f}",
                "source": "model",
            }
        )
    return suggestions


def info() -> list[dict]:
    """One entry per trained aspect, with artifact paths and metrics."""
    entries: list[dict] = []
    root = models_dir()
    if not root.is_dir():
        return entries
    for aspect_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = aspect_dir / "metrics.json"
        entry: dict = {"aspect": aspect_dir.name, "dir": str(aspect_dir)}
        if metrics_path.is_file():
            entry["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            entry["metrics"] = None
        entries.append(entry)
    return entries
