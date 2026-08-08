"""Training, inference, and artifact inspection for aspect-label classifiers.

Two backends share one data path (lake label store + lake CLI session text):

- sklearn (default): TF-IDF + logistic regression, artifacts directly in
  ``$TLT_HOME/models/<aspect>/`` (``model.joblib`` + ``metrics.json``);
- hf (``--model <hf-model-id>``): fine-tuned HuggingFace sequence classifier,
  artifacts in ``$TLT_HOME/models/<aspect>/hf-<sanitized-model-id>/``
  (``save_pretrained`` output + ``metrics.json``).

When both backends exist for an aspect, inference uses the newest artifact by
``trained_at``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import __version__, jobs, lake

# Below this many labeled sessions a classifier is not meaningful, so train
# refuses with an explicit message instead of fitting noise. This is a product
# floor, not a library requirement.
MIN_LABELED_SESSIONS = 8

# Cross-validated accuracy is reported once every class can spare members for
# stratified folds; below that the metric would be noise, so it is omitted.
MIN_SESSIONS_FOR_CV = 10

# HF fine-tuning additionally needs 2 sessions per class so the stratified
# holdout split keeps every class on both sides.
MIN_PER_CLASS_HF = 2

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


class HfExtraMissing(Exception):
    """Raised when --model is used without the optional 'hf' extra installed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _frame_from_labels(
    labels: dict[str, dict], subject: str, min_text_chars: int | None = None
) -> tuple[list[str], list[str], dict[str, int]]:
    """Preselected label records joined with their lake text.

    ``subject`` names the selection in error messages ("aspect 'topic'" for
    train, "job 'topic-v1' (…)" for run). Returns (texts, values,
    counts_per_value). Raises NotEnoughData with the exact numbers when the
    selection cannot be trained.
    """
    n_labeled = len(labels)
    if n_labeled < MIN_LABELED_SESSIONS:
        raise NotEnoughData(
            f"{subject} has {n_labeled} labeled session(s); "
            f"at least {MIN_LABELED_SESSIONS} are required to train. "
            "Add labels with 'transcript-lake label add' and retry."
        )
    texts_by_id = lake.session_texts(list(labels))
    texts: list[str] = []
    values: list[str] = []
    skipped = 0
    skipped_short = 0
    for sid, record in labels.items():
        entry = texts_by_id.get(sid)
        if not entry or not entry["text"].strip():
            skipped += 1
            continue
        if min_text_chars is not None and len(entry["text"]) < min_text_chars:
            skipped_short += 1
            continue
        texts.append(entry["text"])
        values.append(str(record["value"]))
    if skipped:
        lake.warn(f"{skipped} labeled session(s) had no text in the lake and were skipped")
    if skipped_short:
        lake.warn(
            f"{skipped_short} labeled session(s) were shorter than "
            f"scope.min_text_chars={min_text_chars} and were skipped"
        )
    distinct = sorted(set(values))
    if len(texts) < MIN_LABELED_SESSIONS or len(distinct) < 2:
        raise NotEnoughData(
            f"{subject} has {len(texts)} usable labeled session(s) "
            f"across {len(distinct)} distinct value(s); at least "
            f"{MIN_LABELED_SESSIONS} sessions and 2 distinct values are "
            "required to train. Add labels with 'transcript-lake label add' "
            "and retry."
        )
    counts = {value: values.count(value) for value in distinct}
    return texts, values, counts


def _training_frame(aspect: str) -> tuple[list[str], list[str], dict[str, int]]:
    """Labeled sessions joined with their lake text (train command path)."""
    return _frame_from_labels(lake.load_labels(aspect), f"aspect '{aspect}'")


def _base_metrics(aspect: str, backend: str, model_desc: str, texts, counts) -> dict:
    return {
        "aspect": aspect,
        "backend": backend,
        "trained_at": _now(),
        "trainer_version": __version__,
        "model": model_desc,
        "n_sessions": len(texts),
        "classes": sorted(counts),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# sklearn backend
# ---------------------------------------------------------------------------


def _train_sklearn(
    aspect: str,
    frame: tuple | None = None,
    out_name: str | None = None,
    job: dict | None = None,
) -> dict:
    texts, values, counts = frame if frame is not None else _training_frame(aspect)

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

    out_dir = _aspect_dir(out_name or aspect)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    metrics = _base_metrics(aspect, "sklearn", "tfidf(1-2gram, sublinear) + logistic-regression", texts, counts)
    metrics.update(
        {
            "cv_accuracy": cv_accuracy,
            "cv_folds": cv_folds,
            "model_path": str(model_path),
        }
    )
    if job is not None:
        metrics["job"] = job
    joblib.dump(pipeline, model_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# HuggingFace backend (optional 'hf' extra: torch + transformers)
# ---------------------------------------------------------------------------


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("-")


def _hf_imports():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        raise HfExtraMissing(
            "fine-tuning with --model requires the optional 'hf' extra; "
            "install it with: pip install '.[hf]' (adds torch and transformers)"
        )
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def _hf_device(torch) -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _TextDataset:
    def __init__(self, encodings, label_ids):
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item["labels"] = self.label_ids[index]
        return item


def _train_hf(
    aspect: str,
    model_id: str,
    epochs: float,
    batch_size: int,
    lr: float,
    max_length: int,
    frame: tuple | None = None,
    out_name: str | None = None,
    job: dict | None = None,
) -> dict:
    torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments = _hf_imports()

    texts, values, counts = frame if frame is not None else _training_frame(aspect)
    too_small = {value: n for value, n in counts.items() if n < MIN_PER_CLASS_HF}
    if too_small:
        detail = ", ".join(f"'{value}' has {n}" for value, n in sorted(too_small.items()))
        raise NotEnoughData(
            f"aspect '{aspect}': HF fine-tuning requires at least "
            f"{MIN_PER_CLASS_HF} sessions per class ({detail}). "
            "Add labels with 'transcript-lake label add' and retry."
        )

    classes = sorted(counts)
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    label_ids = [label2id[value] for value in values]

    # Stratified holdout for an honest eval number; the test side must hold at
    # least one session per class.
    n_test = max(len(classes), round(len(texts) * 0.2))
    x_train, x_eval, y_train, y_eval = train_test_split(
        texts, label_ids, test_size=n_test, stratify=label_ids, random_state=0
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(classes), id2label=id2label, label2id=label2id
    )

    def encode(batch_texts):
        return tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True)

    train_ds = _TextDataset(encode(x_train), y_train)
    eval_ds = _TextDataset(encode(x_eval), y_eval)

    out_dir = _aspect_dir(out_name or aspect) / f"hf-{_sanitize_model_id(model_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import tempfile

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": float((preds == gold).mean())}

    device = _hf_device(torch)
    with tempfile.TemporaryDirectory(prefix="tlt-hf-") as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            seed=0,
            report_to=[],
            disable_tqdm=True,
            save_strategy="no",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        eval_result = trainer.evaluate()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics = _base_metrics(aspect, "hf", f"fine-tuned {model_id} (sequence classification)", texts, counts)
    metrics.update(
        {
            "base_model": model_id,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
            },
            "device": device,
            "eval_accuracy": round(float(eval_result.get("eval_accuracy", 0.0)), 4)
            if "eval_accuracy" in eval_result
            else None,
            "eval_loss": round(float(eval_result.get("eval_loss", 0.0)), 4)
            if "eval_loss" in eval_result
            else None,
            "eval_sessions": len(x_eval),
            "model_path": str(out_dir),
        }
    )
    if job is not None:
        metrics["job"] = job
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def train(
    aspect: str,
    model_id: str | None = None,
    epochs: float = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 512,
) -> dict:
    if model_id is None:
        return _train_sklearn(aspect)
    return _train_hf(aspect, model_id, epochs, batch_size, lr, max_length)


# ---------------------------------------------------------------------------
# Declarative training jobs (run command)
# ---------------------------------------------------------------------------


def _label_ts(record: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(record.get("ts", "")).replace("Z", "+00:00"))
    except ValueError:
        return None


def _select_labels(job: dict) -> dict[str, dict]:
    """Label records matching the job's evaluator and scope filters."""
    scope = job["scope"]
    labels = lake.load_labels(scope["aspect"])
    evaluator = job["evaluator"]
    selected = {
        sid: record
        for sid, record in labels.items()
        if str(record.get("source", "")).strip() == evaluator
    }
    if scope["runtimes"]:
        allowed = set(scope["runtimes"])
        selected = {
            sid: record
            for sid, record in selected.items()
            if record.get("runtime") in allowed
        }
    if scope["since"] is not None:
        since = scope["since"]
        selected = {
            sid: record
            for sid, record in selected.items()
            if (_label_ts(record) or datetime.min.replace(tzinfo=timezone.utc)) >= since
        }
    if scope["values"]:
        allowed_values = set(scope["values"])
        selected = {
            sid: record
            for sid, record in selected.items()
            if str(record.get("value")) in allowed_values
        }
    return selected


def _job_scope_json(scope: dict) -> dict:
    return {
        "aspect": scope["aspect"],
        "runtimes": scope["runtimes"],
        "since": scope["since"].isoformat() if scope["since"] is not None else None,
        "values": scope["values"],
        "min_text_chars": scope["min_text_chars"],
    }


def resolve_job(job: dict) -> dict:
    """Resolve a validated job spec to its training data, without training.

    Returns the selected labels plus per-class counts for the resolved summary.
    """
    labels = _select_labels(job)
    counts: dict[str, int] = {}
    for record in labels.values():
        value = str(record.get("value"))
        counts[value] = counts.get(value, 0) + 1
    return {"labels": labels, "counts": dict(sorted(counts.items()))}


def job_summary(job: dict, resolved: dict) -> dict:
    """The resolved-summary printed before training."""
    return {
        "name": job["name"],
        "task": job["task"],
        "evaluator": job["evaluator"],
        "model": job["model"],
        "scope": _job_scope_json(job["scope"]),
        "sessions_found": len(resolved["labels"]),
        "counts": resolved["counts"],
    }


def run_job(job: dict, resolved: dict) -> dict:
    """Train from a resolved job and persist spec copy + job metadata."""
    scope = job["scope"]
    subject = (
        f"job '{job['name']}' (evaluator '{job['evaluator']}', "
        f"aspect '{scope['aspect']}')"
    )
    frame = _frame_from_labels(
        resolved["labels"], subject, min_text_chars=scope["min_text_chars"]
    )
    job_meta = {
        "name": job["name"],
        "task": job["task"],
        "evaluator": job["evaluator"],
        "scope": _job_scope_json(scope),
    }
    if job["model"] == jobs.SKLEARN_MODEL:
        metrics = _train_sklearn(
            scope["aspect"], frame=frame, out_name=job["name"], job=job_meta
        )
    else:
        metrics = _train_hf(
            scope["aspect"], job["model"], 3, 8, 2e-5, 512,
            frame=frame, out_name=job["name"], job=job_meta,
        )
    spec_copy = dict(job_meta, model=job["model"])
    (_aspect_dir(job["name"]) / "job.yaml").write_text(
        yaml.safe_dump(spec_copy, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _artifacts(aspect: str) -> list[dict]:
    """All trained artifacts for one aspect, oldest first."""
    base = _aspect_dir(aspect)
    found: list[dict] = []
    sklearn_metrics = base / "metrics.json"
    if (base / "model.joblib").is_file() and sklearn_metrics.is_file():
        metrics = json.loads(sklearn_metrics.read_text(encoding="utf-8"))
        found.append({"backend": "sklearn", "dir": base, "metrics": metrics})
    if base.is_dir():
        for sub in sorted(base.glob("hf-*")):
            metrics_path = sub / "metrics.json"
            if sub.is_dir() and metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                found.append({"backend": "hf", "dir": sub, "metrics": metrics})
    return sorted(found, key=lambda a: a["metrics"].get("trained_at", ""))


def _active_artifact(aspect: str) -> dict:
    artifacts = _artifacts(aspect)
    if not artifacts:
        raise FileNotFoundError(
            f"no trained model for aspect '{aspect}' under {_aspect_dir(aspect)}; "
            f"run 'transcript-label-trainer train --aspect {aspect}' first"
        )
    return artifacts[-1]  # newest wins


def _infer_sklearn(artifact: dict, texts: list[str]) -> list[tuple[str, float]]:
    pipeline = joblib.load(artifact["dir"] / "model.joblib")
    probas = pipeline.predict_proba(texts)
    return [
        (str(pipeline.classes_[int(proba.argmax())]), float(proba.max()))
        for proba in probas
    ]


def _infer_hf(artifact: dict, texts: list[str]) -> list[tuple[str, float]]:
    torch, AutoModelForSequenceClassification, AutoTokenizer, _, _ = _hf_imports()
    out_dir = artifact["dir"]
    max_length = int(artifact["metrics"].get("hyperparameters", {}).get("max_length", 512))
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    device = _hf_device(torch)
    model.to(device)
    model.eval()
    results: list[tuple[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            batch = texts[start : start + 8]
            encoded = tokenizer(
                batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
            ).to(device)
            probas = torch.softmax(model(**encoded).logits, dim=-1)
            for proba in probas:
                best = int(proba.argmax())
                results.append((str(model.config.id2label[best]), float(proba[best])))
    return results


def infer(aspect: str, session: str | None = None, limit: int | None = None) -> list[dict]:
    artifact = _active_artifact(aspect)

    if session:
        targets = [{"session_id": session}]
    else:
        labeled = set(lake.load_labels(aspect))
        targets = [s for s in lake.all_sessions() if s["session_id"] not in labeled]
        if limit is not None:
            targets = targets[:limit]

    texts_by_id = lake.session_texts([t["session_id"] for t in targets])
    usable = [
        (t, texts_by_id[t["session_id"]])
        for t in targets
        if texts_by_id.get(t["session_id"], {}).get("text", "").strip()
    ]
    if not usable:
        return []

    predict = _infer_sklearn if artifact["backend"] == "sklearn" else _infer_hf
    predictions = predict(artifact, [entry["text"] for _, entry in usable])

    suggestions: list[dict] = []
    for (target, entry), (value, confidence) in zip(usable, predictions):
        suggestions.append(
            {
                "ts": _now(),
                "session_id": target["session_id"],
                "runtime": entry.get("runtime") or target.get("runtime"),
                "aspect": aspect,
                "value": value,
                "note": f"confidence={confidence:.2f}",
                "source": "model",
            }
        )
    return suggestions


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def info() -> list[dict]:
    """One entry per trained aspect: every artifact, newest marked active."""
    entries: list[dict] = []
    root = models_dir()
    if not root.is_dir():
        return entries
    for aspect_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        artifacts = _artifacts(aspect_dir.name)
        if not artifacts:
            entries.append({"aspect": aspect_dir.name, "dir": str(aspect_dir), "artifacts": []})
            continue
        entries.append(
            {
                "aspect": aspect_dir.name,
                "dir": str(aspect_dir),
                "active": artifacts[-1]["backend"],
                "artifacts": [
                    {
                        "backend": a["backend"],
                        "dir": str(a["dir"]),
                        "metrics": a["metrics"],
                    }
                    for a in artifacts
                ],
            }
        )
    return entries
