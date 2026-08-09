"""The frozen evaluation split, and the Brama teacher's verdict on it.

Two things live here, and they are two halves of one question — *does the
trained model work?*

**The frozen split.** Comparing two models over time only means something when
both were scored on the same untouched sessions, so the holdout is decided
once and written to ``$TLT_HOME/models/<name>/eval-split.json`` next to the
artifacts. Every later run of the same job reads that file back: sessions
labeled since then can only join the training side, a session already in the
holdout is never trained on, and the file is never rewritten. It is on by
default (``eval_split: false`` in the job spec turns it off).

**The judge.** Accuracy on the holdout says how often the model matched the
ground-truth label; it does not say whether the label it chose was defensible
for that transcript. So ``evaluate`` additionally sends each holdout session —
its text, the model's prediction, the ground truth — to a Brama-routed teacher
and asks for one word. A Brama error fails that one session and is counted,
exactly like ``autolabel``; if no session could be judged at all, the
gateway's own error is surfaced verbatim and nothing is written, because a
fabricated verdict is worse than no verdict.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from . import brama, jobs, lake, model

SPLIT_FILE = "eval-split.json"
JUDGE_FILE = "judge.json"

# The judge answers with exactly one of these.
JUDGE_VALUES = ["acceptable", "unacceptable"]


class SplitError(Exception):
    """Raised when a frozen split file is missing, unusable, or disabled."""


def split_path(name: str) -> Path:
    """Where the frozen holdout of one job (or aspect) is persisted."""
    return model.aspect_dir(name) / SPLIT_FILE


def judge_path(name: str) -> Path:
    return model.aspect_dir(name) / JUDGE_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _choose_holdout(
    session_ids: list[str], values: list[str], fraction: float, seed: int
) -> list[str]:
    """Pick the holdout, stratified per class and reproducible from the seed.

    A class never loses all of its sessions to the holdout, and any class with
    at least two sessions contributes at least one — otherwise the holdout
    could not report per-class counts for it. Shuffling is keyed by seed *and*
    class, so a class that gains labels does not reshuffle the others.
    """
    by_value: dict[str, list[str]] = {}
    for session_id, value in zip(session_ids, values):
        by_value.setdefault(value, []).append(session_id)
    chosen: list[str] = []
    for value in sorted(by_value):
        members = sorted(by_value[value])
        if len(members) < 2:
            continue
        count = min(max(1, int(len(members) * fraction)), len(members) - 1)
        shuffled = list(members)
        random.Random(f"{seed}:{value}").shuffle(shuffled)
        chosen.extend(shuffled[:count])
    return sorted(chosen)


def read_split(name: str) -> dict:
    """Read a frozen split file. Raises SplitError when it cannot be trusted."""
    path = split_path(name)
    if not path.is_file():
        raise SplitError(
            f"no frozen evaluation split at {path}; train '{name}' first, or "
            "the run that produced these artifacts had eval_split: false"
        )
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"frozen evaluation split {path} is unreadable: {exc}")
    session_ids = frozen.get("session_ids")
    if not isinstance(session_ids, list) or not all(
        isinstance(sid, str) and sid for sid in session_ids
    ):
        raise SplitError(
            f"frozen evaluation split {path} has no usable 'session_ids' list; "
            "it is never rewritten automatically — fix or remove it by hand"
        )
    return frozen


def resolve_split(
    name: str,
    session_ids: list[str],
    values: list[str],
    eval_split: dict,
    subject: str,
) -> dict:
    """Load or create the frozen holdout for this artifact directory.

    Returns the resolution: which rows of the selection train, which are held
    out, and the provenance of the file. The file is written only on the first
    run, and only after the training side is known to clear the minimums, so a
    run that fails cannot leave a split behind that a later run would inherit.
    """
    total = len(session_ids)
    if not eval_split["enabled"]:
        return {
            "enabled": False,
            "fraction": None,
            "seed": None,
            "created_at": None,
            "path": None,
            "frozen_sessions": 0,
            "train_index": list(range(total)),
            "holdout_index": [],
            "holdout_ids": [],
            "missing_from_selection": 0,
            "reused": False,
        }

    path = split_path(name)
    if path.is_file():
        frozen = read_split(name)
        holdout_ids = list(frozen["session_ids"])
        fraction = frozen.get("fraction")
        seed = frozen.get("seed")
        created_at = frozen.get("created_at")
        reused = True
        if fraction != eval_split["fraction"] or seed != eval_split["seed"]:
            lake.warn(
                f"eval_split in the spec (fraction={eval_split['fraction']}, "
                f"seed={eval_split['seed']}) differs from the frozen split in "
                f"{path} (fraction={fraction}, seed={seed}); the frozen file "
                "wins — that is what frozen means"
            )
    else:
        fraction = eval_split["fraction"]
        seed = eval_split["seed"]
        created_at = _now()
        holdout_ids = _choose_holdout(session_ids, values, fraction, seed)
        reused = False

    holdout = set(holdout_ids)
    holdout_index = [i for i, sid in enumerate(session_ids) if sid in holdout]
    train_index = [i for i in range(total) if i not in set(holdout_index)]

    train_values = sorted({values[i] for i in train_index})
    if len(train_index) < model.MIN_LABELED_SESSIONS or len(train_values) < 2:
        raise model.NotEnoughData(
            f"{subject} has {total} usable labeled session(s), of which "
            f"{len(holdout_index)} are held out by the frozen evaluation split "
            f"(fraction={fraction}, seed={seed}), leaving {len(train_index)} "
            f"session(s) across {len(train_values)} distinct value(s) to train "
            f"on; at least {model.MIN_LABELED_SESSIONS} sessions and 2 distinct "
            "values are required. Add labels with 'transcript-lake label add', "
            "or set 'eval_split: false' in the job spec to train on every "
            "labeled session."
        )

    if not reused:
        model.aspect_dir(name).mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fraction": fraction,
                    "seed": seed,
                    "created_at": created_at,
                    "session_ids": holdout_ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "enabled": True,
        "fraction": fraction,
        "seed": seed,
        "created_at": created_at,
        "path": str(path),
        "frozen_sessions": len(holdout_ids),
        "train_index": train_index,
        "holdout_index": holdout_index,
        "holdout_ids": [session_ids[i] for i in holdout_index],
        "missing_from_selection": len(holdout_ids) - len(holdout_index),
        "reused": reused,
    }


def split_json(split: dict) -> dict:
    """The part of a split resolution that belongs in metrics.json."""
    return {
        "enabled": split["enabled"],
        "fraction": split["fraction"],
        "seed": split["seed"],
        "created_at": split["created_at"],
        "path": split["path"],
        "frozen_sessions": split["frozen_sessions"],
        "holdout_sessions": len(split["holdout_index"]),
        "train_sessions": len(split["train_index"]),
        "missing_from_selection": split["missing_from_selection"],
        "reused": split["reused"],
    }


def holdout_report(gold: list[str], predictions: list[tuple[str, float]]) -> dict:
    """Accuracy, per-class counts and confusion pairs on the frozen holdout.

    This is not the HF backend's ``in_training_eval``: that one is a
    stratified slice of the training side, resplit on every run. This one is
    the frozen set, identical across runs and across backends.
    """
    counts: dict[str, int] = {}
    correct_per_class: dict[str, int] = {}
    confusion: dict[tuple[str, str], int] = {}
    correct = 0
    for actual, (predicted, _confidence) in zip(gold, predictions):
        counts[actual] = counts.get(actual, 0) + 1
        correct_per_class.setdefault(actual, 0)
        if predicted == actual:
            correct_per_class[actual] += 1
            correct += 1
        else:
            key = (actual, predicted)
            confusion[key] = confusion.get(key, 0) + 1
    total = len(gold)
    pairs = [
        {"gold": actual, "predicted": predicted, "n": n}
        for (actual, predicted), n in sorted(
            confusion.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "n_sessions": total,
        "accuracy": round(correct / total, 4) if total else None,
        "counts": dict(sorted(counts.items())),
        "correct": dict(sorted(correct_per_class.items())),
        "confusion": pairs,
    }


# ---------------------------------------------------------------------------
# The Brama teacher's verdict
# ---------------------------------------------------------------------------


def judge_prompt(aspect: str, task: str | None, gold: str, prediction: str, text: str) -> list[dict]:
    """Ask the teacher whether one prediction is defensible for one session."""
    purpose = f"\nWhat the classifier is for: {task}" if task else ""
    return [
        {
            "role": "system",
            "content": (
                "You audit a small classifier that assigns aspect labels to "
                "coding-agent session transcripts. Answer with exactly one "
                f"word, {JUDGE_VALUES[0]} or {JUDGE_VALUES[1]}, and nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Aspect: {aspect}{purpose}\n"
                f"Ground-truth label recorded by the evaluator: {gold}\n"
                f"Label predicted by the classifier: {prediction}\n\n"
                "Is the predicted label a defensible reading of this "
                f"transcript on this aspect? Answer {JUDGE_VALUES[0]} if it is "
                "(including when it differs from the ground truth but the "
                f"session genuinely supports it), {JUDGE_VALUES[1]} if it is "
                "not.\n\n"
                f"Transcript:\n{text}"
            ),
        },
    ]


def _judge_sessions(
    client: "brama.BramaClient",
    judge_model: str,
    aspect: str,
    task: str | None,
    sessions: list[dict],
    texts: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """One verdict per holdout session; a Brama error fails only its session."""
    records: list[dict] = []
    failures: list[dict] = []
    for session in sessions:
        session_id = session["session_id"]
        try:
            answer = client.chat(
                judge_model,
                judge_prompt(
                    aspect,
                    task,
                    session["gold"],
                    session["prediction"],
                    texts[session_id]["text"],
                ),
            )
            verdict, _exact = brama.parse_answer(answer, JUDGE_VALUES)
            if verdict is None:
                failures.append(
                    {
                        "session_id": session_id,
                        "error": f"unparseable judge answer: {answer[:80]!r}",
                    }
                )
                continue
            records.append(dict(session, verdict=verdict))
        except brama.BramaError as exc:
            failures.append({"session_id": session_id, "error": str(exc)})
    return records, failures


def evaluate(
    name: str, judge: bool | None = None, judge_model: str | None = None
) -> dict:
    """Score the trained model on its frozen holdout and have Brama judge it.

    ``name`` is a job name or a bare aspect — both are directory names under
    ``$TLT_HOME/models/``. Raises SplitError when there is no frozen holdout,
    FileNotFoundError when nothing is trained, and BramaError when the gateway
    could not judge a single session.
    """
    artifact = model.active_artifact(name)
    metrics = artifact["metrics"]
    aspect = metrics["aspect"]
    job_meta = metrics.get("job") or {}
    spec_judge = job_meta.get("judge") or jobs.default_judge()
    judge_enabled = spec_judge.get("enabled", True) if judge is None else judge
    model_id = judge_model or spec_judge.get("model") or brama.DEFAULT_MODEL

    frozen = read_split(name)
    labels = model.labels_for_artifact(metrics)
    holdout_ids = [sid for sid in frozen["session_ids"] if sid in labels]
    missing = len(frozen["session_ids"]) - len(holdout_ids)
    if not holdout_ids:
        raise SplitError(
            f"none of the {len(frozen['session_ids'])} frozen holdout session(s) "
            f"in {split_path(name)} still carry a ground-truth label for aspect "
            f"'{aspect}'; there is nothing to evaluate"
        )

    texts = lake.session_texts(holdout_ids)
    usable = [sid for sid in holdout_ids if texts.get(sid, {}).get("text", "").strip()]
    no_text = len(holdout_ids) - len(usable)
    if no_text:
        lake.warn(f"{no_text} frozen holdout session(s) had no text in the lake")
    if not usable:
        raise SplitError(
            f"none of the {len(holdout_ids)} frozen holdout session(s) have text "
            "in the lake; there is nothing to evaluate"
        )

    gold = [str(labels[sid]["value"]) for sid in usable]
    predictions = model.predict(artifact, [texts[sid]["text"] for sid in usable])
    report = holdout_report(gold, predictions)

    sessions = [
        {
            "session_id": session_id,
            "runtime": texts[session_id].get("runtime"),
            "gold": actual,
            "prediction": predicted,
            "confidence": round(confidence, 4),
            "correct": predicted == actual,
        }
        for session_id, actual, (predicted, confidence) in zip(usable, gold, predictions)
    ]

    result = {
        "name": name,
        "aspect": aspect,
        "backend": artifact["backend"],
        "model_path": metrics.get("model_path"),
        "trained_at": metrics.get("trained_at"),
        "evaluated_at": _now(),
        "eval_split": {
            "path": str(split_path(name)),
            "fraction": frozen.get("fraction"),
            "seed": frozen.get("seed"),
            "created_at": frozen.get("created_at"),
            "frozen_sessions": len(frozen["session_ids"]),
            "missing_ground_truth": missing,
            "skipped_no_text": no_text,
        },
        "holdout_evaluation": report,
    }

    if not judge_enabled:
        result["judge"] = {"enabled": False}
        result["sessions"] = sessions
        return result

    client = brama.BramaClient.from_env()
    records, failures = _judge_sessions(
        client, model_id, aspect, job_meta.get("task"), sessions, texts
    )
    if not records:
        # No usable provider route: surface the gateway's own words and write
        # nothing. A verdict nobody produced is not a verdict.
        raise brama.BramaError(
            f"the Brama judge ({model_id}) could not judge any of the "
            f"{len(sessions)} holdout session(s); first error: "
            f"{failures[0]['error'] if failures else '(no sessions to judge)'}"
        )

    acceptable = sum(1 for record in records if record["verdict"] == JUDGE_VALUES[0])
    result["judge"] = {
        "enabled": True,
        "model": model_id,
        "judged": len(records),
        "acceptable": acceptable,
        "unacceptable": len(records) - acceptable,
        "failed": len(failures),
        "agreement_rate": round(acceptable / len(records), 4),
    }
    result["sessions"] = records
    result["failures"] = failures

    path = judge_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["judge_path"] = str(path)
    return result
