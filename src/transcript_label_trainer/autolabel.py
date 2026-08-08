"""Automatic labeling by a Brama teacher — operator-mandated zero-touch.

For each lake session with no label on the aspect yet, a Brama-routed model
classifies the reconstructed session text into one allowed value, and the
result is applied immediately through the lake CLI
(``label add --source brama:<model>``). The lake validates sessions and owns
the write; what there is not, by design, is a human review step.

Human labels are never overwritten: a session already labeled with the aspect
by ANY source is skipped, so reruns are idempotent.
"""

from __future__ import annotations

import os
import subprocess

from . import brama, lake

AUTOLABEL_NOTE = "autolabel"


class LabelApplyError(Exception):
    """Raised when the lake CLI refuses a label write."""


def _label_add(session_id: str, aspect: str, value: str, source: str) -> None:
    env = dict(os.environ, LAKE_DATA=str(lake.lake_data()))
    done = subprocess.run(
        lake.lake_cli()
        + [
            "label", "add", session_id,
            "--aspect", aspect,
            "--value", value,
            "--source", source,
            "--note", AUTOLABEL_NOTE,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        raise LabelApplyError(f"lake label add failed: {detail[:200]}")


def autolabel(
    aspect: str,
    values: list[str],
    brama_model: str | None = None,
    limit: int | None = None,
    runtime: str | None = None,
) -> dict:
    """Label every unlabeled session for an aspect via the Brama teacher.

    Returns the summary dict (labeled / skipped_labeled / failed counts with
    per-session results and failures).
    """
    model_id = brama_model or brama.DEFAULT_MODEL
    source = f"brama:{model_id}"
    client = brama.BramaClient.from_env()

    already = set(lake.load_labels(aspect))  # latest per session, any source
    sessions = lake.all_sessions()
    if runtime:
        sessions = [s for s in sessions if s.get("runtime") == runtime]
    skipped_labeled = sum(1 for s in sessions if s["session_id"] in already)
    targets = [s for s in sessions if s["session_id"] not in already]
    if limit is not None:
        targets = targets[:limit]

    texts = lake.session_texts([t["session_id"] for t in targets])
    results: list[dict] = []
    failures: list[dict] = []
    no_text = 0
    for target in targets:
        sid = target["session_id"]
        entry = texts.get(sid)
        if not entry or not entry["text"].strip():
            no_text += 1
            continue
        try:
            answer = client.chat(model_id, brama.build_prompt(aspect, values, entry["text"]))
            value, _exact = brama.parse_answer(answer, values)
            if value is None:
                failures.append(
                    {"session_id": sid, "error": f"unparseable answer: {answer[:80]!r}"}
                )
                continue
            _label_add(sid, aspect, value, source)
            results.append({"session_id": sid, "value": value})
        except (brama.BramaError, LabelApplyError) as exc:
            failures.append({"session_id": sid, "error": str(exc)[:200]})

    return {
        "aspect": aspect,
        "brama_model": model_id,
        "source": source,
        "labeled": len(results),
        "skipped_labeled": skipped_labeled,
        "skipped_no_text": no_text,
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
