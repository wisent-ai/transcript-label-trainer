"""Read-only access to Transcript Lake state.

Two contracts are consumed here, neither is reimplemented:

- the append-only label store at ``<storage root>/labels/*.ndjson``, owned by
  ``transcript-lake label`` — this module only ever reads it;
- the canonical ``events``/``sessions`` DuckDB views, reached by shelling out
  to the lake CLI (``query --json``) so the SQL setup in sql/views.sql stays
  the lake's own code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .placement import resolve_placement

TEXT_CAP = 12_000  # characters of concatenated session text kept per session

DEFAULT_LAKE_CLI = (
    Path.home()
    / "Documents"
    / "CodingProjects"
    / "Wisent"
    / "transcript-lake"
    / "src"
    / "cli.mjs"
)


def lake_cli() -> list[str]:
    """Command prefix that invokes the lake CLI."""
    override = os.environ.get("TLT_LAKE_CLI")
    if override:
        return override.split()
    return ["node", str(DEFAULT_LAKE_CLI)]


def load_labels(aspect: str) -> dict[str, dict]:
    """Latest label record per session for one aspect.

    The store is append-only, so the record with the newest ``ts`` wins for
    each ``session_id``. A missing labels directory simply means zero labels.
    """
    labels_dir = resolve_placement().storage_root / "labels"
    latest: dict[str, dict] = {}
    if not labels_dir.is_dir():
        return latest
    for path in sorted(labels_dir.glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("aspect") != aspect or not record.get("session_id"):
                continue
            current = latest.get(record["session_id"])
            if current is None or str(record.get("ts", "")) >= str(current.get("ts", "")):
                latest[record["session_id"]] = record
    return latest


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def query(sql: str) -> list[dict]:
    """Run SQL over the lake views and return rows as dicts."""
    env = dict(os.environ)
    env["LAKE_DATA"] = str(resolve_placement().storage_root)
    proc = subprocess.run(
        lake_cli() + ["query", "--json", sql],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"lake query failed: {detail}")
    out = proc.stdout.strip()
    if not out:
        return []
    return json.loads(out)


def all_sessions() -> list[dict]:
    """Every session known to the lake: {runtime, session_id}."""
    return query("SELECT runtime, session_id FROM sessions ORDER BY last_ts DESC")


def session_texts(session_ids: list[str]) -> dict[str, dict]:
    """Concatenated user+assistant text per session, ordered by ts.

    Returns {session_id: {"runtime": str, "text": str}} for sessions that have
    at least one text event; capped at TEXT_CAP characters per session.
    """
    if not session_ids:
        return {}
    wanted = ", ".join(_quote(s) for s in session_ids)
    rows = query(
        "SELECT session_id, runtime, ts, text FROM events "
        "WHERE event_type IN ('user', 'assistant') AND text IS NOT NULL "
        f"AND session_id IN ({wanted}) "
        "ORDER BY ts"
    )
    texts: dict[str, dict] = {}
    for row in rows:
        sid = row["session_id"]
        entry = texts.setdefault(sid, {"runtime": row.get("runtime"), "parts": []})
        entry["parts"].append(str(row["text"]))
    result = {}
    for sid, entry in texts.items():
        result[sid] = {"runtime": entry["runtime"], "text": "\n".join(entry["parts"])[:TEXT_CAP]}
    return result


def warn(message: str) -> None:
    sys.stderr.write(f"transcript-label-trainer: {message}\n")
