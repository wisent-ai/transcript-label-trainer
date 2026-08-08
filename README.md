# Transcript Label Trainer

**Transcript Label Trainer trains small local classifiers that predict aspect
labels for coding-agent sessions in the Transcript Lake, and emits label
suggestions — it never writes to the lake.**

The lake's labeler owns an append-only store of aspect labels at
`~/.transcript-lake/labels/*.ndjson`, one record per aspect label on a session:

```json
{"ts": "...", "session_id": "...", "runtime": "...", "aspect": "topic", "value": "...", "note": "...", "source": "manual"}
```

Aspects are independent dimensions ("kąty"): topic, quality, reviewed,
task-type — many per session. This repository turns the manual labels into a
model that suggests the rest.

## Product boundary

Transcript Label Trainer owns:

- training one TF-IDF + logistic-regression classifier per aspect over the
  manual labels in the lake's label store;
- session-text reconstruction, by shelling out to the lake CLI's read-only
  `query` command (user + assistant text per session, ordered by `ts`, capped
  at 12 KB);
- emitting suggestion records shaped exactly like label-store records, with
  `source="model"` and the confidence in `note`;
- its own model artifacts, under `~/.transcript-label-trainer/models/`
  (override with `TLT_HOME`) — runtime state, outside this repository.

Transcript Label Trainer does not own:

- the lake, its ingest, its events, or its views — that is
  [`wisent-ai/transcript-lake`](https://github.com/wisent-ai/transcript-lake),
  consumed here read-only through its own CLI;
- the label store or the label vocabulary — the lake's labeler owns
  `labels/*.ndjson`; this tool reads it and never writes it;
- applying suggestions. Review the emitted records, then apply them through
  the lake's labeler: `transcript-lake label add <session-id> --aspect <name>
  --value <v>`;
- model serving, cloud training, or GPUs. Everything runs locally on CPU with
  scikit-learn.

## Quick start

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Train one aspect from the manual labels in the lake:

```sh
transcript-label-trainer train --aspect reviewed
```

With too few labeled sessions this fails cleanly, stating the minimum and the
actual count — that is correct behavior, not a crash. The minimum is 8 labeled
sessions across at least 2 distinct values.

Emit suggestions for sessions that have no label on that aspect yet:

```sh
transcript-label-trainer infer --aspect reviewed --limit 20
```

Each suggestion is a label-store-shaped record with `source="model"`:

```json
{
  "ts": "2026-08-08T05:00:00Z",
  "session_id": "abc123",
  "runtime": "claude",
  "aspect": "reviewed",
  "value": "yes",
  "note": "confidence=0.83",
  "source": "model"
}
```

Suggestions are printed to stdout and nothing is written to the lake. To apply
them, review and feed the accepted ones to the lake's labeler:

```sh
transcript-label-trainer infer --aspect reviewed --limit 20 > suggestions.json
# review, then apply each accepted record:
transcript-lake label add <session-id> --aspect reviewed --value <value> --note "confidence=0.83"
```

Note: `label add` currently records every applied label with `source="manual"`;
the labeler reserves `source="model"` but exposes no flag for it yet. Until it
does, the confidence in `--note` is what marks an applied suggestion.

Inspect trained aspects, artifact paths, and metrics:

```sh
transcript-label-trainer info
```

`python -m transcript_label_trainer ...` is equivalent to the console script.

## Environment

- `LAKE_DATA` — lake data root, default `~/.transcript-lake`. Resolved exactly
  like the lake CLI resolves it, and passed through to it.
- `TLT_LAKE_CLI` — override how the lake CLI is invoked. Default:
  `node ~/Documents/CodingProjects/Wisent/transcript-lake/src/cli.mjs`.
- `TLT_HOME` — trainer state root, default `~/.transcript-label-trainer`.
  Models live under `$TLT_HOME/models/<aspect>/` as `model.joblib` +
  `metrics.json`.

## Requirements

- Python 3.10+, scikit-learn (installed into the venv by the quick start).
- Node.js and DuckDB, because session text comes from the lake CLI's `query`
  command, which runs DuckDB over the lake's own views.
