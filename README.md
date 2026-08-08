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

- training one classifier per aspect over the manual labels in the lake's
  label store — TF-IDF + logistic regression by default, or a fine-tuned
  HuggingFace transformer when `--model` is given (optional `hf` extra);
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
  --value <v> --source model`;
- model serving or cloud training. Everything runs locally with scikit-learn,
  plus a local HuggingFace fine-tune on CPU or Apple-silicon MPS when the
  optional `hf` extra is installed.

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
transcript-lake label add <session-id> --aspect reviewed --value <value> --source model --note "confidence=0.83"
```

Inspect trained aspects, artifact paths, and metrics:

```sh
transcript-label-trainer info
```

`python -m transcript_label_trainer ...` is equivalent to the console script.

## Fine-tuning a HuggingFace model

By default `train` fits TF-IDF + logistic regression. With `--model` it
fine-tunes any HuggingFace sequence-classification model instead. This needs
the optional `hf` extra (torch + transformers); without it, `train --model`
fails with a message telling you to install it:

```sh
pip install -e '.[hf]'
```

Transcripts are mixed Polish and English, so prefer a multilingual base model:

```sh
transcript-label-trainer train --aspect topic \
  --model distilbert-base-multilingual-cased \
  --epochs 3 --batch-size 8 --lr 2e-5 --max-length 512
```

The data path is identical: labels from the lake label store, session text via
the lake CLI. The same minimum of 8 labeled sessions and 2 distinct values
applies, and the HF path additionally requires at least 2 sessions per class so
the stratified holdout split keeps every class on both sides; a stratified
holdout provides the `eval_accuracy` in the metrics.

Artifacts land in `$TLT_HOME/models/<aspect>/hf-<sanitized-model-id>/` — the
`save_pretrained` model and tokenizer plus a `metrics.json` with the same
fields as the sklearn metrics (aspect, counts, classes, n_sessions, …) plus the
hyperparameters, base model, device, and holdout evaluation. Training runs on
CPU by default and uses Apple-silicon MPS automatically when
`torch.backends.mps.is_available()`.

When both a sklearn and an HF artifact exist for an aspect, `infer` uses the
newest one by training time; `info` lists every backend per aspect and marks
the active one.

## Training jobs

For repeatable runs, declare the job in a YAML spec instead of flags. A job
answers four questions: **WHO** evaluated the transcripts (`evaluator` — the
exact label-store source that counts as ground truth; only labels with exactly
this source are used), **WHICH** model to train (`model` — `tfidf-logreg` for
the sklearn backend, any other string is a HuggingFace model id), the **SCOPE**
of training data (`scope`), and the **TASK** (`task` — free text stored with
the artifacts and shown by `info`).

```yaml
name: topic-v1
task: classify the primary topic of the session
evaluator: manual
model: tfidf-logreg
scope:
  aspect: topic
  runtimes: [claude, codex, kimi]  # optional; default is all runtimes
  since: "2026-07-01"              # optional; label ts must be on/after this
  values: [bugfix, feature, chore] # optional; restrict to these values
  min_text_chars: 200              # optional; skip shorter session texts
```

Every field is validated with a clear error — there are no silent defaults.
Note that `evaluator: manual` matches only `manual` exactly, not `human` or
`brama:…`; to train on a teacher's labels, name it, e.g.
`evaluator: brama:claude-opus-4.6`. Model-sourced labels are never ground
truth unless you explicitly say so, because self-training on the model's own
predictions is a confirmation loop.

```sh
transcript-label-trainer run jobs/example-topic.yaml
```

`run` prints a resolved summary (name, task, evaluator, model, scope, and the
sessions found per class) before training, then trains exactly like `train`
does. Artifacts land in `$TLT_HOME/models/<name>/` with a copy of the spec
(`job.yaml`), and `metrics.json` carries the job metadata. `train` and `infer`
are unchanged; `run` is a layer over the same code path.

## Environment

- `LAKE_DATA` — lake data root, default `~/.transcript-lake`. Resolved exactly
  like the lake CLI resolves it, and passed through to it.
- `TLT_LAKE_CLI` — override how the lake CLI is invoked. Default:
  `node ~/Documents/CodingProjects/Wisent/transcript-lake/src/cli.mjs`.
- `TLT_HOME` — trainer state root, default `~/.transcript-label-trainer`.
  Models live under `$TLT_HOME/models/<aspect>/`: sklearn as `model.joblib` +
  `metrics.json`, HF fine-tunes in `hf-<model-id>/` subdirectories.

## Requirements

- Python 3.10+, scikit-learn and PyYAML (installed into the venv by the quick
  start).
- Optionally torch + transformers via the `hf` extra, for `--model`
  fine-tuning. CPU is the baseline; MPS is used automatically on Apple
  silicon.
- Node.js and DuckDB, because session text comes from the lake CLI's `query`
  command, which runs DuckDB over the lake's own views.
