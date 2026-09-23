<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="transcript-label-trainer by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/transcript-label-trainer) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/transcript-label-trainer/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# Transcript Label Trainer

Small Models for Your Custom Harness Needs.

Your harness makes dozens of repetitive calls. What goal am I working towards? Is
the user frustrated?

Don’t use your smart model for this. Save tokens and train a smaller, faster,
task-specific model using this repository to analyse your sessions faster and
cheaper.

Intelligence to analyse your AI <> Human interactions at minimal cost.

The lake's labeler owns an append-only store of aspect labels at
`~/.transcript-lake/labels/*.ndjson`, one record per aspect label on a session:

```json
{"ts": "...", "session_id": "...", "runtime": "...", "aspect": "topic", "value": "...", "note": "...", "source": "manual"}
```

Aspects are independent dimensions ("kąty"): topic, quality, reviewed,
task-type — many per session. This repository turns the manual labels into a
model that suggests the rest.

Navigable repository-local operator documentation begins at
[`docs/index.html`](docs/index.html), including the complete
[`docs/corpus-import.html`](docs/corpus-import.html) CLI/GUI contract. The
installed binary serves those same pages from `transcript-label-trainer gui`;
durable platform documentation is also published at
[`wisent.com/docs/ground-truth`](https://wisent.com/docs/ground-truth).

## Product boundary

Transcript Label Trainer owns:

- adopting an existing canonical schema-v1 dataset bundle into an immutable,
  content-addressed corpus store under the training root, then training one
  classifier per aspect over those labels — TF-IDF + logistic regression by
  default, or a fine-tuned HuggingFace transformer when `--model` is given
  (optional `hf` feature);
- session-text reconstruction, by shelling out to the lake CLI's read-only
  `query` command (user + assistant text per session, ordered by `ts`, capped
  at 12 KB);
- emitting suggestion records shaped exactly like label-store records, with
  `source="model"` and the confidence in `note`;
- its own model artifacts, under the training root Stado places this trainer
  on — runtime state, outside this repository (see *Placement*), including the
  frozen evaluation split (`eval-split.json`) and the teacher's verdict
  (`judge.json`);

Transcript Label Trainer does not own:

- the lake, its ingest, its events, or its views — that is
  [`wisent-ai/transcript-lake`](https://github.com/wisent-ai/transcript-lake),
  consumed here read-only through its own CLI;
- the label store or the label vocabulary — the lake's labeler owns
  `labels/*.ndjson`; this tool reads it and never writes it;
- applying `infer` suggestions. Review the emitted records, then apply them
  through the lake's labeler: `transcript-lake label add <session-id> --aspect
  <name> --value <v> --source model`. (`autolabel` is the deliberate
  exception: it writes through the lake CLI without a human staging queue and
  can require the independent Brama `best` gate.);
- model serving, the compute-target registry, or remote job lifecycle. Stado
  owns placement, source checkout, scoped secrets, execution, logs, and the
  terminal outcome; this trainer only prepares and submits the declared work.

## Quick start

Installing the binary and the first runs: [docs/guide/quick-start.md](docs/guide/quick-start.md).

## CLI

The global flags and every subcommand: [docs/guide/cli.md](docs/guide/cli.md).

## Pipeline, step by step

What actually happens, in order, when this repository is used end to end.
Every step names its owner, because half of them are deliberately not this
repository's.

1. **Transcripts land in the lake.** Vendor runtimes write raw transcripts;
   `transcript-lake` ingests and privacy-masks them into its canonical store.
   This repository reads that store read-only, through the lake CLI, and
   never writes it.
2. **Curation.** The trainer exports candidate rows for the task at hand —
   session texts for aspect classifiers, masked user messages for goal
   titles, decision envelopes for the lifecycle contract — cleaned of
   machine noise before any model sees them.
3. **Labeling.** Ground truth comes from a named evaluator: manual labels in
   the lake's label store, or a Brama-routed teacher (`autolabel`,
   `lifecycle-review`, the teacher stage of `goal-model`). Every record
   carries provenance; model-sourced labels are never ground truth unless
   the job names them, because self-training on the model's own predictions
   is a confirmation loop.
4. **Independent review.** A second, independent Brama route (`best` by
   default) audits the labels before anything trains on them. A failed or
   unparseable review fails that row, never invents a verdict.
5. **Frozen splits.** Train/eval membership is written once and reused
   forever (`eval-split.json`, `--split train|eval`). Nothing is ever
   promoted into a holdout, and no backend trains on one.
6. **Training.** Aspect classifiers train locally in seconds. Fine-tunes
   (`goal-model`, `lifecycle-model`, `humanizer-model`) are submitted
   through Stado to one named exclusive GPU target from the canonical
   registry — Stado owns checkout, scoped secrets, execution, logs, and the
   terminal outcome.
7. **Qualification.** The student's held-out predictions face an independent
   judge (`evaluate`, `goal-audit`, `lifecycle-audit`) with an explicit
   quality gate. The gate failing means no artifact ships; there is no
   override.
8. **Publication.** Only qualified artifacts are published, with their
   manifests, metrics, and judge verdicts, through Stado storage.
9. **Serving and use.** Consumers own the rest: Oko installs and serves the
   lifecycle model on loopback, Jeden and jeden-desktop call it and record
   its decisions in their session ledgers. This repository trains models;
   it never serves them.

## Training, evaluation and models

- HuggingFace fine-tuning, training jobs and the frozen evaluation split with its judge: [docs/guide/training.md](docs/guide/training.md).
- Automatic labeling and the reviewed Jeden goal and Oko lifecycle models: [docs/guide/models.md](docs/guide/models.md).
- Where training runs, decided by Stado: [docs/guide/placement.md](docs/guide/placement.md).

## Environment

- `TLT_HOME` — trainer state root. Overrides the Stado training declaration;
  models live under `$TLT_HOME/models/<aspect>/` (tfidf-logreg as `model.json` +
  `metrics.json`, HF fine-tunes in `hf-<model-id>/` subdirectories, plus the
  job's `eval-split.json` and, once `evaluate` has run, `judge.json`).
- `LAKE_DATA` — lake data root. Overrides the Stado storage declaration, and
  is passed through to the lake CLI.
- `TLT_LAKE_CLI` — override how the lake CLI is invoked, split on whitespace
  into a command and its arguments. Default: `transcript-lake` on `PATH`,
  falling back to
  `~/Documents/CodingProjects/Wisent/transcript-lake/target/release/transcript-lake`
  when the name is not found there.
- `TLT_DATASET_BUNDLE` — internal read-only dataset bundle used by a pinned
  Stado job instead of reaching back into the source machine's lake.
- `TLT_REPO_REF` — exact lowercase commit used for Stado's source checkout;
  normally resolved from this checkout automatically.

## Requirements

- A Rust toolchain at version `1.85` or newer, to build the binary. The
  tfidf-logreg backend needs nothing else at run time.
- The `hf` cargo feature, for `--model` fine-tuning.
- The lake CLI and DuckDB, because session text comes from the lake CLI's
  `query` command, which runs DuckDB over the lake's own views.

## Unreleased changes

- `run JOB --compute-target TARGET` now exports a minimal read-only dataset,
  submits the exact trainer commit through Stado, pins execution to the named
  compute target, follows the job, and runs the semantic evaluation there.
- `autolabel --best` audits proposed labels before writing them;
  `evaluate --best` independently audits both stored labels and the first
  judge's opinions through Brama's `best` route. Both are quality gates with
  machine-readable records and nonzero status for nonsensical results.

- Transcript Label Trainer is now implemented in Rust and ships as one binary.
  Existing command behavior remains compatible; the Stado and `--best`
  surfaces above are additive. Existing label records, `metrics.json`
  (including the `backend` value, still literally `sklearn` for the
  tfidf-logreg artifact), `eval-split.json`, `job.yaml`, and the job spec YAML
  keep their shapes; `--best` adds its audit records only to command output and
  `judge.json`. One file changed name during the Rust migration: `model.json`
  replaces `model.joblib`, because the fitted vectorizer and classifier are now
  stored as JSON anything can read.
- **Retrain any model the Python build produced.** A `model.joblib` is a
  pickle this binary cannot read. `info` still lists such an artifact with all
  its metrics; only inference refuses, saying the artifact "holds
  model.joblib, a pickle written by the Python build that this binary cannot
  read" and naming the `train`/`run` command that produces `model.json`. The
  labels it was trained on are untouched in the lake, so retraining is the
  whole migration.
- Installation changed: `cargo install --path .` replaces the virtualenv and
  `pip install -e .`. The HuggingFace fine-tune backend is the `hf` cargo
  feature (`cargo install --path . --features hf`), built on candle and
  tokenizers rather than torch and transformers.
- Python, pip, a virtualenv, scikit-learn and PyYAML are no longer
  prerequisites, and neither is Node: the lake CLI this tool shells out to is a
  Rust binary now, so `TLT_LAKE_CLI` defaults to `transcript-lake` on `PATH`.
  DuckDB is still required, because session text still comes from the lake
  CLI's `query` command.