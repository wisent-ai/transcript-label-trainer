<!-- Moved out of README.md; the README links here. -->
## Fine-tuning a HuggingFace model

By default `train` fits TF-IDF + logistic regression. With `--model` it
fine-tunes a HuggingFace sequence-classification model instead. This needs
the optional `hf` feature (candle-core, candle-nn, tokenizers, hf-hub);
without it, `train --model` fails with a message telling you to build it in:

```sh
cargo build --release --features hf
```

Use `cargo install --path . --features hf` instead to replace the installed
binary. Fine-tuning supports the `distilbert` and `bert` architectures, which
covers `distilbert-base-multilingual-cased` and `bert-base-multilingual-cased`;
any other `model_type` fails with a sentence naming those two rather than
pretending to train.

Transcripts are mixed Polish and English, so prefer a multilingual base model.
Every run states its backend's settings; none has a default. A job states
them in its `training` section and `train` takes the same keys as flags spelled
with dashes, read as YAML scalars and validated by the same code. A missing one
is refused as `training.<key> (--<flag>) is required: <model> assumes no value
for it`, an invalid one as `training.<key> (--<flag>) must be <what>, got
<value>`, and a flag of the other backend as `--<flag> does not apply to model
'<model>'`. Every value is recorded under `hyperparameters` in `metrics.json`.

A fine-tune (`--model`) states `--epochs`, `--batch-size`, `--learning-rate`,
`--max-length`, `--seed` (head initialisation, in-training slice and shuffle),
`--weight-decay` (AdamW, zero or more), `--max-grad-norm` (global gradient
clip) and `--in-training-eval-share` (the share of the training side sliced off
to watch the loss). Prediction reads the sequence length, batch size and seed
back from the artifact. A base model whose `config.json` states no
`initializer_range` or classifier dropout is refused rather than given one.

```sh
transcript-label-trainer train --aspect topic \
  --model distilbert-base-multilingual-cased \
  --epochs E --batch-size B --learning-rate LR --max-length TOKENS \
  --seed S --weight-decay W --max-grad-norm G --in-training-eval-share F \
  --eval-split-fraction F --eval-split-seed N
```

TF-IDF + logistic regression (no `--model`) states `--ngram-max` (longest word
n-gram), `--lowercase`, `--sublinear-tf` and `--smooth-idf` (each `true` or
`false`), `--min-df` (terms in fewer documents are dropped), `--max-df` (terms
in more than this share of documents are dropped), `--c` (inverse L2 strength),
`--max-iter`, `--tol` (the solver stops once no gradient component exceeds it),
`--lbfgs-memory` (correction pairs kept), `--armijo-c1` (share of the predicted
decrease a line-search step must achieve) and `--max-backtracks` (line-search
candidates tried before a step is abandoned).

```sh
transcript-label-trainer train --aspect topic \
  --ngram-max N --lowercase B --sublinear-tf B --smooth-idf B \
  --min-df F --max-df F --c F --max-iter N --tol F --lbfgs-memory N \
  --armijo-c1 F --max-backtracks N \
  --eval-split-fraction F --eval-split-seed N
```

Or state them all at once with `--training scikit-learn` (refused beside any
single setting flag, and for a HuggingFace model). The preset is the
documented defaults of the libraries this backend reproduces:

| setting | value | source |
|---|---|---|
| `ngram_max`, `lowercase`, `sublinear_tf`, `smooth_idf`, `min_df`, `max_df` | 1, true, false, true, 1, 1.0 | [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) `ngram_range=(1, 1)`, `lowercase=True`, `sublinear_tf=False`, `smooth_idf=True`, `min_df=1`, `max_df=1.0` |
| `c`, `max_iter`, `tol` | 1.0, 100, 1e-4 | [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) `C=1.0`, `max_iter=100`, `tol=1e-4` |
| `max_backtracks` | 50 | the `"maxls": 50` LogisticRegression hands L-BFGS-B ([source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py)) |
| `lbfgs_memory` | 10 | [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) `maxcor=10` |
| `armijo_c1` | 1e-4 | SciPy `scalar_search_armijo` `c1=1e-4` ([source](https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py)) |

The line search picks each next step by minimising the quadratic through the
current loss, its slope and the rejected step's loss, as SciPy's Armijo search
does, so no shrink factor is set. Cross-validated accuracy uses stratified
folds in each class's own order, without shuffling, as scikit-learn's
`StratifiedKFold` does by default, so no seed is set and the same labels always
give the same folds. The Train panel's **Fill with scikit-learn's documented
defaults** button types the same values into its fields.

The data path is identical: labels from the lake label store, session text via
Transcript Lake, and the same session floor and frozen evaluation split all
apply, and the HF path additionally requires at least two sessions per class so
its in-training split keeps every class on both sides; that split is a
stratified slice of the *training* side and provides `in_training_eval` in the
metrics. It is not the frozen evaluation split described below, which no
backend ever trains on and which both backends score under
`holdout_evaluation`.

Artifacts land in `<training root>/models/<aspect>/hf-<sanitized-model-id>/` —
`model.safetensors` (the fine-tuned encoder plus classification head),
`config.json` (the base model's config carrying `num_labels`, `id2label`, and
`label2id` for the classes this aspect learned) and `tokenizer.json`, plus a
`metrics.json` with the same fields as the sklearn metrics (aspect, counts,
classes, n_sessions, …) plus the hyperparameters, base model, device, and both
evaluations. Training uses Apple-silicon Metal automatically and CPU
everywhere else, the way the Python backend used MPS: `Cargo.toml` turns
candle's `metal` feature on for macOS only. `metrics.json` records which one
ran under `device`, as `metal` or `cpu`; the Python build wrote `mps` there.

Every artifact's `metrics.json` records `session_text: "whole"`: the
classifier learned from whole session texts, and `infer` and `evaluate` give
it the same. An artifact without that field was trained on texts cut to a
fixed length; `infer` and `evaluate` refuse it with `artifact … was trained on
session text cut to a fixed length; … retrain it with 'transcript-label-trainer
train --aspect …' or 'transcript-label-trainer run <job.yaml>'`.

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
eval_split:                        # required: fraction and seed, or false
  fraction: F                      # share (between 0 and 1) frozen out of training
  seed: N                          # makes the first run's pick reproducible
judge:                             # optional; ON by default, shown with its default
  model: best                      # the Brama alias `evaluate` asks
training: scikit-learn             # required: a preset, or every tfidf-logreg setting:
# training:
#   ngram_max: N
#   lowercase: B
#   sublinear_tf: B
#   smooth_idf: B
#   min_df: F
#   max_df: F
#   c: F
#   max_iter: N
#   tol: F
#   lbfgs_memory: N
#   armijo_c1: F
#   max_backtracks: N
```

Every field is validated with a clear error — there are no silent defaults.
A HuggingFace `model` states `epochs`, `batch_size`, `learning_rate`,
`max_length`, `seed`, `weight_decay`, `max_grad_norm` and
`in_training_eval_share` in its `training` section instead.
Note that `evaluator: manual` matches only `manual` exactly, not `human` or
`brama:…`; to train on a teacher's labels, name it, e.g.
`evaluator: brama:best`. Model-sourced labels are never ground
truth unless you explicitly say so, because self-training on the model's own
predictions is a confirmation loop.

```sh
transcript-label-trainer run jobs/example-topic.yaml
```

`run` prints a resolved summary (name, task, evaluator, model, scope, and the
sessions found per class), then the resolved evaluation split (how many
sessions train, how many are held out, and whether the frozen file was reused),
and only then trains. Artifacts land in `<training root>/models/<name>/` with a
copy of the spec (`job.yaml`), and `metrics.json` carries the job metadata.
`train` and `infer` are unchanged; `run` is a layer over the same code path.

## The frozen evaluation split, and a Brama judge on top of it

Comparing two models over time only means something when both were scored on
the same untouched sessions. So every job and every `train` states a holdout —
`eval_split` with a `fraction` and `seed` (flags `--eval-split-fraction` and
`--eval-split-seed` for `train`), or `eval_split: false` / `--no-eval-split` to
train on everything; the trainer chooses neither number — and the chosen
session ids are written once to `<training root>/models/<name>/eval-split.json`:

```json
{
  "fraction": F,
  "seed": N,
  "created_at": "2026-08-08T22:14:07Z",
  "session_ids": ["019f3a44-…", "session_95aaaf37-…"]
}
```

What "frozen" buys you, and what it costs:

- **Written once, reused forever.** Every later run of the same job reads that
  file back and never rewrites it. Sessions labeled after the first run can
  only join the *training* side — nothing is ever promoted into the holdout —
  and a session in the holdout is never trained on, by either backend.
- **Reproducible from the seed.** The first run picks the holdout stratified
  per class, shuffled by `seed` (and the class name, so labeling one class more
  does not reshuffle the others). No class is ever emptied into the holdout,
  and any class with two or more sessions contributes at least one.
- **It costs training data.** Training needs two distinct values on the
  *training* side, so the holdout comes on top of that; no session count is
  set. Too few and the run fails with the exact numbers and says how to
  disable the split.
- **If the spec's fraction or seed later disagrees with the file, the file
  wins** and the run says so on stderr. That is what frozen means; delete the
  file by hand if you truly want a different holdout, and accept that the
  comparison with older runs is gone.

Both backends report it in `metrics.json` under `holdout_evaluation` —
accuracy, per-class counts, correct-per-class, and the confusion pairs — kept
deliberately separate from the HF backend's `in_training_eval`, which is a
stratified slice of the training side and is resplit on every run.

Accuracy against a stored label says how often the model agreed with whoever
labeled the session. It does not say whether the label the model chose was
defensible. `evaluate` asks that second question of a Brama teacher:

```sh
transcript-label-trainer evaluate topic-v1 --best
```

```
topic-v1 (aspect: topic, backend: sklearn):
    frozen split:  5 session(s), fraction=F, seed=N, created 2026-08-08T22:14:07Z
    split file:    /…/models/topic-v1/eval-split.json
    holdout:       accuracy=0.6 on 5 session(s)
        agent: 3/3 correct
        data: 0/1 correct
        confused data -> agent (1x)
    judge:         <model> calls 4/5 prediction(s) acceptable (agreement_rate=0.8, failed=0)
```

Per holdout session the judge gets the reconstructed session text (the same
lake CLI path and the same whole session training uses), the model's prediction and
the ground-truth label, and answers `acceptable` or `unacceptable` — so a
prediction that differs from the label can still be ruled defensible, and a
prediction that matches it can still be rejected. The verdict, the aggregate
agreement rate and one record per session go to
`<training root>/models/<name>/judge.json`.

`--best` adds a second, independent pass through Brama's `best` subscription
route. For every holdout record it audits both the stored ground-truth label
and the first judge's `acceptable`/`unacceptable` opinion against the
transcript. The four possible outcomes (`both-sensible`, `label-nonsensical`,
`judge-nonsensical`, `both-nonsensical`) and their aggregate counts are stored
under `best_review` in the same `judge.json`; any nonsensical or unreviewed
record makes the command exit nonzero.

Rules, mirroring `autolabel`:

- **Failure isolation.** A Brama error or an unparseable answer fails that one
  session, is counted in `failed`, and is recorded verbatim in `judge.json`.
- **No invented verdict.** If not one session could be judged — no usable
  provider route, no credential — `evaluate` prints the gateway's own error
  verbatim, writes nothing, and exits nonzero. There is no local heuristic
  fallback, because a fabricated verdict is worse than no verdict.
- The judge model comes from the job spec's `judge.model`, or `--brama-model`,
  defaulting to the same teacher `autolabel` uses. `judge: false` in the spec,
  or `--no-judge`, reports the holdout scores alone.
- Auth is `brama.rs`'s single HMAC/Skarbiec path — the same one `autolabel`
  uses. There is no second credential route.

`train` takes the same split as flags: `--eval-split-fraction` and
`--eval-split-seed` are required unless `--no-eval-split` is given.
`evaluate <aspect>` then scores it the same way.

