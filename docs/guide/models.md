<!-- Moved out of README.md; the README links here. -->
## Automatic labeling with a Brama teacher

`autolabel` labels sessions at scale with a model routed through Brama —
Wisent's authenticated, provider-neutral OpenAI-compatible gateway (all LLM
inference goes through Brama; never direct provider keys). With `--best`, each
proposed label is independently audited by Brama's `best` route before it can
reach Transcript Lake; there is still no human staging queue.

```sh
transcript-label-trainer autolabel --aspect tasktype \
  --values bugfix,feature,chore,question --limit 50 --best
```

For each session that has no label on the aspect yet, autolabel reconstructs
the session text and asks the teacher for exactly one of the allowed values.
With `--best`, a second model returns `sensible` or `nonsensical`; only a
`sensible` proposal is applied through the lake's own CLI:

```sh
transcript-lake label add <session-id> --aspect tasktype --value <v> --source brama:<model-id> --note "autolabel; reviewed=best"
```

The lake CLI validates the session and owns the write — that boundary stays;
what changed is only that no human reviews the suggestion. Rules:

- **No overwrite.** A session already labeled with the aspect by ANY source
  is skipped — human labels are sacred, and reruns are idempotent.
- **Semantic gate.** `--best` never applies a proposal rejected as
  `nonsensical`, records it under `rejected`, and exits nonzero if a proposal
  is rejected or the final reviewer cannot answer.
- **Failure isolation.** A Brama error or an unparseable answer fails that
  one session, writes nothing for it, and is counted in the final summary
  (`labeled` / `skipped_labeled` / `failed`).
- The teacher defaults to `codex/gpt-5.6-sol` — one of the few model ids this
  fleet's Brama can actually serve, and multilingual, which the mixed
  Polish/English transcripts need; override with `--brama-model`.
- Auth mirrors jeden: HMAC-signed requests keyed by the Skarbiec item
  `agent:wisent-app`, bearer from `jeden-model-router`, endpoint from
  `BRAMA_URL` (falling back to jeden's own configured URL). Secrets are read
  into memory only, never printed.

The end-to-end story: autolabel an aspect, then train on the teacher's
labels by naming the provenance in a job spec:

```yaml
name: tasktype-v1
task: classify what kind of work the session did
evaluator: brama:codex/gpt-5.6-sol
model: tfidf-logreg
scope:
  aspect: tasktype
```

## Reviewed Jeden goal model

`goal-model` owns the complete small-model pipeline that turns coding-agent
messages into the 3–7 word task goals Jeden displays. It reads messages only
from Transcript Lake's normalized `events` view; raw agent session files are
not an input, so the lake's masking boundary remains intact.

```sh
transcript-label-trainer goal-model \
  --compute-target ubuntu-server-rtx-pro-6000 \
  --limit 1500
```

The command generates task and no-task labels with a Brama teacher, then requires
two review passes through the pinned Brama reviewer before any row enters the
dataset. It holds out reviewed OMP titles plus 32 teacher task rows and 32
teacher no-task rows, submits the remaining JSONL and exact trainer commit to
the named Stado target, and performs an exclusive full fine-tune from pinned
`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`. Held-out rows never
enter training. Every student prediction over that holdout must receive
`both-sensible` from a final Brama `--best` audit or the model remains
unqualified; a rejected candidate is retained for diagnosis but cannot enter
the Jeden Desktop release namespace.

A qualified job publishes the Q4_K_M GGUF, metrics, held-out predictions, the
full final audit, canonical prompt, dependency lock, and checksums under the
content-addressed URI printed as `model artifact:
stado://probierz/artifacts/models/jeden/goal-qwen3-4b/<dataset-sha256>`.
Stado also retains its canonical `status/<job-id>/output/` copy. All model calls
use `brama.rs`; the pipeline has no direct provider credentials or second auth
implementation.

The qualified 4B release is also public at
[`lbartoszcze/jeden-goal-qwen3-4b`](https://huggingface.co/lbartoszcze/jeden-goal-qwen3-4b),
revision `d9ce79f106ead1176b74bb0d9fb875521ca712b1`. Its 2,497,280,320-byte
GGUF has SHA-256
`2512d7a455a50a16742b75d8fe38bf02b46b5d6b607f785be32a6345d999d310`,
the same immutable artifact consumed by Jeden Desktop.

## Reviewed Oko goal-lifecycle model

Oko uses a separate contextual model for lifecycle decisions. Its
`oko-goal-lifecycle-v1` contract classifies each prompt as `startGoal`,
`continueCurrent`, `finishGoal`, or `ignore`, selects an existing goal by
reference when required, and records only explicit open/completion evidence.
The existing title model remains the sole source of titles for newly started
goals.
The lifecycle model therefore emits an empty `title` for every action; Oko
fills a newly started goal's title through the separate title model before
validating or applying the decision.

The two source splits are reviewed independently through Brama, with resumable
JSONL outputs:

```sh
transcript-label-trainer lifecycle-review path/to/train.jsonl \
  --output ~/.transcript-label-trainer/lifecycle-model/reviewed-train.jsonl \
  --split train --brama-model=best
transcript-label-trainer lifecycle-review path/to/eval.jsonl \
  --output ~/.transcript-label-trainer/lifecycle-model/reviewed-eval.jsonl \
  --split eval --brama-model=best
```

When the decision semantics change, prior source rows are reviewed again rather
than retaining labels produced under the old prompt. A contract-only ownership
change, such as moving title generation out of this model, is normalized by
`assemble-curriculum-splits.py` without changing the reviewed action. Deterministic
hard-case curricula from `training/lifecycle-model/curriculum/generate-curriculum.py` are
also sent through Brama; the assembler keeps only examples whose independent
review agrees with the intended action and keeps evaluation curriculum disjoint
from training.

The reviewed files are then submitted together to one exclusive Stado GPU
target:

```sh
transcript-label-trainer lifecycle-model \
  ~/.transcript-label-trainer/lifecycle-model/reviewed-train.jsonl \
  ~/.transcript-label-trainer/lifecycle-model/reviewed-eval.jsonl \
  --compute-target ubuntu-server-rtx-pro-6000
```

The job fine-tunes the pinned Qwen3-4B base, evaluates the untouched reviewed
split, converts and quantizes the model to Q4_K_M GGUF, and runs an independent
Brama `--best` audit over every held-out prediction. Publication requires at
least 99% valid JSON, 90% action accuracy, 88% joint accuracy, perfect finish
precision, and a passing independent audit. Qualified artifacts are
content-addressed under
`stado://releases/oko/models/lifecycle-qwen3-4b/<model-sha256>`; an unqualified
candidate remains available for diagnosis but cannot enter that namespace.

Qualification is measured on the shipped inference surface, not only in the
trainer. For the qualified lifecycle release, the 485-row held-out split on
MLX bf16/Metal produced 100% valid JSON, 94.23% action accuracy, 92.58% joint
accuracy, and 100% finish precision. GGUF/llama.cpp measurements did not meet
the joint gate, so the release manifest declares MLX weights and runtime.

