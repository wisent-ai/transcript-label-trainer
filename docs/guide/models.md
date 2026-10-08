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
- The teacher defaults to Brama's `best` alias, the strongest operator
  subscription the signed identity may use; override it with `--brama-model`.
  Brama resolves the alias and reports unavailable capabilities.
- Auth: HMAC-signed requests as `WISENT_APP_AGENT_ID` (required), keyed by
  `WISENT_APP_AGENT_AUTH_SECRET` or else the vault role `TLT_BRAMA_AGENT_ROLE`
  declares as `ROLE#FIELD`; the bearer is `BRAMA_TOKEN` or else the role
  `TLT_BRAMA_TOKEN_ROLE` declares. Roles are read with `stado credentials get
  --role`, so no vault item is named and replacing one changes nothing here;
  an unset or malformed variable is refused with its name. The endpoint is
  `BRAMA_URL` (falling back to jeden's configured URL, then Stado's service
  directory). Secrets are read into memory only, never printed. A Stado job
  receives the same roles as `--secret-env ENV=ROLE#FIELD`, plus
  `TLT_HF_TOKEN_ROLE` for the humanizer's Hugging Face token.

The end-to-end story: autolabel an aspect, then train on the teacher's
labels by naming the provenance in a job spec:

```yaml
name: tasktype-v1
task: classify what kind of work the session did
evaluator: brama:best
model: tfidf-logreg
scope:
  aspect: tasktype
```

## Reviewed Jeden goal model

`goal-model` owns the complete small-model pipeline that turns coding-agent
messages into the 3–7 word task goals Jeden displays. It reads messages only
from Transcript Lake's normalized `events` view; raw agent session files are
not an input, so the lake's masking boundary remains intact.

Replace `TARGET` below with the registered Stado GPU target selected for training.

```sh
transcript-label-trainer goal-model \
  --compute-target TARGET \
  --limit 1500 --workers N --audit-workers N \
  --ster-options '--rank R --alpha A --epochs E --learning-rate L --accumulation N --max-sequence T --batch-size B --seed S'
```

The command generates task and no-task labels with a Brama teacher, then requires
two review passes through the pinned Brama reviewer before any row enters the
dataset. It holds out reviewed OMP titles plus 32 teacher task rows and 32
teacher no-task rows as gold, and submits the JSONL and exact trainer commit to
the named Stado target. The GPU host must have `ster` installed (`stado product
install ster --surface cli`; without it the job stops at `ster: command not found`).
There the reviewed teacher rows become `transcript-label-trainer goal-examples`: each
one the chat Jeden serves, the goal system prompt as its `system` turn,
`<user>message</user>` as its `prompt` and the reviewed `<goal>…</goal>` or `<goal/>`
as its `completion`. `ster tune sft` fits an adapter to pinned
`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` with exactly the
`--ster-options` given (an empty value is refused before anything is uploaded),
`ster tune merge` folds it in, and llama.cpp's converter exports and quantizes it to
Q4_K_M. Gold rows never enter training.

`transcript-label-trainer goal-evaluate-gguf` then asks that quantized model for every
gold row the way Jeden serves it: `llama-server` on a loopback port the system assigns,
sized by llama-server itself from the model and the device — its documented defaults
`--parallel -1` (auto slots), `--ctx-size 0` (the model's context, fitted to device
memory by `--fit on`) and `--gpu-layers auto`
([llama-server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)),
so the job states none of them — the same system prompt and user turn, decoding constrained to
`<goal/>` or one `<goal>…</goal>` line. It writes `predictions.jsonl` and the exact-match
share in `metrics-gguf.json`; a server that exits before it is ready fails with its
status and log, and a failed request fails the run naming its session. Every one of
those served predictions must receive `both-sensible` from a final Brama `--best` audit
or the model remains unqualified; a rejected candidate is retained for diagnosis but
cannot enter the Jeden Desktop release namespace. To build an example set and see how
the base model answers it before any job:

```sh
transcript-label-trainer goal-examples --rows reviewed-goals.jsonl --output goal-examples.json
ster tune evaluate --model Qwen/Qwen3-4B --examples goal-examples.json --max-sequence 2048 --batch-size 1
```

`goal-examples` refuses a file without a row and a file whose every row is gold.

A qualified job publishes the Q4_K_M GGUF, Ster's training report (`metrics.json`), the
served predictions and their `metrics-gguf.json`, the full final audit, canonical prompt,
the converter's dependency lock, and checksums under the content-addressed URI printed as
`model artifact: stado://probierz/artifacts/models/jeden/goal-qwen3-4b/<key>`, the key
being the dataset and the `--ster-options` together, so other settings never resume or
overwrite a model.

Stado also retains its canonical `status/<job-id>/output/` copy. All model calls
use `brama.rs`; the pipeline has no direct provider credentials or second auth
implementation. The job's `model-manifest.json` is written by `transcript-label-trainer
model-manifest --model goal|lifecycle|humanizer --output-dir OUT [--artifact GGUF]`: each
evidence file's size and SHA-256, the ordered `<artifact>.part-*` parts and the artifact they
rebuild, and `qualified` from the independent gate (`final-judge.json` passed; for the
humanizer, `publication.json` qualified and `audit.json` passed, refused with both values when
not). A missing input is refused by name, `--artifact` is required for goal and lifecycle and
refused for the humanizer. The lifecycle manifest records the served and training metrics and
gates on the judge alone.

A qualified job output enters the Jeden Desktop release namespace through
`transcript-label-trainer release-publish <job-output-uri> --model goal`, which
publishes under `stado://releases/jeden-desktop/models/goal-qwen3-4b/<model-sha256>`.
Each final-audit record carries the judged `message`, reference `goal` and
`student` output beside its `verdict`, so a rejected prediction is read from
`final-judge.json` itself.

The qualified 4B release is also public at
[`lbartoszcze/jeden-goal-qwen3-4b`](https://huggingface.co/lbartoszcze/jeden-goal-qwen3-4b),
revision `d9ce79f106ead1176b74bb0d9fb875521ca712b1`. Its 2,497,280,320-byte
GGUF has SHA-256
`2512d7a455a50a16742b75d8fe38bf02b46b5d6b607f785be32a6345d999d310`,
the same immutable artifact consumed by Jeden Desktop.

This immutable public model is a release reference, not publication configuration.
New qualified goal outputs use `release-publish`; the completed, fixed-artifact
Hugging Face repository migration is not a reusable publication workflow.

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
  --split train --brama-model=best --workers N
transcript-label-trainer lifecycle-review path/to/eval.jsonl \
  --output ~/.transcript-label-trainer/lifecycle-model/reviewed-eval.jsonl \
  --split eval --brama-model=best --workers N
```

When the decision semantics change, prior source rows are reviewed again rather
than retaining labels produced under the old prompt. A contract-only ownership
change, such as moving title generation out of this model, is normalized by
`transcript-label-trainer lifecycle-assemble-curriculum` without changing the reviewed
action. Deterministic hard-case curricula are written by
`transcript-label-trainer lifecycle-generate-curriculum --source reviewed-train.jsonl
--output curriculum.jsonl --per-family N --seed N` from the families and sentences in
`training/lifecycle-model/curriculum/templates.json`; each row copies a real envelope
with an active candidate, keeps only the fields Oko sends, and records its family and
intended action. The curriculum is also sent through Brama; the assembler keeps only
examples whose independent review agrees with the intended action and keeps
evaluation curriculum disjoint from training:

```sh
transcript-label-trainer lifecycle-assemble-curriculum \
  --base-train reviewed-train.jsonl --base-eval reviewed-eval.jsonl \
  --curriculum-train reviewed-curriculum-train.jsonl \
  --curriculum-eval reviewed-curriculum-eval.jsonl \
  --output-train assembled-train.jsonl --output-eval assembled-eval.jsonl \
  --minimum-eval-per-action N
```

Every row's one decision is checked against the decision contract, the actions the
evaluation minimum covers are the ones `lifecycle-output-schema.json` declares, and it
refuses an underrepresented evaluation curriculum, a duplicate id within a split, and an
id shared between the splits; it prints the kept, refused and per-action counts.

To hold a different day out for evaluation, `transcript-label-trainer lifecycle-split-by-day
--train reviewed-train.jsonl --eval reviewed-eval.jsonl --eval-day DAY --output-train
new-train.jsonl --output-eval new-eval.jsonl` merges the reviewed rows by id (refusing one
id with two different rows), puts the rows of that `split_day` into evaluation and the rest
into training, and refuses a split that lacks an action the output schema declares.

The same reviewed rows are Ster labelled decisions with `transcript-label-trainer
lifecycle-decisions --rows reviewed-train.jsonl --output lifecycle-decisions.json`: each
row's decision is checked against the contract, the masked input envelope is the state,
the questions are the ones `training/lifecycle-model/decision/questions.json` declares
(goal_ref offers the row's own candidates by their titles), and the reviewed action,
goal_ref and lifecycle_evidence are the answers. That document is what `ster decisions
benchmark` measures, for example:

```sh
transcript-label-trainer lifecycle-decisions --rows reviewed-eval.jsonl --output eval-decisions.json
ster decisions benchmark --model Qwen/Qwen3-4B --examples eval-decisions.json --output eval-benchmark.json
```

A row without exactly one reviewed decision, a decision that breaks the contract, or a
candidate without a reference or a title is refused by row id.

The model itself is trained on the chat it is served. `transcript-label-trainer
lifecycle-examples --rows reviewed-train.jsonl --output lifecycle-examples.json` writes each
reviewed row as one `ster tune sft` example: the lifecycle system prompt as its `system`
turn, the row's user envelope as the `prompt`, and the reviewed decision, its title
blanked, as the JSON `completion`, with the same row refusals as `lifecycle-decisions`.
To see how far the base model is from the reviewed answers before training:

```sh
transcript-label-trainer lifecycle-examples --rows reviewed-eval.jsonl --output eval-examples.json
ster tune evaluate --model Qwen/Qwen3-4B --examples eval-examples.json --max-sequence 4096 --batch-size 1
```

The reviewed files are then submitted together to one exclusive Stado GPU
target. Every job (goal, lifecycle, humanizer and `run --compute-target`) is pinned to the
`--compute-target` host, which decides the hardware; the goal, lifecycle and humanizer jobs
claim its whole GPU. None states a priority, GPU label or VRAM of its own. Its Stado run id is the model name, the key of its
data and settings and the source commit, so submitting the same run again answers the job
Stado already holds instead of queueing a second one:

```sh
transcript-label-trainer lifecycle-model \
  ~/.transcript-label-trainer/lifecycle-model/reviewed-train.jsonl \
  ~/.transcript-label-trainer/lifecycle-model/reviewed-eval.jsonl \
  --compute-target TARGET --brama-url <the Brama address the job dials> \
  --ster-options '--rank R --alpha A --epochs E --learning-rate L --accumulation N --max-sequence T --batch-size B --seed S'
```

The job trains with Ster on the GPU host, which must have `ster` installed (`stado
product install ster --surface cli`; without it the job stops at `ster: command not
found`): the reviewed training rows become `lifecycle-examples`, `ster tune sft` fits an
adapter to the pinned Qwen3-4B base with exactly the `--ster-options` given (an empty value
is refused before anything is uploaded, and Ster refuses a run missing one of its required
settings by name), and `ster tune merge` folds it into a checkpoint; Ster's report is kept
as `metrics.json`. The job's output URI is keyed by the datasets and the options together,
so other settings never resume or overwrite a model. llama.cpp's own converter then exports
and quantizes it to Q4_K_M GGUF, the untouched reviewed split is evaluated on that GGUF, and
an independent Brama `--best` audit judges every held-out prediction. Publication requires
the passing audit; the served rates are recorded in the manifest beside it.

The quantized model is measured the way production serves it, by
`transcript-label-trainer lifecycle-evaluate-gguf`: `llama-server` on a loopback
port the system assigns, sized by llama-server itself as for the goal model, asked
on as many concurrent requests as the slots it chose (its `/props` `total_slots`;
an answer without a positive one fails the run), each answer constrained to
`lifecycle-output-schema.json` narrowed to the row's candidate references. The
evaluation waits for the server's health answer (reading its log between probes;
a server that exits first fails the run with its status and log path) and for
every answer; a failed request fails the run naming its row, and nothing is
retried. It writes `predictions-gguf.jsonl` and `metrics-gguf.json`; a rate with
nothing to measure (finish precision when no finish was predicted) is `null`.

Oko asks Brama for this model under the alias `OKO_LIFECYCLE_MODEL_ALIAS`
declares, and Brama routes that alias to a model server on a fleet GPU host. The
weights therefore go where that host fetches them, a private Hugging Face
repository, and never into the fleet's release store, which lives on the vault
host where no model runs:

```sh
transcript-label-trainer release-publish <job-output-uri> --model lifecycle \
  --repo OWNER/PRIVATE_REPOSITORY
```

The command joins the verified parts into the qualified artifact, uploads it
with `model-manifest.json` and every evidence file as one new
`lifecycle-publication-<random>` branch, reads that branch's immutable revision
back and checks each file's size and SHA-256 against what it verified. It prints
`repo_id`, `revision`, `artifact`, `digest` and `verified_files`; the serving
host pins that revision. `HF_TOKEN` authenticates the upload. An unqualified
candidate remains in its job output for diagnosis and is never published.

`release-publish` refuses, with exit 1 and a sentence naming the cause, when the
manifest is not qualified, does not require `final-judge.json`, names another
contract, when the judge did not pass or is incomplete, when any evidence file
or model part differs from the manifest's SHA-256, or when the ordered parts do
not rebuild the qualified artifact. It also refuses before downloading anything
when `--model lifecycle` has no `--repo` (`the lifecycle model is served from a
GPU host, which fetches it from a private Hugging Face repository: name it with
--repo OWNER/REPOSITORY; the fleet's release store is not a model store`), when
`--model goal` is given a `--repo` (the goal model ships inside Jeden Desktop
through the release channel), and when the named repository exists and is
public. For the goal model, objects already present at the digest coordinate are
left untouched, so a rerun resumes where a failed one stopped.

Qualification is measured on the shipped inference surface, not only in the
trainer. For the qualified lifecycle release, the 485-row held-out split on
MLX bf16/Metal produced 100% valid JSON, 94.23% action accuracy, 92.58% joint
accuracy, and 100% finish precision. GGUF/llama.cpp measurements did not meet
the joint gate, so the release manifest declares MLX weights and runtime.

## Private personal-voice publication

Choose the destination explicitly before starting the humanizer:

```sh
HUMANIZER_HF_REPO=OWNER/PRIVATE_MODEL \
  transcript-label-trainer humanizer-model --compute-target TARGET
```

The controller forwards that repository into the Stado job. A missing or empty
destination is refused before submission; the worker and publisher also require
it. No account is selected from source code.

The job trains with Ster on the GPU host (`stado product install ster --surface cli`;
without it the job stops at `ster: command not found`). `transcript-label-trainer
humanizer-examples --rows train.jsonl --output examples.json` writes each prepared row as
one `ster tune sft` example (the row's system prompt, the generic source as the prompt,
the user's own target as the completion; a row without exactly one non-empty system,
user and assistant turn is refused by id). `ster tune sft` fits the adapter to the pinned
`TheDrummer/Cydonia-24B-v4.3` base with exactly the `--ster-options` given to
`humanizer-model` (an empty value is refused before the corpus is exported), and its
report is kept as `training.json`. `ster tune export --format peft` writes the adapter
as the vLLM/PEFT directory `humanizer-publish` publishes (`adapter_config.json`,
`adapter_model.safetensors`, the base's `tokenizer.json`), and `ster tune merge` folds
it into the student checkpoint the evaluation asks.

`transcript-label-trainer humanizer-evaluate-gguf` asks the base and the merged
student, both converted to full-precision GGUF so the student answers exactly as the base
with the published adapter attached, for every test row through `llama-server`, one model
after the other, each sized by llama-server itself as for the goal model and each answer
running until the model ends it (llama-server's default). Each answer is scored by chrF
as [sacrebleu](https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py)
computes it by default — character n-grams of orders one to six with whitespace removed
and case kept, precision and recall averaged over the orders, combined with recall weighted
twice (beta 2), all recorded under `chrf` in `metrics.json` — against the target and the
source and by its length against the target's; `predictions.jsonl` carries both answers
for `humanizer-audit`, and `metrics.json` the means and the target chrF gain the manifest
records. A model file that does not exist is refused before a server starts, and a failed
request fails the run naming its row. To see an example set before any job:

```sh
transcript-label-trainer humanizer-examples --rows train.jsonl --output humanizer-examples.json
ster tune evaluate --model TheDrummer/Cydonia-24B-v4.3 --examples humanizer-examples.json \
  --max-sequence 1024 --batch-size 1 --device cuda --precision f16
```

Publication requires the existing model contract and a passed audit. The native
publisher can also publish a retained qualified job output directly:

```sh
transcript-label-trainer humanizer-publish JOB_OUTPUT/adapter \
  --repo OWNER/PRIVATE_MODEL --metrics JOB_OUTPUT/metrics.json \
  --audit JOB_OUTPUT/audit.json --preparation JOB_OUTPUT/preparation.json \
  --output publication.json
```

`HF_TOKEN` authenticates both metadata reads and transfers. The supported `hf`
CLI, version 0.36 or later within the 0.x series, must be installed; `HF_BIN`
selects its executable. `HF_ENDPOINT` optionally selects the Hub base URL.
The Stado worker installs that CLI and invokes the same native publisher.
The output path must not already exist. An existing path is refused before
provider access, so a receipt cannot overwrite training inputs or older evidence.

A public destination is refused before reading or uploading model files.
The receipt reports `repository_not_private`, the `model_info` operation, the
repository and its observed privacy state. Publication never changes repository
visibility. It checks privacy before each transfer and at the final revision.
Each attempt uses its own publication branch, so another writer's main branch
cannot be mistaken for the published commit.

The output receipt records the immutable revision, privacy, file inventory and
verified SHA-256 values. Required adapter files and evidence must match the
pre-transfer inputs: LFS objects are checked by their content-addressed digest,
and ordinary files are read from the immutable revision and hashed. Missing
files, changed bytes, an unpassed audit, a different metrics contract or a
provider error returns exit 1 with `qualified: false` and the actual cause.
The GUI owns corpus import and documentation; model publication uses this CLI
and the Stado training job, not a separate graphical publisher.

### Real publication qualification

`tests/publication/` exercises this command against Hugging Face itself:

```sh
cargo test --locked --test publication
```

Provide `HF_TOKEN` with permission to create and delete isolated model
repositories. `TLT_PUBLICATION_NAMESPACE` optionally selects an authorized
namespace; otherwise the live authenticated account is used. No account name
is stored in source. `TLT_PUBLICATION_FIXTURE` must point to an actual qualified
humanizer job output containing `adapter/`, `metrics.json`, `audit.json` and
`preparation.json`; the runner does not manufacture a passed audit or weights.

The public-destination journey creates an empty isolated public repository,
requires refusal without visibility or revision changes, and deletes it.
The private journey publishes the real fixture into a new private repository,
downloads the required files at the reported immutable revision, compares their
bytes with the source hashes, deletes the repository and confirms its absence.
Every repository has a fresh random name; existing repositories are refused.

The runner requires a clean committed checkout and retains the source revision,
binary checksum, commands, exit statuses, provider observations and receipts
under the ignored `.build/publication-*/report.json` directories. Payload download
caches are removed; reports stay private. Exit 0 means both journeys passed,
exit 1 means failure, and exit 2 means a prerequisite is missing. A blocked
journey is not a pass.
