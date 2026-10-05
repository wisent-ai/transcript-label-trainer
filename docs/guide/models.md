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
  --limit 1500 --workers N --audit-workers N
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
  --compute-target TARGET --brama-url <the Brama address the job dials> \
  --eval-parallel N --eval-slot-context N --eval-gpu-layers N
```

The job fine-tunes the pinned Qwen3-4B base, evaluates the untouched reviewed
split, converts and quantizes the model to Q4_K_M GGUF, and runs an independent
Brama `--best` audit over every held-out prediction. Publication requires at
least 99% valid JSON, 90% action accuracy, 88% joint accuracy, perfect finish
precision, and a passing independent audit.

The quantized model is measured the way production serves it, by
`transcript-label-trainer lifecycle-evaluate-gguf`: `llama-server` on a loopback
port the system assigns, with `--eval-parallel` slots of `--eval-slot-context`
tokens and `--eval-gpu-layers` layers offloaded, each answer constrained to
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
