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
  --ster-options trl
```

The job trains with Ster on the GPU host, which must have `ster` installed (`stado
product install ster --surface cli`; without it the job stops at `ster: command not
found`): the reviewed training rows become `lifecycle-examples`, `ster tune sft` fits an
adapter to the pinned Qwen3-4B base with exactly the `--ster-options` given (an empty value
is refused before anything is uploaded, and Ster refuses a run missing one of its required
settings by name), and `ster tune merge` folds it into a checkpoint; Ster's report is kept
as `metrics.json`. The job's output URI and workspace are keyed by the datasets, expanded
options and exact trainer source revision. Source admission precedes upload. llama.cpp's converter then exports
and quantizes it to Q4_K_M GGUF, the untouched reviewed split is evaluated on that GGUF, and
an independent Brama `--best` audit judges every held-out prediction. Publication requires
the passing audit; the served rates are recorded in the manifest beside it.

The quantized model is measured the way production serves it, by
`transcript-label-trainer lifecycle-evaluate-gguf`: `llama-server` on a loopback
port the system assigns, sized by llama-server itself as for the goal model, asked
on as many concurrent requests as the slots it chose (its `/props` `total_slots`;
an answer without a positive one fails the run), each answer decoded greedily, as
Oko asks the model, and constrained to `lifecycle-output-schema.json` narrowed to
the row's candidate references. The evaluation waits for the server's health
answer (reading its log between probes; a server that exits first fails the run
with its status and log path) and for every answer; a failed request fails the run
naming its row, and nothing is retried. Inference refusals include the HTTP
status, endpoint and complete response body; malformed successful replies include
their body and JSON parse cause. An error status is never accepted as a prediction
even if it contains a message. An answer outside the schema counts as invalid, and
why it is outside is written on stderr beside its row. It writes
`predictions-gguf.jsonl` and `metrics-gguf.json`; a rate with nothing to measure
(finish precision when no finish was predicted) is `null`.

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
