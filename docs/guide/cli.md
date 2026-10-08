<!-- Moved out of README.md; the README links here. -->
## CLI

One binary exposes the commands below. Global flags: `--training-root PATH` (where
model artifacts and adopted corpora live; beats `$TLT_HOME` and the Stado
registry declaration), `--storage-root PATH` (the live lake data root; beats
`$LAKE_DATA` and the registry), and `-V` / `--version`. Global root flags must
precede the subcommand. Every subcommand answers `--help` with its full
contract.

First use:

| command | does |
|---|---|
| `onboarding` | walk the published first-use journey to your first suggestions |
| `corpus-adopt BUNDLE [--json]` | validate, retain, and select an existing canonical dataset bundle; reports imported, unchanged, conflicting, and rejected counts |
| `corpus-status [--json]` | show the selected adopted corpus and every retained corpus |
| `corpus-select CORPUS [--json]` | make a retained corpus the input train, infer and evaluate read |
| `corpus-remove CORPUS [--json]` | remove a retained corpus and its bundle; refused for the selected one while others remain; removing the last one removes the registry |
| `gui [--bind IP] [--port PORT]` | serve the embedded graphical corpus importer, the Train panel and navigable HTML documentation on a loopback listener; prints but does not open its session-token URL. The Train panel runs `train` (POST `/api/train` with `aspect`, `model`, `eval_split` and `training`, each setting as typed text read like `train`'s flags; its fill button types a preset's documented values) and answers with the metrics or the same refusal `train` prints; `not_enough_data` marks the too-few-labels refusal |

Aspect classifiers (local, cheap, sklearn or HF):

| command | does |
|---|---|
| `train` | train a classifier for one aspect from manual lake labels |
| `run` | execute a declarative training job (YAML spec) |
| `evaluate` | score a trained model on its frozen holdout and have a Brama teacher judge whether the predictions are acceptable |
| `infer` | emit label suggestions for unlabeled sessions; never writes to the lake |
| `info` | list trained aspects, artifacts, and metrics |
| `autolabel` | label every unlabeled session for an aspect via a Brama teacher (zero-touch) |
| `aspect-discover` | propose new aspect dimensions from recent sessions via a Brama teacher (writes nothing). The teacher is first asked about every session in one call; when Brama answers `context_length_exceeded`, each half is asked instead, so the routed model's own context decides the batch. A session that does not fit alone is listed under `failures` with Brama's answer |

Goal models (fine-tunes trained on a Stado GPU target, gated before publish):

| command | does |
|---|---|
| `goal-model` | curate masked lake messages, teacher-label task goals through Brama, require an independent `best` review, train the served model with `ster tune sft` on the named exclusive Stado GPU target with exactly the `--ster-options` given (required; an empty value is refused as `--ster-options cannot be empty`), ask the quantized model for every held-out gold row with `goal-evaluate-gguf`, and publish GGUF artifacts only after those served predictions pass a second `best` audit. `--limit N` (most teacher-labeled candidates) is required, at least 1; the teacher and audit calls run as many at once as Brama measured each route to carry (see [Concurrency](#concurrency)) |
| `goal-examples` | write `--rows` (reviewed goal JSONL) to `--output` as the Ster example set `ster tune sft` trains on: every row not marked gold, the goal system prompt as its `system`, `<user>message</user>` as its `prompt` and `<goal>…</goal>` or `<goal/>` as its `completion`; answers the example and held-out counts, and refuses a file without a row or whose every row is gold |
| `goal-evaluate-gguf` | start `llama-server` (`--server`) on the quantized `--model` on a system-assigned loopback port, sized by llama-server itself (slots `--parallel -1` auto, context loaded from the model and fitted to device memory, `--gpu-layers auto`, its documented defaults), wait for its health answer, ask every gold row of `--dataset` the way Jeden serves it with decoding constrained to `<goal/>` or one `<goal>…</goal>` line, and write `--predictions` (what `goal-audit` reads) and the exact-match share to `--metrics`; a failed request fails the run naming its session; the server log goes to `--server-log` |
| `goal-audit` | independently audit student goal predictions (JSONL of message, reference goal, student output) with as many parallel Brama calls as Brama measured the route to carry, and write the complete audit record |
| `lifecycle-review` | classify masked Oko training envelopes through a named Brama route with as many parallel reviews as Brama measured that route to carry, enforce the `oko-goal-lifecycle-v1` contract, and write ordered JSONL with reviewer provenance for an immutable `--split train\|eval` |
| `lifecycle-model` | upload immutable reviewed train and held-out datasets, train the served model with `ster tune sft` on the named Stado GPU target with exactly the `--ster-options` given (required; an empty value is refused as `--ster-options cannot be empty`), measure the quantized model with `lifecycle-evaluate-gguf`, audit every held-out decision through Brama `best` with as many concurrent calls as Brama measured `best` to carry, and publish the candidate only when at most `--audit-max-wrong-share F` of the decisions are wrong (required) |
| `lifecycle-examples` | write `--rows` (reviewed lifecycle JSONL) to `--output` as the Ster example set `ster tune sft` trains on: the lifecycle system prompt as each example's `system`, the user envelope as its `prompt`, the reviewed decision with its title blanked as its `completion`; a row without one reviewed decision or with a decision outside the contract is refused by id |
| `lifecycle-evaluate-gguf` | start `llama-server` (`--server`) on the quantized `--model` on a system-assigned loopback port, sized by llama-server itself as `goal-evaluate-gguf` describes, wait for its health answer, read the slots it chose from its `/props` `total_slots` (an answer without a positive one fails the run naming it) and ask every `--dataset` row on that many concurrent requests with decoding constrained to `--output-schema` narrowed to the row's candidates, and write `--predictions` and `--metrics`; a failed request fails the run naming its row; the server log goes to `--server-log` |
| `lifecycle-audit` | judge every held-out student decision independently with as many concurrent Brama calls as Brama measured the route to carry, reject inferred completion, retain the full verdict record with its `thresholds`, and fail the gate when more than `--max-wrong-share F` of the decisions (a share between none and all) are semantically wrong, any is unjudgeable or a dangerous finish, or any audit call failed. `--max-wrong-share` is required; a share outside that range is refused as `--max-wrong-share must be a share from 0 to 1, not <F>` |
| `humanizer-model` | require an explicit `HUMANIZER_HF_REPO`, the corpus bounds `--limit N` (most targets taken), `--min-targets N` (fewer refuses the export as `humanizer corpus produced only <n> clean targets; --min-targets asks for <N>`) and `--max-per-session N`, the authored-text bounds a user turn must meet to count as written by hand (`--min-target-chars N`, `--max-target-chars N`, `--max-target-lines N`, `--min-target-words N` words holding a letter, `--min-meaningful-share F` of characters that are letters, digits or spaces; recorded under `authored` in the export summary) and `--attempts N` (the times one question is asked; no default), export masked likely-authored user turns, derive inverse style-transfer inputs through Brama (`humanizer-prepare`), train a LoRA adapter on the pinned base, audit it against that base (`humanizer-audit`), and publish only a qualified private adapter revision; its Brama calls run as many at once as Brama measured each route to carry |
| `humanizer-model` training | the job trains with `ster tune sft` and also requires `--ster-options OPTIONS` (an empty value is refused as `--ster-options cannot be empty`), checked before the corpus is exported; its output URI is keyed by the targets and the options together |
| `humanizer-examples` | write a prepared split (`--rows`) to `--output` as the Ster example set `ster tune sft` trains on: each row's system prompt as `system`, the generic source as `prompt`, the user's target as `completion`; a row without exactly one non-empty system, user and assistant turn is refused by id, a file without a row by name |
| `humanizer-evaluate-gguf` | ask `--base` and then `--student` (full-precision GGUFs) for every row of `--dataset` through `llama-server` (`--server`, `--server-log`; sized by llama-server itself, each answer running until the model ends it), score each answer by chrF as sacrebleu computes it by default (character orders one to six, whitespace removed, case kept, beta 2; recorded under `chrf` in the metrics) against target and source and by length against the target, and write `--predictions` (what `humanizer-audit` reads) and `--metrics` (means, target chrF gain, `--base-model`, `--base-revision`); a missing model is refused before a server starts, a failed request fails the run naming its row |
| `humanizer-prepare` | turn the exported targets into session-separated `train.jsonl` and `test.jsonl` plus `preparation.json`: a Brama teacher (`--teacher-model`, default `best`) rewrites each target as generic AI prose, rewrites that drop a URL, e-mail address or number are refused as `source_contract`, and an independent review (`--review-model`, default `best`) must call the pair usable; how long a rewrite may be is the review's judgement, not a ratio. Each session goes to a split by its digest: below the 0.25 [scikit-learn's `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) documents as its test share to test, the rest to train; `preparation.json` records it as `test_share`. `--attempts N` is required, at least 1; the calls run as many at once as Brama measured the route to carry. Refuses with exit 1 and writes nothing when either split holds no accepted row; the message names the split and the rejection reasons. The Stado job runs it before training |
| `humanizer-audit` | judge every held-out case of `predictions.jsonl` (`id`, `source`, `target`, `base`, `student`) through `--brama-model` (default `best`), write the complete record to `--output`, and exit 1 unless the adapter beats the base it was trained from on the same cases: higher mean voice match, semantic fidelity and pass rate no lower, AI-boilerplate rate no higher. No bound is stated; the base's measured scores are the bound, and the record's `gate` names the rule. `--attempts N` is required, at least 1, and the calls run as many at once as Brama measured the route to carry; a case the judge cannot answer in `--attempts` asks fails the audit as `audit incomplete` and writes nothing |
| `humanizer-publish MODEL` | require `--repo OWNER/REPOSITORY`, `--metrics`, `--audit`, `--preparation` and `--output`; refuse public destinations before reading model files; publish on a separate branch and verify the private immutable revision and required file checksums; write a JSON receipt and exit 1 on refusal or failure |
| `release-publish` | move a qualified job output (`SOURCE` Stado URI, `--model goal\|lifecycle`) into its release namespace under the artifact's SHA-256 after verifying the manifest, the passed final judge, every file checksum and the reassembled parts; see [models.md](models.md) for the refusals |

Exit statuses across the CLI: `0` success, `1` failed command, `2` usage
error or a run the lake does not hold enough labeled data for.

### Goals quickstart

The goal path in three commands. Replace `TARGET` with a registered Stado GPU target:

```sh
# 1. Title model: curate, teacher-label, review, train, audit, publish GGUF.
transcript-label-trainer goal-model --compute-target TARGET --limit N \
  --ster-options trl

# 2. Lifecycle datasets: review masked envelopes into immutable splits.
transcript-label-trainer lifecycle-review envelopes.jsonl \
  --split train --output reviewed-train.jsonl
transcript-label-trainer lifecycle-review held-out.jsonl \
  --split eval --output reviewed-eval.jsonl

# 3. Lifecycle model: train, audit every held-out decision, gate, publish.
transcript-label-trainer lifecycle-model reviewed-train.jsonl reviewed-eval.jsonl \
  --compute-target TARGET --brama-url <the Brama address the job dials> \
  --ster-options trl \
  --audit-max-wrong-share F
```

Each step refuses to continue when its gate fails: no reviewed dataset, no
training; no passed audit, no publication. There is no flag that skips a gate.

### Concurrency

No command takes a worker count. Every command that fans out to a Brama route
reads the route's `measured_concurrency` from Brama's `GET /v1/aliases`:
`carried` is the most calls in flight the route has answered at once. Without
a capacity refusal at that level the command sends one call more than
`carried`, so the measurement grows run by run; after a capacity refusal with
`carried` or fewer calls beside it, it stays at `carried`. A route Brama has
measured nothing for, or one refused with nothing else in flight, gets one
call at a time. A Brama that cannot be reached fails the command with
`Brama unreachable at <url> reading its measured concurrency: <error>`.

