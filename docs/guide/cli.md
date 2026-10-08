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
| `goal-model` | curate masked lake messages, teacher-label task goals through Brama, require an independent `best` review, train the served model with `ster tune sft` on the named exclusive Stado GPU target with exactly the `--ster-options` given (required; an empty value is refused as `--ster-options cannot be empty`), ask the quantized model for every held-out gold row with `goal-evaluate-gguf`, and publish GGUF artifacts only after those served predictions pass a second `best` audit. `--limit N` (most teacher-labeled candidates), `--workers N` (parallel teacher calls while curating) and `--audit-workers N` (parallel calls in the job's final audit) are required, at least 1 |
| `goal-examples` | write `--rows` (reviewed goal JSONL) to `--output` as the Ster example set `ster tune sft` trains on: every row not marked gold, the goal system prompt as its `system`, `<user>message</user>` as its `prompt` and `<goal>…</goal>` or `<goal/>` as its `completion`; answers the example and held-out counts, and refuses a file without a row or whose every row is gold |
| `goal-evaluate-gguf` | start `llama-server` (`--server`) on the quantized `--model` on a system-assigned loopback port, sized by llama-server itself (slots `--parallel -1` auto, context loaded from the model and fitted to device memory, `--gpu-layers auto`, its documented defaults), wait for its health answer, ask every gold row of `--dataset` the way Jeden serves it with decoding constrained to `<goal/>` or one `<goal>…</goal>` line, and write `--predictions` (what `goal-audit` reads) and the exact-match share to `--metrics`; a failed request fails the run naming its session; the server log goes to `--server-log` |
| `goal-audit` | independently audit student goal predictions (JSONL of message, reference goal, student output) with `--workers N` parallel Brama calls (required, at least 1) and write the complete audit record |
| `lifecycle-review` | classify masked Oko training envelopes through a named Brama route with `--workers N` parallel reviews (required, at least 1), enforce the `oko-goal-lifecycle-v1` contract, and write ordered JSONL with reviewer provenance for an immutable `--split train\|eval` |
| `lifecycle-model` | upload immutable reviewed train and held-out datasets, train the served model with `ster tune sft` on the named Stado GPU target with exactly the `--ster-options` given (required; an empty value is refused as `--ster-options cannot be empty`), measure the quantized model with `lifecycle-evaluate-gguf`, audit every held-out decision through Brama `best` with `--audit-workers N` concurrent calls, and publish the candidate only when at most `--audit-max-wrong-share F` of the decisions are wrong (both required) |
| `lifecycle-examples` | write `--rows` (reviewed lifecycle JSONL) to `--output` as the Ster example set `ster tune sft` trains on: the lifecycle system prompt as each example's `system`, the user envelope as its `prompt`, the reviewed decision with its title blanked as its `completion`; a row without one reviewed decision or with a decision outside the contract is refused by id |
| `lifecycle-evaluate-gguf` | start `llama-server` (`--server`) on the quantized `--model` on a system-assigned loopback port, sized by llama-server itself as `goal-evaluate-gguf` describes, wait for its health answer, read the slots it chose from its `/props` `total_slots` (an answer without a positive one fails the run naming it) and ask every `--dataset` row on that many concurrent requests with decoding constrained to `--output-schema` narrowed to the row's candidates, and write `--predictions` and `--metrics`; a failed request fails the run naming its row; the server log goes to `--server-log` |
| `lifecycle-audit` | judge every held-out student decision independently with `--workers N` concurrent Brama calls (the route's own concurrency allowance), reject inferred completion, retain the full verdict record with its `thresholds`, and fail the gate when more than `--max-wrong-share F` of the decisions (a share between none and all) are semantically wrong, any is unjudgeable or a dangerous finish, or any audit call failed. Both are required; a share outside that range is refused as `--max-wrong-share must be a share from 0 to 1, not <F>` |
| `humanizer-model` | require an explicit `HUMANIZER_HF_REPO`, the corpus bounds `--limit N` (most targets taken), `--min-targets N` (fewer refuses the export as `humanizer corpus produced only <n> clean targets; --min-targets asks for <N>`) and `--max-per-session N`, the authored-text bounds a user turn must meet to count as written by hand (`--min-target-chars N`, `--max-target-chars N`, `--max-target-lines N`, `--min-target-words N` words holding a letter, `--min-meaningful-share F` of characters that are letters, digits or spaces; recorded under `authored` in the export summary), `--workers N` and `--attempts N` (the job's parallel Brama calls and the times one question is asked; no defaults), the six quality-gate bounds `humanizer-audit` takes (`--min-semantic-fidelity`, `--min-voice-match`, `--min-pass-rate`, `--max-boilerplate-rate`, `--min-voice-gain`, `--min-semantic-delta`) and the preparation bounds `humanizer-prepare` takes (`--min-train-rows`, `--min-validation-rows`, `--min-test-rows`, `--min-length-ratio`, `--max-length-ratio`, `--test-share`, `--validation-share`), export masked likely-authored user turns, derive inverse style-transfer inputs through Brama, train a LoRA adapter on the pinned base, audit it against that gate, and publish only a qualified private adapter revision |
| `humanizer-model` training | the job trains with `ster tune sft` and also requires `--ster-options OPTIONS` (an empty value is refused as `--ster-options cannot be empty`), checked before the corpus is exported; its output URI is keyed by the targets and the options together |
| `humanizer-examples` | write a prepared split (`--rows`) to `--output` as the Ster example set `ster tune sft` trains on: each row's system prompt as `system`, the generic source as `prompt`, the user's target as `completion`; a row without exactly one non-empty system, user and assistant turn is refused by id, a file without a row by name |
| `humanizer-evaluate-gguf` | ask `--base` and then `--student` (full-precision GGUFs) for every row of `--dataset` through `llama-server` (`--server`, `--server-log`; sized by llama-server itself, each answer running until the model ends it), score each answer by chrF as sacrebleu computes it by default (character orders one to six, whitespace removed, case kept, beta 2; recorded under `chrf` in the metrics) against target and source and by length against the target, and write `--predictions` (what `humanizer-audit` reads) and `--metrics` (means, target chrF gain, `--base-model`, `--base-revision`); a missing model is refused before a server starts, a failed request fails the run naming its row |
| `humanizer-prepare` | turn the exported targets into session-separated `train.jsonl`, `validation.jsonl` and `test.jsonl` plus `preparation.json`: a Brama teacher (`--teacher-model`, default `best`) rewrites each target as generic AI prose, rewrites that drop a URL, e-mail address or number or whose length falls outside `--min-length-ratio F`–`--max-length-ratio F` times the target's are refused as `source_contract`, and an independent review (`--review-model`, default `best`) must call the pair usable. Each session goes to a split by its digest: `--test-share F` of sessions to test, `--validation-share F` to validation, the rest to train; shares that are negative or leave train nothing are refused as `--test-share <F> and --validation-share <F> must be shares from 0 to 1 that together leave train a share`. `--workers N` and `--attempts N` are required, at least 1, and so are the length ratios, the shares and the split minimums `--min-train-rows N`, `--min-validation-rows N` and `--min-test-rows N`, recorded in `preparation.json` as `source_length_ratio`, `held_out_shares` and `split_minimums`. Refuses with exit 1 and writes nothing when any split has fewer accepted rows than its minimum; the message names the counts and the rejection reasons. The Stado job runs it before training, with the values `humanizer-model` was given |
| `humanizer-audit` | judge every held-out case of `predictions.jsonl` (`id`, `source`, `target`, `base`, `student`) through `--brama-model` (default `best`), write the complete record to `--output` with the `gate` it was held to, and exit 1 when the adapter misses that gate. Every bound is required and is the caller's to state: `--min-semantic-fidelity`, `--min-voice-match`, `--min-pass-rate`, `--max-boilerplate-rate`, `--min-voice-gain` (student voice match less the base's) and `--min-semantic-delta` (student semantic fidelity less the base's; negative allows a loss); a missing one is refused as `<flag> is required: this command assumes no value for it`. The Stado job reads them from `HUMANIZER_MIN_SEMANTIC_FIDELITY`, `HUMANIZER_MIN_VOICE_MATCH`, `HUMANIZER_MIN_PASS_RATE`, `HUMANIZER_MAX_BOILERPLATE_RATE`, `HUMANIZER_MIN_VOICE_GAIN` and `HUMANIZER_MIN_SEMANTIC_DELTA`. `--workers N` and `--attempts N` are required, at least 1; a case the judge cannot answer in `--attempts` asks fails the audit as `audit incomplete` and writes nothing |
| `humanizer-publish MODEL` | require `--repo OWNER/REPOSITORY`, `--metrics`, `--audit`, `--preparation` and `--output`; refuse public destinations before reading model files; publish on a separate branch and verify the private immutable revision and required file checksums; write a JSON receipt and exit 1 on refusal or failure |
| `release-publish` | move a qualified job output (`SOURCE` Stado URI, `--model goal\|lifecycle`) into its release namespace under the artifact's SHA-256 after verifying the manifest, the passed final judge, every file checksum and the reassembled parts; see [models.md](models.md) for the refusals |

Exit statuses across the CLI: `0` success, `1` failed command, `2` usage
error or a run the lake does not hold enough labeled data for.

### Goals quickstart

The goal path in three commands. Replace `TARGET` with a registered Stado GPU target:

```sh
# 1. Title model: curate, teacher-label, review, train, audit, publish GGUF.
transcript-label-trainer goal-model --compute-target TARGET --limit N --workers N --audit-workers N \
  --ster-options trl

# 2. Lifecycle datasets: review masked envelopes into immutable splits.
transcript-label-trainer lifecycle-review envelopes.jsonl \
  --split train --output reviewed-train.jsonl --workers N
transcript-label-trainer lifecycle-review held-out.jsonl \
  --split eval --output reviewed-eval.jsonl --workers N

# 3. Lifecycle model: train, audit every held-out decision, gate, publish.
transcript-label-trainer lifecycle-model reviewed-train.jsonl reviewed-eval.jsonl \
  --compute-target TARGET --brama-url <the Brama address the job dials> \
  --ster-options trl \
  --audit-workers N --audit-max-wrong-share F
```

Each step refuses to continue when its gate fails: no reviewed dataset, no
training; no passed audit, no publication. There is no flag that skips a gate.

