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
| `gui [--bind IP] [--port PORT]` | serve the embedded graphical corpus importer and navigable HTML documentation on a loopback listener; prints but does not open its session-token URL |

Aspect classifiers (local, cheap, sklearn or HF):

| command | does |
|---|---|
| `train` | train a classifier for one aspect from manual lake labels |
| `run` | execute a declarative training job (YAML spec) |
| `evaluate` | score a trained model on its frozen holdout and have a Brama teacher judge whether the predictions are acceptable |
| `infer` | emit label suggestions for unlabeled sessions; never writes to the lake |
| `info` | list trained aspects, artifacts, and metrics |
| `autolabel` | label every unlabeled session for an aspect via a Brama teacher (zero-touch) |
| `aspect-discover` | propose new aspect dimensions from recent sessions via a Brama teacher (writes nothing) |

Goal models (fine-tunes trained on a Stado GPU target, gated before publish):

| command | does |
|---|---|
| `goal-model` | curate masked lake messages, teacher-label task goals through Brama, require an independent `best` review, train on the named exclusive Stado GPU target, and publish GGUF artifacts only after the held-out gold predictions pass a second `best` audit |
| `goal-audit` | independently audit student goal predictions (JSONL of message, reference goal, student output) and write the complete audit record |
| `lifecycle-review` | classify masked Oko training envelopes through a named Brama route, enforce the `oko-goal-lifecycle-v1` contract, and write ordered JSONL with reviewer provenance for an immutable `--split train\|eval` |
| `lifecycle-model` | upload immutable reviewed train and held-out datasets, fine-tune on the named Stado GPU target, audit every held-out decision through Brama `best`, and publish the candidate only when the lifecycle quality gate passes |
| `lifecycle-audit` | judge every held-out student decision independently, reject inferred completion, retain the full verdict record, and fail the gate when more than two percent are semantically wrong |
| `humanizer-model` | require an explicit `HUMANIZER_HF_REPO`, export masked likely-authored user turns, derive inverse style-transfer inputs through Brama, train a LoRA adapter on the pinned base, and publish only a qualified private adapter revision |
| `humanizer-prepare` | turn the exported targets into session-separated `train.jsonl`, `validation.jsonl` and `test.jsonl` plus `preparation.json`: a Brama teacher (`--teacher-model`, default `codex/gpt-5.6-sol`) rewrites each target as generic AI prose, rewrites that drop a URL, e-mail address or number or fall outside 0.65–2.5× the target's length are refused as `source_contract`, and an independent review (`--review-model`, default `best`) must call the pair usable. Refuses with exit 1 and writes nothing when any split has fewer than 700 / 70 / 70 accepted rows; the message names the counts and the rejection reasons. The Stado job runs it before training |
| `humanizer-audit` | judge every held-out case of `predictions.jsonl` (`id`, `source`, `target`, `base`, `student`) through `--brama-model` (default `best`), write the complete record to `--output`, and exit 1 when the adapter misses the gate (student semantic fidelity ≥ 0.95, voice match ≥ 0.80, pass rate ≥ 0.90, boilerplate ≤ 0.08, voice gain over base ≥ 0.15, semantic loss ≤ 0.02). A case the judge cannot answer in four attempts fails the audit as `audit incomplete` and writes nothing |
| `humanizer-publish MODEL` | require `--repo OWNER/REPOSITORY`, `--metrics`, `--audit`, `--preparation` and `--output`; refuse public destinations before reading model files; publish on a separate branch and verify the private immutable revision and required file checksums; write a JSON receipt and exit 1 on refusal or failure |
| `release-publish` | move a qualified job output (`SOURCE` Stado URI, `--model goal\|lifecycle`) into its release namespace under the artifact's SHA-256 after verifying the manifest, the passed final judge, every file checksum and the reassembled parts; see [models.md](models.md) for the refusals |

Exit statuses across the CLI: `0` success, `1` failed command, `2` usage
error or a run the lake does not hold enough labeled data for.

### Goals quickstart

The goal path in three commands. Replace `TARGET` with a registered Stado GPU target:

```sh
# 1. Title model: curate, teacher-label, review, train, audit, publish GGUF.
transcript-label-trainer goal-model --compute-target TARGET

# 2. Lifecycle datasets: review masked envelopes into immutable splits.
transcript-label-trainer lifecycle-review envelopes.jsonl \
  --split train --output reviewed-train.jsonl
transcript-label-trainer lifecycle-review held-out.jsonl \
  --split eval --output reviewed-eval.jsonl

# 3. Lifecycle model: train, audit every held-out decision, gate, publish.
transcript-label-trainer lifecycle-model reviewed-train.jsonl reviewed-eval.jsonl \
  --compute-target TARGET --brama-url https://brama.wisent.com
```

Each step refuses to continue when its gate fails: no reviewed dataset, no
training; no passed audit, no publication. There is no flag that skips a gate.

