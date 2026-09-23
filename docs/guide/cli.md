<!-- Moved out of README.md; the README links here. -->
## CLI

One binary, seventeen subcommands. Global flags: `--training-root PATH` (where
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
| `humanizer-model` | export masked likely-authored user turns, derive inverse style-transfer inputs through Brama, train a LoRA adapter on the pinned base, and publish only a qualified private adapter revision |

Exit statuses across the CLI: `0` success, `1` failed command, `2` usage
error or a run the lake does not hold enough labeled data for.

### Goals quickstart

The goal path in three commands, assuming a registered Stado GPU target:

```sh
# 1. Title model: curate, teacher-label, review, train, audit, publish GGUF.
transcript-label-trainer goal-model --compute-target ubuntu-server-rtx-pro-6000

# 2. Lifecycle datasets: review masked envelopes into immutable splits.
transcript-label-trainer lifecycle-review envelopes.jsonl \
  --split train --output reviewed-train.jsonl
transcript-label-trainer lifecycle-review held-out.jsonl \
  --split eval --output reviewed-eval.jsonl

# 3. Lifecycle model: train, audit every held-out decision, gate, publish.
transcript-label-trainer lifecycle-model reviewed-train.jsonl reviewed-eval.jsonl \
  --compute-target ubuntu-server-rtx-pro-6000 --brama-url https://brama.wisent.com
```

Each step refuses to continue when its gate fails: no reviewed dataset, no
training; no passed audit, no publication. There is no flag that skips a gate.

