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
after the other, each sized by llama-server itself as for the goal model, each answer
running until the model ends it and sampled with llama-server's own defaults, because
Echo asks the humanizer without a sampling setting of its own. Each answer is scored by chrF
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
CLI is the release `training/humanizer-model/run.sh` installs; `HF_BIN` selects
its executable and `HF_ENDPOINT` the Hub base URL, each taking the value
huggingface_hub documents when unset; a set but empty `HF_BIN` is refused.
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
