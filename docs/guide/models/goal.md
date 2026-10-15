## Reviewed Jeden goal model

`goal-model` owns the complete small-model pipeline that turns coding-agent
messages into the 3–7 word task goals Jeden displays. It reads messages only
from Transcript Lake's normalized `events` view; raw agent session files are
not an input, so the lake's masking boundary remains intact.

Replace `TARGET` below with the registered Stado GPU target selected for training.

```sh
transcript-label-trainer goal-model \
  --compute-target TARGET \
  --ster-options trl
```

The command generates task and no-task labels with a Brama teacher, whose answer
must carry the `<goal>…</goal>` or `<goal/>` tags the goal system prompt asks for
(an answer without them fails that candidate, quoting it). Every row, the reviewed
OMP titles included, then needs an independent `sensible` choice from
`best-decision-model` on Brama's typed decision endpoint before it enters the
dataset; whether a goal has the 3–7 words Jeden displays is that reviewer's
judgement, stated in the choice it is offered. It holds out reviewed OMP titles and, of the teacher's task rows and its
no-task rows each, the share scikit-learn's
[`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
documents as its test size (0.25, rounded up so each class holds at least one out)
as gold; a dataset without a held-out task row, a held-out no-task row and a
training row is refused with its counts. It submits the JSONL and exact trainer commit to
the named Stado target. The GPU host must have `ster` installed (`stado product
install ster --surface cli`; without it the job stops at `ster: command not found`).
There the reviewed teacher rows become `transcript-label-trainer goal-examples`: each
one the chat Jeden serves, the goal system prompt as its `system` turn,
`<user>message</user>` as its `prompt` and the reviewed `<goal>…</goal>` or `<goal/>`
as its `completion`. `ster tune sft` fits an adapter to `Qwen/Qwen3-4B` at the
revision `BASE_REVISION` pins in `src/stado/manifest.rs`, with exactly the
`--ster-options` given (an empty value is refused before anything is uploaded),
`ster tune merge` folds it in, and llama.cpp's converter exports and quantizes it to
Q4_K_M. Gold rows never enter training.

`--ster-options trl` stands for the documented defaults of Hugging Face's LoRA
fine-tuning stack, written out before anything is uploaded:

| Ster option | value | source |
|---|---|---|
| `--rank`, `--alpha` | 8, 8 | [PEFT `LoraConfig`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) `r=8`, `lora_alpha=8` |
| `--epochs`, `--accumulation`, `--batch-size`, `--seed` | 3, 1, 8, 42 | [Transformers `TrainingArguments`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) `num_train_epochs=3.0`, `gradient_accumulation_steps=1`, `per_device_train_batch_size=8`, `seed=42` |
| `--learning-rate`, `--max-sequence` | 2e-5, 1024 | [TRL `SFTConfig`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py) `learning_rate=2e-5`, `max_length=1024` |

Any other text is passed to Ster as written. The job's output is keyed by the expanded
options, so the preset and the same options written out are the same run.

`transcript-label-trainer goal-evaluate-gguf` then asks that quantized model for every
gold row the way Jeden serves it: `llama-server` on a loopback port the system assigns,
sized by llama-server itself from the model and the device — its documented defaults
`--parallel -1` (auto slots), `--ctx-size 0` (the model's context, fitted to device
memory by `--fit on`) and `--gpu-layers auto`
([llama-server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)),
so the job states none of them — the same system prompt and user turn with thinking off,
sampled as Jeden Desktop runs the model: temperature 0.7, top-p 0.8, top-k 20 and min-p 0,
the setting the [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md)
states for non-thinking mode (the card warns against greedy decoding), constrained to
`<goal/>` or one `<goal>…</goal>` line. It writes `predictions.jsonl` and the exact-match
share in `metrics-gguf.json`; a server that exits before it is ready fails with its
status and log, and a failed request fails the run naming its session. Every one of
those served predictions must receive `both-sensible` from a final Brama `--best` audit,
a typed choice among `both-sensible`, `label-nonsensical`, `student-nonsensical` and
`both-nonsensical` asked of `best-decision-model` (`goal-audit --brama-model` takes
`decision-model` or `best-decision-model`; any other alias is refused before the
predictions are read), or the model remains unqualified; a rejected candidate is retained for diagnosis but
cannot enter the Jeden Desktop release namespace. An audit record of the same review
over the same predictions is resumed; a record that is not JSON is refused rather than
overwritten. To build an example set and see how
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
being the dataset, expanded `--ster-options` and exact trainer source revision together,
so another source revision or settings cannot resume or overwrite that model.

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
`final-judge.json` itself. The release Jeden Desktop consumes is the one whose
SHA-256 its goal distiller pins; that digest, not a copy on this page, names it.
