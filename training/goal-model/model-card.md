---
license: apache-2.0
base_model: Qwen/Qwen3-4B
pipeline_tag: text-generation
tags:
  - gguf
  - qwen3
  - text-generation
  - goal-formulation
  - coding-agents
  - multilingual
language:
  - en
  - pl
model-index:
  - name: Jeden Goal Qwen3 4B
    results:
      - task:
          type: text-generation
          name: Coding-agent goal formulation
        dataset:
          name: Private privacy-masked Transcript Lake goal pairs
          type: private
        metrics:
          - type: exact_match
            value: 3.92156862745098
            name: Exact match
---

# Jeden Goal Qwen3 4B

Jeden Goal Qwen3 4B is a task-specific Qwen3-4B model fine-tuned to turn a coding-agent request into a concise 3–7 word imperative goal. It preserves product names and technical identifiers, follows the user's language, and emits only `<goal>…</goal>` or `<goal/>` when a continuation has no self-contained task.

The repository contains the Q4_K_M GGUF used by [Jeden Desktop](https://github.com/wisent-ai/jeden-desktop). Inference is local through `llama.cpp`; transcript text is not sent to Hugging Face or another inference service. Jeden Desktop downloads the immutable model once, verifies its SHA-256, and caches both the model and generated goals on the Mac.

## Artifact

| Field | Value |
|---|---|
| Base model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) |
| Format | GGUF, Q4_K_M |
| File | `jeden-goal-qwen3-4b-q4_k_m.gguf` |
| Bytes | `2,497,280,320` |
| SHA-256 | `2512d7a455a50a16742b75d8fe38bf02b46b5d6b607f785be32a6345d999d310` |
| Context used by Jeden Desktop | 2,048 tokens |
| Generation | temperature 0, maximum 40 tokens, reasoning disabled |

## Prompt contract

Use the system prompt in [`goal-system-prompt.md`](goal-system-prompt.md), then provide the request as:

```text
<user>the login button is broken on mobile somehow, can you fix?</user>
```

Expected output shape:

```text
<goal>Fix login button on mobile</goal>
```

A context-dependent continuation has no self-contained task:

```text
<user>yes, do that</user>
<goal/>
```

Run locally with a recent `llama-cli`:

```bash
llama-cli \
  --model jeden-goal-qwen3-4b-q4_k_m.gguf \
  --system-prompt-file goal-system-prompt.md \
  --prompt '<user>sprawdź aplikacje desktopowe Brama i Skarbiec</user>' \
  --single-turn --reasoning off --ctx-size 2048 \
  --n-predict 40 --temp 0 --no-display-prompt --simple-io
```

## Training

The model was fine-tuned for 3 epochs at a learning rate of `1e-5` on 564 curated multilingual goal pairs derived from privacy-masked coding-agent sessions. The raw transcripts and training rows are private and are not included in this repository.

The held-out split contains 51 manually curated rows. Training evaluation recorded:

- exact match: `2/51` (`3.92%`);
- final evaluation loss: `0.5859`;
- independent semantic audit: `51/51` outputs judged sensible;
- nonsensical or unparseable outputs: `0`.

Exact match is intentionally strict: semantically equivalent language changes such as `Dodaj CLI i MCP do Weles` versus `Add CLI and MCP to Weles` count as failures. The semantic audit is therefore the release qualification criterion, while exact match remains visible as a diagnostic.

## Limitations

- The model formulates goals; it is not a general assistant and should not be used to answer the request.
- Polish inputs can occasionally produce English goals.
- It can slightly broaden scope, for example changing “identify the crash cause” into “identify and resolve the crash.”
- It only sees the supplied request. Context-dependent continuations should produce `<goal/>` rather than infer missing context.
- The model can reproduce biases or mistakes from Qwen3-4B and the private distillation labels.

## Provenance and license

The qualified GGUF is content-addressed by the SHA-256 above. Training and qualification are owned by [`wisent-ai/transcript-label-trainer`](https://github.com/wisent-ai/transcript-label-trainer); runtime integration is owned by [`wisent-ai/jeden-desktop`](https://github.com/wisent-ai/jeden-desktop).

This derivative follows the Apache 2.0 license of `Qwen/Qwen3-4B`. See [`LICENSE`](LICENSE).
