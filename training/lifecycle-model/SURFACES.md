# The trainer's numbers are not the shipped model's numbers

Measured 2026-08-18 on candidate `abd65dca6468` (full fine-tune of
`Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`), one
held-out split of 485 rows, greedy decoding on both sides:

The gap turned out to be the GGUF path itself. Serving the same fine-tune
through `mlx_lm` on the same machine and the same 485-row split, greedy, no
grammar constraint:

| metric | trainer, HF bf16 on CUDA | served, Q4_K_M GGUF on Metal | served, MLX bf16 on Metal | gate |
|---|---|---|---|---|
| valid_json | 1.0000 | 1.0000 | 1.0000 | ≥ 0.99 |
| action_accuracy | 0.9381 | 0.8495 | 0.9423 | ≥ 0.90 |
| goal_ref_accuracy | — | 0.9485 | 0.9794 | — |
| evidence_accuracy | 0.9897 | 0.9072 | 0.9938 | — |
| joint_accuracy | 0.9237 | 0.7381 | 0.9258 | ≥ 0.88 |
| finish_precision | 1.0000 | 0.9787 | 1.0000 | = 1.0 |

MLX beats even the trainer's own numbers, which is what a lossless path should
do: the same weights, the same arithmetic family, no conversion. The qualified
release therefore ships MLX weights and declares its runtime, and
`oko/scripts/install-goal-lifecycle-model.py` reads that declaration instead of
assuming llama-server.

Two smaller facts worth keeping. Constrained decoding is a safety net rather
than the thing that produces valid output: the MLX path emitted 485 parseable
decisions with no grammar at all. And more training made the GGUF surface worse
while the trainer's own numbers improved — five epochs scored 0.9381 for the
trainer and 0.8495 served, against 0.9258 and 0.9340 at three epochs — so a
run tuned against the trainer's metric can be tuned away from production.

| metric | trainer, HF bf16 on CUDA | served, Q4_K_M on Metal | served, Q8_0 on Metal |
|---|---|---|---|
| valid_json | 1.0000 | 1.0000 | 1.0000 |
| action_accuracy | 0.9567 | 0.8990 | 0.8928 |
| goal_ref_accuracy | 0.9918 | 0.9608 | 0.9381 |
| evidence_accuracy | 0.9505 | 0.8598 | 0.8907 |
| joint_accuracy | 0.9072 | 0.7670 | 0.7732 |
| finish_precision | 1.0000 | 0.8889 | 0.9231 |

The gap is not quantization, and it is not the schema constraint. What was
checked, and what each check ruled out:

- **Quantization.** `Q8_0` is no better than `Q4_K_M` (joint 0.7732 vs 0.7670),
  and the unquantized `bf16` GGUF scores the same as both on a 14-row startGoal
  probe (8/14, against 12/14 from the trainer's own generation). Precision is
  not what is lost.
- **The grammar.** Serving the same rows with and without `response_format`
  gives the identical count (5/14 on an earlier probe). Constrained decoding
  shapes the output, it does not move the decision.
- **The sampler.** `top_k=1, top_p=1, min_p=0, repeat_penalty=1` reproduces the
  server default result exactly (5/14 both ways).
- **The prompt.** `llama-server /apply-template` and the HF tokenizer render
  byte-identical text, including the empty `<think>` block that
  `enable_thinking: false` inserts.
- **The tokenizer.** `/tokenize` and the HF tokenizer return identical id
  sequences (1510 tokens) for the same prompt.
- **The weights.** The student ties its output projection
  (`tie_word_embeddings: true`, no `lm_head.weight`), and the GGUF matches:
  398 tensors, `token_embd.weight` present, no `output.weight`. Nothing is
  dropped in conversion.
- **The context.** Server logs report `truncated = 0` on every request, prompts
  of 1150–1500 tokens against a 4096-token slot.

What remains is the inference implementation itself: the trainer measures HF
kernels on CUDA, production runs llama.cpp kernels on Metal. On a task whose
classes are separated by small logit margins, that difference moves roughly one
decision in ten.

**Consequence for the gate.** `run.sh` qualifies on `metrics-gguf.json`, the
served surface, and never on `metrics.json`. A release whose manifest carries
the trainer's numbers is describing a model nobody serves. When Oko will run the
model on a Mac, the qualifying measurement belongs on Metal too — the same
backend, the same quantization, the same loopback contract.
