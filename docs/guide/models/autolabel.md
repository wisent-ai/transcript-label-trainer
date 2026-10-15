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
the session text and asks Brama's typed decision endpoint (`POST
/v1/decisions`) to choose one of the allowed values. The answer is the label
Brama returns in the decision's structure; no reply text is searched for a
value. With `--best`, a second choice — `sensible` or `nonsensical` — is asked
of `best-decision-model`; only a `sensible` proposal is applied through the
lake's own CLI:

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
- **Failure isolation.** A Brama refusal, a decision without a choice, or a
  choice outside the allowed values fails that one session with Brama's own
  words, writes nothing for it, and is counted in the final summary
  (`labeled` / `skipped_labeled` / `failed`).
- The decision endpoint answers exactly two aliases: `decision-model`, the
  route the gateway pays for, and `best-decision-model`, which delegates to
  `best`, the strongest operator subscription the signed identity may use.
  The teacher defaults to `best`, which asks `best-decision-model`; name
  either decision alias with `--brama-model`. Any other alias is refused
  before a session is read, naming the two that can answer, and an empty
  `--brama-model` is refused by name. The summary records the alias given
  (`brama_model`) and the one asked (`decision_model`). No sampling setting
  is sent: how the routed model decides is its own.
- Auth: HMAC-signed requests as `WISENT_APP_AGENT_ID` (required), keyed by
  `WISENT_APP_AGENT_AUTH_SECRET` or else the vault role `TLT_BRAMA_AGENT_ROLE`
  declares as `ROLE#FIELD`; the bearer is `BRAMA_TOKEN` or else the role
  `TLT_BRAMA_TOKEN_ROLE` declares. Roles are read with `stado credentials get
  --role`, so no vault item is named and replacing one changes nothing here;
  an unset or malformed variable is refused with its name, and so is one set
  but empty. The endpoint is `BRAMA_URL` when it is set, otherwise where
  Stado's service directory places Brama; when Stado cannot place it, the
  refusal carries Stado's reason. Secrets are read into memory only, never
  printed. A Stado job receives the same roles as `--secret-env
  ENV=ROLE#FIELD`, plus `TLT_HF_TOKEN_ROLE` for the humanizer's Hugging Face
  token.

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
