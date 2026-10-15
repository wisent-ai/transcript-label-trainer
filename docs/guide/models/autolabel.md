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
the session text and asks the teacher for exactly one of the allowed values.
With `--best`, a second model returns `sensible` or `nonsensical`; only a
`sensible` proposal is applied through the lake's own CLI:

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
- **Failure isolation.** A Brama error or an unparseable answer fails that
  one session, writes nothing for it, and is counted in the final summary
  (`labeled` / `skipped_labeled` / `failed`).
- The teacher defaults to Brama's `best` alias, the strongest operator
  subscription the signed identity may use; override it with `--brama-model`.
  Brama resolves the alias and reports unavailable capabilities. No sampling
  setting is sent: how the routed model samples is its own.
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
