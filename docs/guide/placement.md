<!-- Moved out of README.md; the README links here. -->
## Placement: Stado decides where this runs

Stado owns the canonical compute-target registry, and that registry — not this
repository and not an environment variable — is the authority on where label
models are trained and where the lake keeps its data. Two declarations carry
it, both per registry target:

| Key | Meaning |
|---|---|
| `targets[<this machine>].transcript_lake.root` | the **storage root**: the lake data root labels and session text are read out of |
| `targets[<host>].training` | `{enabled, kinds, models_dir}` — the host that trains, and the **training root** for model artifacts on it. This trainer claims the kind `label-model`. |

Register both through the checked-in script, never by hand:

```sh
./scripts/register-placement.sh
```

It pulls the canonical document, merges the two declarations into it, and
pushes only if the merge changed something — so a second run leaves the
registry byte-identical, and no key another publisher added is ever dropped.
`TRAINING_HOST`, `TRAINING_ROOT` and `LAKE_DATA` override what it declares;
the machine it declares the lake root for is whatever `stado registry self`
says this box is.

### Execute on one named compute target

`run --compute-target` turns the local command into a Stado job pinned to one
canonical registry target:

```sh
transcript-label-trainer run jobs/example-topic.yaml \
  --compute-target <training-target>
```

The submitter resolves the job against the local Transcript Lake, exports only
the selected labels and their whole transcript text, and uploads that
read-only, content-addressed bundle plus the validated YAML through Probierz's
`inputs/transcript-label-trainer/` object boundary. Stado then clones this
repository at one exact commit, pins the job with `--pinned-host`, injects the
Brama signing and bearer references through `--secret-env`, and streams
`stado job watch --follow`
until the target reports a terminal state. The remote command trains under the
target's declared `training.models_dir`; when the default split and judge are
enabled, it immediately runs `evaluate <name> --best`, so nonsensical labels
or final judge opinions fail the Stado job rather than becoming a successful
artifact.

The input-staging workspace is keyed by the full submitted run identity:
dataset, job specification and exact trainer source revision. Repeating the
same run selects the same staging path; another source revision selects a
different one. This does not move model artifacts away from the target's
declared training root or promise filesystem retention after a host is retired.

### Qualify lifecycle submission and source-isolated outputs

`cargo test --test lifecycle-dispatch` runs real `lifecycle-model` jobs, not a
dispatch simulation. It needs `STADO_BIN`, `TLT_TEST_SOURCE_REVISION` matching
the clean checkout, another existing full `TLT_TEST_ALTERNATE_SOURCE_REVISION`,
and readable reviewed `TLT_TEST_TRAIN_DATASET` / `TLT_TEST_EVAL_DATASET` files.
Declare `TLT_TEST_COMPUTE_TARGET`, target-reachable `TLT_TEST_BRAMA_URL` and
`TLT_TEST_STER_OPTIONS`; the normal product credential declarations still apply.
Both source revisions must be available to the real Stado worker.

The journey refuses an invalid source before submission, then trains and
qualifies the actual lifecycle model. Repeating identical data, settings and
source must return the same persisted job and unchanged published manifest.
Another source must have its own job, workspace identity and output URI.
It reads terminal Stado state, retrieves both published model manifests and
checks the downloaded final-judge and training/served-metrics files against
their declared byte counts and checksums. A failed model gate is a failed
journey, not permission to omit that model run.

Commands, exit statuses, output, source revisions, binary checksum and downloaded
evidence remain under `.build/lifecycle-dispatch-<run>/`. Immutable remote jobs
and qualified model artifacts remain available under their recorded identities;
the runner neither deletes shared jobs nor changes its input datasets. This
qualifies lifecycle dispatch/publication, not generic `run` placement or GUI
rendering, and does not prove that a retired host retains staging files.


### Resolution order

Each root is resolved independently, strongest layer first:

1. **flag** — `--training-root` / `--storage-root`, before the subcommand;
2. **env** — `TLT_HOME` / `LAKE_DATA`;
3. **stado** — the declarations above;
4. **local-fallback** — `~/.transcript-label-trainer` and `~/.transcript-lake`.

### The local fallback is an exception, not a default

Falling back is never silent. `info` prints the resolved placement, and
`source` reports the *weakest* layer any root needed, so one root quietly
going local cannot hide behind another that resolved:

```
placement:
    source:        local-fallback
    training host: <training-target>
    training root: <local-home>/.transcript-label-trainer
    storage root:  <local-home>/.transcript-lake
    fallback:      training root … — local fallback because Stado places
                   label-model training on <training-target> at
                   <declared-training-root>, and this machine is
                   <local-target>; storage root … declared in the Stado registry
```

Everything that can stop Stado from answering degrades this way and names
itself in the `fallback` line: the `stado` binary absent from `PATH`, the
registry unreachable, this machine not declaring `transcript_lake.root`, no
host declaring the `label-model` training kind, or — as above — training
placed on a host that is not the one running the command. Resolution never
raises; a control plane that is down must not stop a local run, only stop
being invisible about it.

`info --json` carries the same thing under `placement`, next to `aspects`.

