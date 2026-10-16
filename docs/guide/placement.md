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

The submitter first admits the exact trainer source revision. Only an absent
`TLT_REPO_REF` selects checkout HEAD; a present blank, non-Unicode or malformed
value refuses with the variable's name. Generic runs refuse before dataset
export or remote uploads. The submitter then resolves
the job against the local Transcript Lake, exports only
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

Every Stado command the trainer runs, here and for releases, lifecycle and
goal jobs, uses the program `TLT_STADO_BIN` names, or `stado` on `PATH` when it
is unset; a set but empty or non-UTF-8 value is refused with the variable's
name. Inside a job on the training host, the script fetches its inputs with
the Stado that runs the job, which Stado exports to every job as `STADO_BIN`;
a job that has none stops before fetching anything, its shell reporting
`STADO_BIN: Stado exports STADO_BIN to every job it runs, and this job has none`,
rather than guessing a Stado under `$HOME`.

Every remote job - `run --compute-target`, `goal-model`, `lifecycle-model`
and `humanizer-model` - works in `<models_dir>/jobs/<run>` on its target,
where `<models_dir>` is the training root that target declares as
`targets.<target>.training.models_dir` and the trainer reads it from the Stado
registry before submitting. The path is the target's own, never one resolved
on the machine that submits; a target the registry does not answer for, or
one declaring a root that is not an absolute path, is refused before anything
is submitted, naming the registry path and what it said. `<run>` is keyed by
the full submitted run identity: dataset, job specification or settings, and
exact trainer source revision. Repeating the same run selects the same work
directory; another source revision selects a different one. This does not
promise filesystem retention after a host is retired.

### Qualify generic remote submission and replay

`cargo test --test run-dispatch` executes real `run --compute-target` jobs.
Run it on the dedicated training host with the worker's staging filesystem
visible at the same path. Declare `STADO_BIN` (the Stado the test itself runs;
the trainer it starts reads `TLT_STADO_BIN`), `TLT_TEST_COMPUTE_TARGET`,
`TLT_TEST_SOURCE_REVISION` matching the clean checkout, and a different existing
`TLT_TEST_ALTERNATE_SOURCE_REVISION`; both revisions must be reachable by Stado.
`TLT_TEST_JOB_SPEC` names a reviewed `tfidf-logreg` job whose evaluator and scope
select real Transcript Lake labels. The runner changes only its copy's name to
a unique qualification name, leaving training settings and model gates intact.
The ordinary lake and Brama credential declarations still apply.

`TLT_TEST_WORKER_TRAINING_ROOT` must match the target's declared
`training.models_dir`; the journey reads that registry field before submitting
and reads each job's staged inputs from `jobs/tlt-<run-id>` under it.
Unavailable worker files fail the journey rather than reducing it to an argv
check.

Malformed, empty and whitespace-only source declarations must refuse before a
job ID or trained model appears.
Successful runs must have terminal persisted Stado jobs, byte-identical staged
specifications, actual trained model files and whole-session metrics. Identical
replay must return the same job without changing staged data or model, metrics
and specification checksums. Another source must use a different job and actual
staging directory, preserve the first staged inputs, and train from the same
dataset. Model output still uses the target's declared training root; this test
does not claim source-isolated model output for generic jobs.

Commands, exit statuses, binary/source identities, model checksums and copies of
metrics and persisted specifications remain in `.build/run-dispatch-<run>/`.
After successful terminal runs the journey removes only its uniquely named
model. A failure retains that model path in the report instead of deleting
files a possibly running job still owns. Remote jobs and their staged inputs
remain as evidence. This covers generic dispatch, not lifecycle publication or
the graphical training screen.

### Qualify lifecycle submission and source-isolated outputs

`cargo test --test lifecycle-dispatch` runs real `lifecycle-model` jobs, not a
dispatch simulation. It needs `STADO_BIN`, `TLT_TEST_SOURCE_REVISION` matching
the clean checkout, another existing full `TLT_TEST_ALTERNATE_SOURCE_REVISION`,
and readable reviewed `TLT_TEST_TRAIN_DATASET` / `TLT_TEST_EVAL_DATASET` files.
Declare `TLT_TEST_COMPUTE_TARGET`, target-reachable `TLT_TEST_BRAMA_URL` and
`TLT_TEST_STER_OPTIONS`; the normal product credential declarations still apply.
Both source revisions must be available to the real Stado worker.

The journey refuses malformed, empty and whitespace-only sources before
submission, then trains and qualifies the actual lifecycle model.
Repeating identical data, settings and
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
   `~` is `$HOME`; the trainer refuses to start when `HOME` is unset or
   empty rather than placing these under the current directory.

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

