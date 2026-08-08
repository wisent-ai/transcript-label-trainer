# Working here

- **Never write to the lake.** The label store at `$LAKE_DATA/labels/` is
  read-only in this repository. Applying suggestions happens through
  `transcript-lake label add`, which the lake owns.
- **Do not vendor or copy lake code.** Reach lake data only through the lake
  CLI (`query --json` over its views).
- **Artifacts live outside the repo.** Models go to `$TLT_HOME/models/`
  (default `~/.transcript-label-trainer/models/`), never into the working
  tree.
- **No tests.** Standing operator policy forbids creating or running tests.
  Verification is running the CLI against the real (or a scratch `LAKE_DATA`)
  lake and reading the output.
- Keep dependencies minimal: scikit-learn and its transitives in the base
  install. torch and transformers belong only to the optional `hf` extra;
  import them lazily so the base install never needs them. CPU is the
  baseline, MPS is picked up automatically on Apple silicon.
