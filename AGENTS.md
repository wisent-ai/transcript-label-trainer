# Working here

- **Never write to the lake directly.** The label store under the resolved
  storage root (`<storage root>/labels/`) is never opened for writing from
  this repository. The one designed write path is `autolabel`, which applies
  teacher labels THROUGH the lake CLI (`transcript-lake label add`), which
  validates and owns the write — operator-mandated zero-touch. Everything
  else (`infer` output) stays staged for a human to apply the same way.
- **Do not vendor or copy lake code.** Reach lake data only through the lake
  CLI (`query --json` over its views).
- **Artifacts live outside the repo.** Models go to `<training root>/models/`,
  never into the working tree. Both roots come from `placement.rs`, which
  resolves flag > env > the Stado registry > a visible local fallback — never
  read `TLT_HOME` or `LAKE_DATA` anywhere else.
- **No tests.** Standing operator policy forbids creating or running tests.
  Verification is running the CLI against the real (or a scratch `LAKE_DATA`)
  lake and reading the output.
- Keep dependencies minimal: serde, serde_yaml, reqwest and rand in the
  default build, and no ML framework at all. candle and tokenizers belong
  only to the optional `hf` cargo feature; gate them behind
  `#[cfg(feature = "hf")]` so the default build never compiles them. CPU is
  the baseline; candle's `metal` feature is on for macOS only, so Apple
  silicon fine-tunes on the GPU automatically.
- **All LLM inference goes through Brama.** Never call a provider directly
  and never print credential values — secrets are read into memory only.
