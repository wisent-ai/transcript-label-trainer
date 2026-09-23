<!-- Moved out of README.md; the README links here. -->
## Quick start

```sh
cargo install --path .
```

`cargo install` places `transcript-label-trainer` in `~/.cargo/bin`, which must
be on `PATH`. To build without installing, run `cargo build --release` and
invoke `target/release/transcript-label-trainer` directly. Building needs a
Rust toolchain at version `1.85` or newer; nothing else.

### Graphical corpus importer

Launch the product-owned browser workspace:

```sh
transcript-label-trainer gui
```

The command binds an available port on `127.0.0.1`, prints a session-token URL
and the documentation URL, and waits; it does not open a browser. Copy the
printed **Corpus importer** URL into a local browser, select a real schema-v1
dataset-bundle JSON, and import. The frontend is compiled into the installed
binary and calls the same atomic Rust adoption engine as `corpus-adopt`. It
shows the resolved training and storage roots, all imported/unchanged/
conflicting/rejected counts, the stable source identity, and the registry state
read back from disk.

Choose roots before the command and optionally pin the listener:

```sh
transcript-label-trainer \
  --training-root /srv/wisent/trainer \
  --storage-root /srv/wisent/lake \
  gui --bind 127.0.0.1 --port 8765
```

Only loopback IPs are accepted (`127.0.0.1` by default; `::1` is supported);
port `0` selects an available port. Requests require the exact listener host,
API calls require the per-session token, and mutation also requires the exact
origin. Uploads are capped at 16 MiB before JSON parsing. Importing never starts
training, evaluation, inference, teacher/judge work, or a fleet job. Open the
served **Documentation** navigation or read
[`docs/corpus-import.html`](../../docs/corpus-import.html for the exact schema,
selection rules, identities, flags, refusals, and diagnostic meanings.

`transcript-label-trainer onboarding` walks the first-use journey this
repository ships in `onboarding_first_use.json`. Its first action can adopt a
real existing corpus through the same operation as `corpus-adopt`:

```sh
transcript-label-trainer onboarding --corpus /path/to/dataset-bundle.json
```

The source is the schema-v1 `dataset-bundle` JSON written by this trainer's
Transcript Lake/Stado export boundary: top-level `schema_version`, `aspect`, and
`labels`; each label has `session_id`, `value`, `source`, RFC 3339 `ts`,
nullable `runtime`, and nonempty reconstructed `text`. Unknown fields, malformed
timestamps, empty required values, and duplicate native session IDs reject the
whole file before corpus state changes. No unsupported row is dropped.

Validated content is retained at
`<training root>/corpora/<sha256>.json`; the selected native content identity
and every older corpus remain in `<training root>/corpora/registry.json`.
Identical content is idempotent and reports every row unchanged. A different
valid corpus is preserved beside earlier data and becomes selected. Use
`--skip-corpus` to keep an empty usable trainer and adopt later.

Progress is recorded per operator per machine under
`~/.local/state/transcript-label-trainer/onboarding.json`, outside the training
root and outside the lake; `--reset` replays the journey. The journey continues
through training to real label suggestions, which are recorded by `infer` only
when emitted.

Train one aspect from the manual labels in the lake:

```sh
transcript-label-trainer train --aspect reviewed
```

With too few labeled sessions this fails cleanly, stating the minimum and the
actual count — that is correct behavior, not a crash. The minimum is 8 labeled
sessions across at least 2 distinct values *on the training side*: a fifth of
the labels is frozen out of training by default, so in practice about 10
labeled sessions get you started. `--no-eval-split` trains on all of them.

Emit suggestions for sessions that have no label on that aspect yet:

```sh
transcript-label-trainer infer --aspect reviewed --limit 20
```

Each suggestion is a label-store-shaped record with `source="model"`:

```json
{
  "ts": "2026-08-08T05:00:00Z",
  "session_id": "abc123",
  "runtime": "claude",
  "aspect": "reviewed",
  "value": "yes",
  "note": "confidence=0.83",
  "source": "model"
}
```

Suggestions are printed to stdout and nothing is written to the lake. To apply
them, review and feed the accepted ones to the lake's labeler:

```sh
transcript-label-trainer infer --aspect reviewed --limit 20 > suggestions.json
# review, then apply each accepted record:
transcript-lake label add <session-id> --aspect reviewed --value <value> --source model --note "confidence=0.83"
```

Inspect trained aspects, artifact paths, and metrics:

```sh
transcript-label-trainer info
```

