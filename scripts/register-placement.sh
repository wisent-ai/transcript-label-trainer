#!/usr/bin/env bash
#
# Declare this trainer's placement in the canonical Stado registry.
#
# Stado, not this repository, owns the answer to "where does label-model
# training run" and "where does the lake keep its data". This script writes
# those two declarations and nothing else:
#
#   targets[<this machine>].transcript_lake.root  — the lake data root
#   targets[<training host>].training             — enabled, kinds, models_dir
#
# It is idempotent: it pulls the canonical document, merges the declarations
# into it, and pushes only when the merge actually changed something. Running
# it twice leaves the registry byte-identical, and it never deletes a key some
# other publisher added — the whole pulled document is written back.
#
# Override the training placement with TRAINING_HOST / TRAINING_ROOT, and the
# lake root with LAKE_DATA.

set -euo pipefail

TRAINING_HOST="${TRAINING_HOST:-ubuntu-server-rtx-pro-6000}"
TRAINING_ROOT="${TRAINING_ROOT:-/mnt/wd16tb/stado/training}"
TRAINING_KIND="label-model"
LAKE_ROOT="${LAKE_DATA:-$HOME/.transcript-lake}"

command -v stado >/dev/null 2>&1 || {
  echo "register-placement: the 'stado' CLI is not on PATH" >&2
  exit 1
}

# Which registry target this machine is. `stado registry self` prints
# "<name>\t<kind>\t<hostname>" for the local box.
LAKE_HOST="$(stado registry self | head -n 1 | cut -f 1)"
[ -n "$LAKE_HOST" ] || {
  echo "register-placement: 'stado registry self' did not name this machine" >&2
  exit 1
}

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
current="$workdir/current.json"
merged="$workdir/merged.json"

stado registry pull >"$current"

# The merge signals "already declared" with 0 and "changed" with 3, so its
# non-zero exit must not trip `set -e`.
set +e
TRAINING_HOST="$TRAINING_HOST" TRAINING_ROOT="$TRAINING_ROOT" \
TRAINING_KIND="$TRAINING_KIND" LAKE_HOST="$LAKE_HOST" LAKE_ROOT="$LAKE_ROOT" \
python3 - "$current" "$merged" <<'PY'
import json
import os
import sys

current_path, merged_path = sys.argv[1], sys.argv[2]
training_host = os.environ["TRAINING_HOST"]
training_root = os.environ["TRAINING_ROOT"]
training_kind = os.environ["TRAINING_KIND"]
lake_host = os.environ["LAKE_HOST"]
lake_root = os.environ["LAKE_ROOT"]

with open(current_path, encoding="utf-8") as handle:
    registry = json.load(handle)

targets = registry.get("targets")
if not isinstance(targets, list):
    sys.exit("register-placement: the canonical registry carries no 'targets' list")
by_name = {t.get("name"): t for t in targets if isinstance(t, dict)}

for name in (lake_host, training_host):
    if name not in by_name:
        sys.exit(
            f"register-placement: {name!r} is not a registered Stado target; "
            "onboard it with 'stado registry host add' first"
        )

# Merge into whatever the target already carries, so an unrelated key inside
# these blocks survives a rerun.
lake_block = by_name[lake_host].setdefault("transcript_lake", {})
lake_block["root"] = lake_root

training_block = by_name[training_host].setdefault("training", {})
training_block["enabled"] = True
training_block["models_dir"] = training_root
kinds = training_block.get("kinds")
if not isinstance(kinds, list):
    kinds = []
if training_kind not in kinds:
    kinds.append(training_kind)
training_block["kinds"] = sorted(kinds)

with open(merged_path, "w", encoding="utf-8") as handle:
    json.dump(registry, handle, indent=2, sort_keys=True)
    handle.write("\n")

with open(current_path, encoding="utf-8") as handle:
    unchanged = json.load(handle) == registry
sys.exit(0 if unchanged else 3)
PY
status=$?
set -e

if [ "$status" -eq 0 ]; then
  echo "register-placement: registry already declares this placement, nothing to push"
elif [ "$status" -eq 3 ]; then
  stado registry validate "$merged"
  stado registry push "$merged"
  echo "register-placement: pushed the placement declarations"
else
  exit "$status"
fi

after="$workdir/after.json"
stado registry pull >"$after"

echo
echo "canonical registry now declares:"
python3 - "$after" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    registry = json.load(handle)
for target in registry.get("targets", []):
    name = target.get("name")
    for key in ("transcript_lake", "training"):
        block = target.get(key)
        if block:
            print("  targets[%s].%s = %s" % (name, key, json.dumps(block, sort_keys=True)))
PY
