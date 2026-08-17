#!/bin/sh
set -eu

work=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
source_root="$work/audit-source"
target_root="$work/audit-target"
revision=e3dd6f65be729c7e75f12b518b28953cb738f66c
/usr/bin/git -C "$source_root" fetch origin "$revision"
/usr/bin/git -C "$source_root" checkout --detach "$revision"
CARGO_TARGET_DIR="$target_root" /root/.cargo/bin/cargo build \
  --manifest-path "$source_root/Cargo.toml" --locked --release
/usr/bin/install -m 755 \
  "$target_root/release/transcript-label-trainer" \
  "$work/cargo-target/release/transcript-label-trainer"
/usr/bin/sha256sum "$work/cargo-target/release/transcript-label-trainer"
