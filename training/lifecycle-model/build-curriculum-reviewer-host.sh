#!/bin/sh
set -eu

work=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
source_root="$work/audit-source"
target_root="$work/audit-target"
revision=11a04eb07c31fa7bb80c6406c2a121b789f84f6c
/usr/bin/git -C "$source_root" fetch origin "$revision"
/usr/bin/git -C "$source_root" checkout --detach "$revision"
CARGO_TARGET_DIR="$target_root" /root/.cargo/bin/cargo build \
  --manifest-path "$source_root/Cargo.toml" --locked --release
/usr/bin/install -m 755 \
  "$target_root/release/transcript-label-trainer" \
  "$work/cargo-target/release/transcript-label-trainer"
/usr/bin/sha256sum "$work/cargo-target/release/transcript-label-trainer"
