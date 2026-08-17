#!/bin/sh
set -eu

revision=f6be63e200e5f463ba41b0cf7087846664977c5e
candidate=/mnt/wd16tb/wisent-staging/oko-lifecycle-model-b5de55bd
source_root="$candidate/audit-source"
target_root="$candidate/audit-target"
if [ ! -d "$source_root/.git" ]; then
  /usr/bin/git clone https://github.com/wisent-ai/transcript-label-trainer.git "$source_root"
fi
/usr/bin/git -C "$source_root" fetch origin "$revision"
/usr/bin/git -C "$source_root" checkout --detach "$revision"
CARGO_TARGET_DIR="$target_root" /root/.cargo/bin/cargo build \
  --manifest-path "$source_root/Cargo.toml" --locked --release
binary="$candidate/cargo-target/release/transcript-label-trainer"
[ -e "$binary.before-concurrency-fix" ] || /bin/cp -p "$binary" "$binary.before-concurrency-fix"
/usr/bin/install -m 755 "$target_root/release/transcript-label-trainer" "$binary"
/usr/bin/sha256sum "$binary"
