#!/bin/sh
set -eu
work=/mnt/wd16tb/stado/jobs/jeden-goal-prompt-v5
repo="$work/audit-source"
commit=5dde942e5a6cecd338dbf110a7703291933d0778
rm -rf "$repo"
git init -q "$repo"
git -C "$repo" remote add origin https://github.com/wisent-ai/transcript-label-trainer.git
git -C "$repo" fetch --depth 1 origin "$commit"
git -C "$repo" checkout -q FETCH_HEAD
CARGO_TARGET_DIR="$work/audit-target" /root/.cargo/bin/cargo build --manifest-path "$repo/Cargo.toml" --locked --release --bin transcript-label-trainer
cp "$work/audit-target/release/transcript-label-trainer" /root/.stado/files/transcript-label-trainer-audit
chmod 700 /root/.stado/files/transcript-label-trainer-audit
/root/.stado/files/transcript-label-trainer-audit --version
