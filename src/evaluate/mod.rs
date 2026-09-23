//! The frozen evaluation split, and the Brama teacher's verdict on it.
//!
//! Three checks live here, and they answer one question — *does the trained
//! model and the evidence used to assess it make sense?*
//!
//! **The frozen split.** Comparing two models over time only means something
//! when both were scored on the same untouched sessions, so the holdout is
//! decided once and written to `$TLT_HOME/models/<name>/eval-split.json` next
//! to the artifacts. Every later run of the same job reads that file back:
//! sessions labeled since then can only join the training side, a session
//! already in the holdout is never trained on, and the file is never rewritten.
//! It is on by default (`eval_split: false` in the job spec turns it off).
//!
//! **The judge.** Accuracy on the holdout says how often the model matched the
//! ground-truth label; it does not say whether the label it chose was
//! defensible for that transcript. So [`evaluate`] additionally sends each
//! holdout session — its text, the model's prediction, the ground truth — to a
//! Brama-routed teacher and asks for one word. A Brama error fails that one
//! session and is counted, exactly like `autolabel`; if no session could be
//! judged at all, the gateway's own error is surfaced verbatim and nothing is
//! written, because a fabricated verdict is worse than no verdict.
//!
//! **The final review.** With `--best`, Brama's `best` route independently
//! audits both the stored ground-truth label and the first judge's opinion.
//! Any nonsensical or unreviewed record makes the command fail after the
//! evidence has been written to `judge.json`.
//!
//! # The shuffle is deterministic, and it is not Python's
//!
//! The Python original shuffled each class with `random.Random(f"{seed}:{value}")`,
//! whose Mersenne-Twister seeding from a string is a CPython implementation
//! detail that cannot be reproduced here. The scheme used instead, fixed and
//! documented so a first run is reproducible from the seed on any machine and
//! any build:
//!
//! 1. the per-class key is the UTF-8 bytes of `"{seed}:{class}"`, the same
//!    string Python keyed on — so a class that gains labels still does not
//!    reshuffle the others;
//! 2. that string is hashed with SHA-256 and the 32-byte digest becomes the
//!    seed of a ChaCha20 stream (`rand_chacha::ChaCha20Rng::from_seed`);
//! 3. the class members, sorted lexicographically first, are permuted by a
//!    descending Fisher–Yates: for `i` from `len - 1` down to `1`, swap
//!    position `i` with position `next_u64() % (i + 1)`.
//!
//! Fisher–Yates is spelled out here rather than taken from `rand`'s
//! `SliceRandom` so a `rand` upgrade cannot silently pick a different holdout.
//! Splits already on disk are unaffected either way: the file is written once
//! and never rewritten.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::util::{float_repr, json_truthy, now_iso, Error, Result, TrainFailure};
use crate::{brama, jobs, lake, model};

mod split_file;
mod resolve_split;
mod judge_sessions;
mod evaluate_items;

pub use split_file::*;
pub use resolve_split::*;
pub use judge_sessions::*;
pub use evaluate_items::*;
