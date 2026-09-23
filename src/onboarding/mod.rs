//! First-use walkthrough. The journey Echo publishes for this product is the
//! one shipped in `onboarding_first_use.json` at the repository root, compiled
//! into the binary; this command walks that definition rather than a second
//! copy of the same words, so the screens an operator reads here are the
//! screens the control plane holds.
//!
//! The first success is not "the walkthrough asked whether it worked": it is
//! suggestions actually emitted by a trained classifier over real session
//! text, so [`record_first_success`] is called from `model::infer`, where that
//! happens, and this command only reads back what inference recorded.
//!
//! Progress is operator state for this machine, not training state: it lives
//! outside the training root and outside the lake, so `--training-root`,
//! `--storage-root` and a Stado placement on another host never move it.
//! `--reset` discards the recorded attempt and replays the journey from its
//! entry screen in the same invocation.
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::util::{home_dir, now_iso, Error, Result};
use crate::{corpus, lake, model};

mod product_id;
mod report;
mod set;

pub use product_id::*;
pub use report::*;
pub use set::*;
