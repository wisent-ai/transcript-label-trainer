//! Where training runs and where lake data lives — resolved from Stado.
//!
//! Stado owns the canonical compute-target registry, and that registry is the
//! authority on placement:
//!
//! - `targets[<this machine>].transcript_lake.root` — the lake data root, i.e.
//!   the storage root this trainer reads labels and session text out of;
//! - `targets[<host>].training` — the host that trains label models, with
//!   `models_dir` as the artifact root on that host.
//!
//! Resolution order, strongest first:
//!
//! 1. an explicit CLI flag (`--training-root` / `--storage-root`),
//! 2. the environment (`TLT_HOME` / `LAKE_DATA`),
//! 3. the Stado registry declarations above,
//! 4. a local fallback under `~`.
//!
//! The fallback is an exception, not a default. Anything that stops Stado from
//! answering — the binary absent, the registry unreachable, no declaration for
//! this machine, training placed on a host that is not this one — degrades to
//! the local path and writes the reason into [`Placement::detail`], which
//! `info` prints. Resolution never fails: a broken control plane must not stop
//! a local run, it must only stop being invisible.
//!
//! [`Placement::source`] reports the *weakest* layer any root depended on, so
//! one silent local fallback cannot hide behind a root that did resolve.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::util::{home_dir, json_text, json_truthy};

mod targets_key;
mod index_by_name;

pub use targets_key::*;
pub use index_by_name::*;
