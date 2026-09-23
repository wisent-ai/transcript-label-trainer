//! Read-only access to Transcript Lake state.
//!
//! Two contracts are consumed here, neither is reimplemented:
//!
//! - the append-only label store at `<storage root>/labels/*.ndjson`, owned by
//!   `transcript-lake label` — this module only ever reads it;
//! - the canonical `events`/`sessions` DuckDB views, reached by shelling out
//!   to the lake CLI (`query --json`) so the SQL setup in sql/views.sql stays
//!   the lake's own code.
//!
//! The one write path is [`label_add`], and it is a write the lake performs:
//! `autolabel` hands the lake CLI a `label add`, and the lake validates the
//! session and appends the record. Nothing here opens the label store for
//! writing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bail;
use crate::placement::resolve_placement;
use crate::util::{home_dir, json_text, json_truthy, Error, Result};

mod text_cap;
mod session_texts;

pub use text_cap::*;
pub use session_texts::*;
