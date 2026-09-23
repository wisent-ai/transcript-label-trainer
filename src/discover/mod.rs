//! Aspect discovery — a Brama teacher reads recent masked sessions and
//! proposes aspect dimensions grounded in what the user asked for and how the
//! agent answered.
//!
//! This writes nothing. The lake's labeler owns the aspect vocabulary, so the
//! output is a proposal document: each proposal names the aspect, its allowed
//! values, why it matters, which sampled sessions ground it, and the exact
//! `autolabel`/`train` commands that would turn it into a model. With
//! `--best`, Brama's independent `best` route must call a proposal sensible
//! before it is kept; the audit failing is an exit-status matter, not a
//! silently shorter list.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::util::Result;
use crate::{brama, lake};

mod chunk_sessions;
mod discover_items;

pub use chunk_sessions::*;
pub use discover_items::*;
