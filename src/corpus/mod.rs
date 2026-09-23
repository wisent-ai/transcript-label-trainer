//! Adoption of the trainer's existing canonical dataset-bundle format.
//!
//! Transcript Lake still owns live labels. An adopted bundle is an immutable,
//! self-contained training input under the trainer placement root; training
//! reads the selected bundle through the same `lake::load_labels` boundary used
//! by pinned Stado jobs.
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::placement::resolve_placement;
use crate::util::{now_iso, Error, Result};

mod bundle_schema;
mod read_registry;

pub use bundle_schema::*;
pub use read_registry::*;
