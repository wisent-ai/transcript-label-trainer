//! Privacy-masked goal-model data preparation and independent semantic audits.
//!
//! Messages come only from Transcript Lake's normalized `events` view. Raw
//! agent session files are deliberately outside this boundary: the lake owns
//! masking, while this module owns teacher/reviewer provenance and model gates.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::brama::{BramaClient, Message, BEST_MODEL, DEFAULT_MODEL};
use crate::util::{now_iso, Error, Result};
use crate::{bail, lake};

mod system_prompt;
mod messages;
mod audit_prompt;

pub use system_prompt::*;
pub use messages::*;
pub use audit_prompt::*;
