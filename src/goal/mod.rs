//! Privacy-masked goal-model data preparation and independent semantic audits.
//!
//! Messages come only from Transcript Lake's normalized `events` view. Raw
//! agent session files are deliberately outside this boundary: the lake owns
//! masking, while this module owns teacher/reviewer provenance and model gates.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::brama::{BramaClient, Message, BEST_MODEL, DEFAULT_MODEL};
use crate::util::{now_iso, Error, Result};
use crate::{bail, brama, lake};

mod audit_prompt;
mod messages;
mod served;
mod system_prompt;

pub use audit_prompt::*;
pub use messages::*;
pub(crate) use served::{evaluate_gguf, export_examples, GoalEvaluation};
pub use system_prompt::*;

/// What a panicked worker said, in its own words when it gave any.
pub(crate) fn panic_words(panic: &(dyn std::any::Any + Send)) -> String {
    match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
        (Some(words), _) => (*words).to_string(),
        (None, Some(words)) => words.clone(),
        (None, None) => "a panic that carried no message".to_string(),
    }
}
