//! Declarative training jobs: load and validate a YAML job spec.
//!
//! A job spec captures the four things an operator declares per training run:
//! WHO evaluated the transcripts (evaluator — the exact label-store source that
//! counts as ground truth), WHICH model to train (model), the SCOPE of training
//! data (scope), and the TASK (free text stored with the artifacts).
//!
//! Two more sections govern how the run is judged, and both are ON unless the
//! spec turns them off: `eval_split` freezes a holdout of labeled sessions that
//! training never sees, and `judge` has a Brama-routed teacher rule on whether
//! the trained model's holdout predictions are acceptable.
//!
//! Every field is validated here; invalid specs fail with clear errors and no
//! silent defaults.
use std::path::Path;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, SecondsFormat};
use serde::{Deserialize, Serialize};
use serde_yaml::Value as Yaml;

use crate::bail;
use crate::brama;
use crate::util::{float_repr, Result};

mod source_pattern;
mod eval_split;

pub use source_pattern::*;
pub use eval_split::*;
