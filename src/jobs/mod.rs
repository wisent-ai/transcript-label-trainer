//! Declarative training jobs: load and validate a YAML job spec.
//!
//! A job spec captures the four things an operator declares per training run:
//! WHO evaluated the transcripts (evaluator — the exact label-store source that
//! counts as ground truth), WHICH model to train (model), the SCOPE of training
//! data (scope), and the TASK (free text stored with the artifacts).
//!
//! Two more sections govern how the run is judged. `eval_split` is required:
//! the spec states the holdout's fraction and seed (or `false`), and that
//! holdout is frozen out of training; `judge` has a Brama-routed teacher rule
//! on whether the trained model's holdout predictions are acceptable, on
//! unless the spec says `judge: false`. A HuggingFace model also states its
//! `training` settings.
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
