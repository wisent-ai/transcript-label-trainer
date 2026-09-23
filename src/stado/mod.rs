//! Submit a declarative trainer run to one canonical Stado compute target.
//!
//! The submitter exports only the job's selected labels and transcript text,
//! stores that read-only dataset in Probierz's input boundary, and pins a
//! source checkout at one exact commit to the requested target.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::jobs::{Job, SKLEARN_MODEL};
use crate::util::{home_dir, Error, Result};
use crate::{lake, model};

mod repository;
mod execute_goal_model;

pub use repository::*;
pub use execute_goal_model::*;
