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

use crate::jobs::{Job, SKLEARN_MODEL};
use crate::util::{Error, Result};
use crate::{lake, model};

mod execute_goal_model;
mod manifest;
mod release;
mod repository;

pub use execute_goal_model::*;
pub(crate) use manifest::{write_manifest, ManifestModel};
pub use release::*;
pub use repository::*;

/// The work directory, shell-quoted, that a job named `name` keeps its inputs,
/// scratch and outputs in on `compute_target`: `<root>/jobs/<name>`, where
/// `<root>` is the training root that target declares as
/// `targets.<target>.training.models_dir` in the Stado registry. The path is
/// the target's own, never one resolved on the machine that submits. A target
/// the registry does not answer for, or one declaring a root that is not an
/// absolute path, is refused naming the registry path and what it said.
pub(crate) fn job_work_dir(stado: &Path, compute_target: &str, name: &str) -> Result<String> {
    let path = format!("targets.{compute_target}.training.models_dir");
    let args = [
        OsString::from("registry"),
        OsString::from("pull"),
        OsString::from("--path"),
        OsString::from(&path),
    ];
    let answer = run(stado, &args)?;
    if !answer.status.success() {
        return Err(command_error(stado, &format!("reading {path}"), &answer));
    }
    let root = String::from_utf8(answer.stdout)
        .map_err(|error| Error(format!("the registry's {path} is not UTF-8: {error}")))?;
    let root = root.trim();
    if !Path::new(root).is_absolute() {
        return Err(Error(format!(
            "the registry's {path} is {root:?}, not an absolute training root on {compute_target}"
        )));
    }
    Ok(repository::shell_quote(&format!("{root}/jobs/{name}")))
}
