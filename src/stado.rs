//! Submit a declarative trainer run to one canonical Stado compute target.
//!
//! The submitter exports only the job's selected labels and transcript text,
//! stores that read-only dataset plus the job spec through `stado://datasets`,
//! and pins a source checkout at one exact commit to the requested target.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::jobs::{Job, SKLEARN_MODEL};
use crate::util::{home_dir, Error, Result};
use crate::{lake, model};

const REPOSITORY: &str = "https://github.com/wisent-ai/transcript-label-trainer.git";
const REPO_WORKDIR: &str = "transcript-label-trainer";
const SIGNING_SECRET: &str = "WISENT_APP_AGENT_AUTH_SECRET=agent:wisent-app#value";
const BEARER_SECRET: &str = "BRAMA_TOKEN=jeden-model-router#token";

struct TempDir(PathBuf);

impl TempDir {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| Error(format!("system clock is before Unix epoch: {error}")))?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "transcript-label-trainer-stado-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn stado_bin() -> PathBuf {
    if let Some(path) = std::env::var_os("STADO_BIN").filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    let installed = home_dir().join(".stado/bin/stado");
    if installed.is_file() {
        return installed;
    }
    PathBuf::from("stado")
}

fn run(program: &Path, args: &[OsString]) -> Result<Output> {
    Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| Error(format!("could not run {}: {error}", program.display())))
}

fn command_error(program: &Path, action: &str, output: &Output) -> Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    Error(format!(
        "{action} through {} failed: {detail}",
        program.display()
    ))
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn repo_ref() -> Result<String> {
    let candidate = std::env::var("TLT_REPO_REF").unwrap_or_default();
    let candidate = if candidate.trim().is_empty() {
        let output = Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
            .stdin(Stdio::null())
            .output()
            .map_err(|error| Error(format!("could not resolve trainer source commit: {error}")))?;
        if !output.status.success() {
            return Err(command_error(
                Path::new("git"),
                "resolving trainer source commit",
                &output,
            ));
        }
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        candidate.trim().to_string()
    };
    if candidate.len() != 40
        || !candidate
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(Error(format!(
            "trainer source commit must be one full lowercase SHA, got {candidate:?}; set TLT_REPO_REF"
        )));
    }
    Ok(candidate)
}

fn upload(stado: &Path, uri: &str, path: &Path, content_type: &str) -> Result<()> {
    let args = [
        OsString::from("storage"),
        OsString::from("put"),
        OsString::from("--content-type"),
        OsString::from(content_type),
        OsString::from(uri),
        path.as_os_str().to_owned(),
    ];
    let output = run(stado, &args)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error(stado, &format!("uploading {uri}"), &output))
    }
}

fn cargo_command(job: &Job) -> &'static str {
    if job.model == SKLEARN_MODEL {
        "cargo run --locked --release --"
    } else {
        "cargo run --locked --release --features hf --"
    }
}

fn remote_command(job: &Job, dataset_uri: &str, spec_uri: &str, key: &str) -> String {
    let cargo = cargo_command(job);
    let evaluate = if job.eval_split.enabled && job.judge.enabled {
        format!("; {cargo} evaluate '{}' --best", job.name)
    } else {
        String::new()
    };
    format!(
        "set -euo pipefail; work=\"${{TMPDIR:-/tmp}}/tlt-{key}\"; mkdir -p \"$work\"; \
         stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{dataset_uri}' \"$work/dataset.json\"; \
         \"$stado\" storage get '{spec_uri}' \"$work/job.yaml\"; \
         export TLT_DATASET_BUNDLE=\"$work/dataset.json\"; \
         {cargo} run \"$work/job.yaml\"{evaluate}"
    )
}

fn job_id(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("Job ID: ").map(str::trim))
}

/// Submit, follow, and return the terminal Stado outcome for one trainer job.
pub fn execute(job_path: &str, job: &Job, compute_target: &str) -> Result<i32> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }

    let resolved = model::resolve_job(job)?;
    let temporary = TempDir::create()?;
    let dataset_path = temporary.0.join("dataset.json");
    lake::export_bundle(&job.scope.aspect, &resolved.labels, &dataset_path)?;
    let spec_path = temporary.0.join("job.yaml");
    std::fs::copy(job_path, &spec_path)?;

    let dataset_bytes = std::fs::read(&dataset_path)?;
    let spec_bytes = std::fs::read(&spec_path)?;
    let key = digest(&[dataset_bytes.as_slice(), spec_bytes.as_slice()].concat());
    let base = format!("stado://datasets/transcript-label-trainer/{key}");
    let dataset_uri = format!("{base}/dataset.json");
    let spec_uri = format!("{base}/job.yaml");
    let stado = stado_bin();
    upload(&stado, &dataset_uri, &dataset_path, "application/json")?;
    upload(&stado, &spec_uri, &spec_path, "application/yaml")?;

    let source_ref = repo_ref()?;
    let command = remote_command(job, &dataset_uri, &spec_uri, &key[..16]);
    let mut args = vec![
        OsString::from("submit"),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
        OsString::from("--repo"),
        OsString::from(REPOSITORY),
        OsString::from("--repo-ref"),
        OsString::from(source_ref),
        OsString::from("--repo-workdir"),
        OsString::from(REPO_WORKDIR),
        OsString::from("--repo-extras"),
        OsString::new(),
    ];
    if job.judge.enabled {
        args.extend([
            OsString::from("--secret-env"),
            OsString::from(SIGNING_SECRET),
            OsString::from("--secret-env"),
            OsString::from(BEARER_SECRET),
        ]);
    }
    args.push(OsString::from(command));

    let submitted = run(&stado, &args)?;
    io::stdout().write_all(&submitted.stdout)?;
    io::stderr().write_all(&submitted.stderr)?;
    if !submitted.status.success() {
        return Err(command_error(&stado, "submitting trainer job", &submitted));
    }
    let stdout = String::from_utf8_lossy(&submitted.stdout);
    let id = job_id(&stdout)
        .ok_or_else(|| Error("Stado accepted the job but did not report its job id".to_string()))?;
    let status = Command::new(&stado)
        .args(["job", "watch", id, "--follow"])
        .status()
        .map_err(|error| Error(format!("could not follow Stado job {id}: {error}")))?;
    Ok(status.code().unwrap_or(1))
}

pub struct GoalModelJob {
    pub job_id: String,
    pub output_uri: String,
    pub status: i32,
}

/// Submit the reviewed goal dataset to one exclusive Stado GPU target.
pub fn execute_goal_model(dataset_path: &Path, compute_target: &str) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let dataset_bytes = std::fs::read(dataset_path)?;
    let key = digest(&dataset_bytes);
    let dataset_uri = format!("stado://datasets/transcript-label-trainer/goal-model/{key}.jsonl");
    let output_uri = format!("stado://models/jeden/goal-qwen3-0.6b/{key}");
    let stado = stado_bin();
    upload(&stado, &dataset_uri, dataset_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let command = format!(
        "set -euo pipefail; work=\"${{TMPDIR:-/tmp}}/jeden-goal-{key}\"; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{dataset_uri}' \"$work/reviewed-goals.jsonl\"; \
         ./training/goal-model/run.sh \"$work/reviewed-goals.jsonl\""
    );
    let args = vec![
        OsString::from("submit"),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
        OsString::from("--priority"),
        OsString::from("20"),
        OsString::from("--exclusive"),
        OsString::from("--repo"),
        OsString::from(REPOSITORY),
        OsString::from("--repo-ref"),
        OsString::from(source_ref),
        OsString::from("--repo-workdir"),
        OsString::from(REPO_WORKDIR),
        OsString::from("--repo-extras"),
        OsString::new(),
        OsString::from("--output-uri"),
        OsString::from(&output_uri),
        OsString::from("--secret-env"),
        OsString::from(SIGNING_SECRET),
        OsString::from("--secret-env"),
        OsString::from(BEARER_SECRET),
        OsString::from(command),
    ];
    let submitted = run(&stado, &args)?;
    io::stdout().write_all(&submitted.stdout)?;
    io::stderr().write_all(&submitted.stderr)?;
    if !submitted.status.success() {
        return Err(command_error(
            &stado,
            "submitting goal-model job",
            &submitted,
        ));
    }
    let stdout = String::from_utf8_lossy(&submitted.stdout);
    let id = job_id(&stdout)
        .ok_or_else(|| Error("Stado accepted the goal-model job but reported no id".to_string()))?
        .to_string();
    let status = Command::new(&stado)
        .args(["job", "watch", &id, "--follow"])
        .status()
        .map_err(|error| Error(format!("could not follow Stado job {id}: {error}")))?;
    Ok(GoalModelJob {
        job_id: id,
        output_uri,
        status: status.code().unwrap_or(1),
    })
}
