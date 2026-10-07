use super::*;

pub(crate) const REPOSITORY: &str = "https://github.com/wisent-ai/transcript-label-trainer.git";

pub(crate) const REPO_WORKDIR: &str = "transcript-label-trainer";

/// One `--secret-env` value, `ENV=ROLE#FIELD`: the job's environment variable
/// and the vault role and field `declared_by` names, so no vault item is
/// written here and replacing the item changes nothing in the trainer.
pub(crate) fn secret_env(env: &str, declared_by: &str, holds: &str) -> Result<OsString> {
    let (role, field) = crate::brama::role_reference(declared_by, holds)?;
    Ok(OsString::from(format!("{env}={role}#{field}")))
}

/// The job's signing secret, from the role `TLT_BRAMA_AGENT_ROLE` declares.
pub(crate) fn signing_secret() -> Result<OsString> {
    secret_env(
        "WISENT_APP_AGENT_AUTH_SECRET",
        "TLT_BRAMA_AGENT_ROLE",
        "the agent's signing secret",
    )
}

/// The job's Brama bearer, from the role `TLT_BRAMA_TOKEN_ROLE` declares.
pub(crate) fn bearer_secret() -> Result<OsString> {
    secret_env("BRAMA_TOKEN", "TLT_BRAMA_TOKEN_ROLE", "the Brama bearer")
}

/// Scratch space for one Stado exchange, removed when dropped. It lives in the
/// ignored `target/scratch/` of the checkout this binary was built from, never
/// in the system temporary directory, so a run's leftovers stay inside the one
/// checkout and `cargo clean` removes them.
pub(crate) struct TempDir(pub(crate) PathBuf);

impl TempDir {
    pub(crate) fn create() -> Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| Error(format!("system clock is before Unix epoch: {error}")))?
            .as_nanos();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("scratch")
            .join(format!(
                "transcript-label-trainer-stado-{}-{nonce}",
                std::process::id()
            ));
        std::fs::create_dir_all(&path)
            .map_err(|error| Error(format!("cannot create {}: {error}", path.display())))?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn stado_bin() -> PathBuf {
    if let Some(path) = std::env::var_os("STADO_BIN").filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    let installed = home_dir().join(".stado/bin/stado");
    if installed.is_file() {
        return installed;
    }
    PathBuf::from("stado")
}

pub(crate) fn run(program: &Path, args: &[OsString]) -> Result<Output> {
    Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| Error(format!("could not run {}: {error}", program.display())))
}

pub(crate) fn command_error(program: &Path, action: &str, output: &Output) -> Error {
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

pub(crate) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

pub(crate) fn repo_ref() -> Result<String> {
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

pub(crate) fn upload(stado: &Path, uri: &str, path: &Path, content_type: &str) -> Result<()> {
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

pub(crate) fn cargo_command(job: &Job) -> &'static str {
    if job.model == SKLEARN_MODEL {
        "cargo run --locked --release --"
    } else {
        "cargo run --locked --release --features hf --"
    }
}

pub(crate) fn remote_command(job: &Job, dataset_uri: &str, spec_uri: &str, key: &str) -> String {
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

pub(crate) fn job_id(stdout: &str) -> Option<&str> {
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
    let base = format!("stado://probierz/inputs/transcript-label-trainer/{key}");
    let dataset_uri = format!("{base}/dataset.json");
    let spec_uri = format!("{base}/job.yaml");
    let stado = stado_bin();
    upload(&stado, &dataset_uri, &dataset_path, "application/json")?;
    upload(&stado, &spec_uri, &spec_path, "application/yaml")?;

    let source_ref = repo_ref()?;
    let command = remote_command(job, &dataset_uri, &spec_uri, &key[..16]);
    // The same dataset, spec and source revision is the same run: Stado
    // answers a repeated submission with the job it already holds.
    let run_id = format!("tlt-run-{key}-{source_ref}");
    let mut args = vec![
        OsString::from("submit"),
        OsString::from("--run-id"),
        OsString::from(run_id),
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
            signing_secret()?,
            OsString::from("--secret-env"),
            bearer_secret()?,
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
