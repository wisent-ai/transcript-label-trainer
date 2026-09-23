use super::*;

/// Submit the reviewed goal dataset to one exclusive Stado GPU target.
pub fn execute_goal_model(dataset_path: &Path, compute_target: &str) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let dataset_bytes = std::fs::read(dataset_path)?;
    let key = digest(&dataset_bytes);
    let dataset_uri =
        format!("stado://probierz/inputs/transcript-label-trainer/goal-model/{key}.jsonl");
    let output_uri = format!("stado://probierz/artifacts/models/jeden/goal-qwen3-4b/{key}");
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
        OsString::from("--gpu-type"),
        OsString::from("nvidia-rtx-pro-6000"),
        OsString::from("--vram-gb"),
        OsString::from("48"),
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

/// Submit reviewed Oko lifecycle splits to one exclusive Stado GPU target.
pub fn execute_lifecycle_model(
    train_path: &Path,
    eval_path: &Path,
    compute_target: &str,
    brama_url: &str,
) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let brama_url = brama_url.trim();
    if brama_url.is_empty() {
        return Err(Error("--brama-url cannot be empty".to_string()));
    }
    let brama_url = shell_quote(brama_url);
    let train_bytes = std::fs::read(train_path)?;
    let eval_bytes = std::fs::read(eval_path)?;
    let key = digest(
        format!(
            "train={}\neval={}\n",
            digest(&train_bytes),
            digest(&eval_bytes)
        )
        .as_bytes(),
    );
    let base = format!("stado://probierz/inputs/transcript-label-trainer/lifecycle-model/{key}");
    let train_uri = format!("{base}/reviewed-train.jsonl");
    let eval_uri = format!("{base}/reviewed-eval.jsonl");
    let output_uri = format!("stado://probierz/artifacts/models/oko/lifecycle-qwen3-4b/{key}");
    let stado = stado_bin();
    upload(&stado, &train_uri, train_path, "application/x-ndjson")?;
    upload(&stado, &eval_uri, eval_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let command = format!(
        "set -euo pipefail; work=\"${{TMPDIR:-/tmp}}/oko-lifecycle-{key}\"; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{train_uri}' \"$work/reviewed-train.jsonl\"; \
         \"$stado\" storage get '{eval_uri}' \"$work/reviewed-eval.jsonl\"; \
         export BRAMA_URL={brama_url}; \
         ./training/lifecycle-model/run.sh \
         \"$work/reviewed-train.jsonl\" \"$work/reviewed-eval.jsonl\""
    );
    let args = vec![
        OsString::from("submit"),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
        OsString::from("--priority"),
        OsString::from("20"),
        OsString::from("--gpu-type"),
        OsString::from("nvidia-rtx-pro-6000"),
        OsString::from("--vram-gb"),
        OsString::from("48"),
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
            "submitting lifecycle-model job",
            &submitted,
        ));
    }
    let stdout = String::from_utf8_lossy(&submitted.stdout);
    let id = job_id(&stdout)
        .ok_or_else(|| {
            Error("Stado accepted the lifecycle-model job but reported no id".to_string())
        })?
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

/// Submit the masked personal-voice corpus to one exclusive Stado GPU target.
pub fn execute_humanizer_model(targets_path: &Path, compute_target: &str) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let targets_bytes = std::fs::read(targets_path)?;
    let key = digest(&targets_bytes);
    let targets_uri = format!(
        "stado://probierz/inputs/transcript-label-trainer/humanizer-model/{key}/targets.jsonl"
    );
    let output_uri =
        format!("stado://probierz/artifacts/models/echo/lukasz-humanizer-cydonia-24b-lora/{key}");
    let stado = stado_bin();
    upload(&stado, &targets_uri, targets_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let work_root = crate::placement::resolve_placement()
        .training_root
        .join("humanizer-model")
        .join("jobs");
    let work_root = shell_quote(&work_root.to_string_lossy());
    let command = format!(
        "set -euo pipefail; work={work_root}/echo-humanizer-{key}; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{targets_uri}' \"$work/targets.jsonl\"; \
         HUMANIZER_WORK_DIR=\"$work\" ./training/humanizer-model/run.sh \"$work/targets.jsonl\""
    );
    let args = vec![
        OsString::from("submit"),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
        OsString::from("--priority"),
        OsString::from("20"),
        OsString::from("--gpu-type"),
        OsString::from("nvidia-rtx-pro-6000"),
        OsString::from("--vram-gb"),
        OsString::from("80"),
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
        OsString::from("--secret-env"),
        OsString::from(HUGGINGFACE_SECRET),
        OsString::from(command),
    ];
    let submitted = run(&stado, &args)?;
    io::stdout().write_all(&submitted.stdout)?;
    io::stderr().write_all(&submitted.stderr)?;
    if !submitted.status.success() {
        return Err(command_error(
            &stado,
            "submitting humanizer-model job",
            &submitted,
        ));
    }
    let stdout = String::from_utf8_lossy(&submitted.stdout);
    let id = job_id(&stdout).ok_or_else(|| {
        Error("Stado accepted the humanizer-model job but reported no id".to_string())
    })?;
    let status = Command::new(&stado)
        .args(["job", "watch", id, "--follow"])
        .status()
        .map_err(|error| Error(format!("could not follow Stado job {id}: {error}")))?;
    Ok(GoalModelJob {
        job_id: id.to_string(),
        output_uri,
        status: status.code().unwrap_or(1),
    })
}
