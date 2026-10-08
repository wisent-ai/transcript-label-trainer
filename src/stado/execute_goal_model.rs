use super::*;

/// Submit the reviewed goal dataset to one exclusive Stado GPU target.
/// `audit_workers` is the caller's count of parallel Brama calls for the job's
/// final `goal-audit` and `ster_options` the `ster tune sft` settings it
/// trains with; its quantized evaluation lets llama-server size itself.
pub fn execute_goal_model(
    dataset_path: &Path,
    compute_target: &str,
    audit_workers: usize,
    ster_options: &str,
) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let ster_options = stated_ster_options(ster_options)?;
    let dataset_bytes = std::fs::read(dataset_path)?;
    let key = digest(&dataset_bytes);
    let dataset_uri =
        format!("stado://probierz/inputs/transcript-label-trainer/goal-model/{key}.jsonl");
    let run_key = digest(format!("{key}\nster={ster_options}\n").as_bytes());
    let output_uri = format!("stado://probierz/artifacts/models/jeden/goal-qwen3-4b/{run_key}");
    let stado = stado_bin();
    upload(&stado, &dataset_uri, dataset_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let command = format!(
        "set -euo pipefail; work=\"${{TMPDIR:-/tmp}}/jeden-goal-{run_key}\"; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{dataset_uri}' \"$work/reviewed-goals.jsonl\"; \
         export GOAL_MODEL_WORK_DIR=\"$work\"; export GOAL_STER_OPTIONS={}; \
         GOAL_AUDIT_WORKERS={audit_workers} ./training/goal-model/run.sh \"$work/reviewed-goals.jsonl\"",
        shell_quote(ster_options),
    );
    // The same data, settings and source revision is the same run: Stado
    // answers a repeated submission with the job it already holds.
    let run_id = format!("jeden-goal-{run_key}-{source_ref}");
    let args = vec![
        OsString::from("submit"),
        OsString::from("--run-id"),
        OsString::from(run_id),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
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
        signing_secret()?,
        OsString::from("--secret-env"),
        bearer_secret()?,
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

/// The `ster tune sft` settings a model job trains with, refused when empty:
/// the job states every setting it trains with and assumes none.
pub(crate) fn stated_ster_options(ster_options: &str) -> Result<&str> {
    let ster_options = ster_options.trim();
    if ster_options.is_empty() {
        return Err(Error("--ster-options cannot be empty: the job trains with ster tune sft, which states every setting it trains with".to_string()));
    }
    Ok(ster_options)
}

/// The final audit the lifecycle job runs: its Brama concurrency and the
/// largest share of held-out decisions it may call wrong.
pub struct LifecycleAudit {
    pub workers: usize,
    pub max_wrong_share: f64,
}

/// Submit reviewed Oko lifecycle splits to one exclusive Stado GPU target.
/// `ster_options` are the `ster tune sft` settings the job trains with.
pub fn execute_lifecycle_model(
    train_path: &Path,
    eval_path: &Path,
    compute_target: &str,
    brama_url: &str,
    ster_options: &str,
    audit: &LifecycleAudit,
) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let brama_url = brama_url.trim();
    if brama_url.is_empty() {
        return Err(Error("--brama-url cannot be empty".to_string()));
    }
    let ster_options = stated_ster_options(ster_options)?;
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
    // One model is the data and the settings it was trained with: the work
    // directory a rerun resumes from and the output it publishes to are keyed
    // by both, so other settings never resume or overwrite this model.
    let run_key = digest(format!("{key}\nster={ster_options}\n").as_bytes());
    let ster_options = shell_quote(ster_options);
    let output_uri = format!("stado://probierz/artifacts/models/oko/lifecycle-qwen3-4b/{run_key}");
    let stado = stado_bin();
    upload(&stado, &train_uri, train_path, "application/x-ndjson")?;
    upload(&stado, &eval_uri, eval_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let command = format!(
        "set -euo pipefail; work=\"${{TMPDIR:-/tmp}}/oko-lifecycle-{run_key}\"; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{train_uri}' \"$work/reviewed-train.jsonl\"; \
         \"$stado\" storage get '{eval_uri}' \"$work/reviewed-eval.jsonl\"; \
         export BRAMA_URL={brama_url}; export LIFECYCLE_STER_OPTIONS={ster_options}; \
         export LIFECYCLE_AUDIT_WORKERS={}; export LIFECYCLE_AUDIT_MAX_WRONG_SHARE={}; \
         ./training/lifecycle-model/run.sh \
         \"$work/reviewed-train.jsonl\" \"$work/reviewed-eval.jsonl\"",
        audit.workers,
        audit.max_wrong_share,
    );
    let run_id = format!("oko-lifecycle-{run_key}-{source_ref}");
    let args = vec![
        OsString::from("submit"),
        OsString::from("--run-id"),
        OsString::from(run_id),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
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
        signing_secret()?,
        OsString::from("--secret-env"),
        bearer_secret()?,
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

/// How the humanizer job trains its adapter: the `ster tune sft` settings.
/// Its evaluation lets llama-server size itself and each answer run until the
/// model ends it, and scores by chrF with sacrebleu's documented settings.
pub struct HumanizerTraining {
    pub ster_options: String,
}

/// Submit the masked personal-voice corpus to one exclusive Stado GPU target.
/// `workers` and `attempts` are the caller's counts for the job's Brama
/// preparation and audit, `gate` the quality gate its audit holds the
/// adapter to, `minimums` the split minimums and source length bounds of
/// its preparation and `training` how it trains and measures the adapter;
/// the job is refused without them.
#[allow(clippy::too_many_arguments)]
pub fn execute_humanizer_model(
    targets_path: &Path,
    compute_target: &str,
    hf_repo: &str,
    workers: usize,
    attempts: usize,
    gate: &crate::humanizer::AuditGate,
    minimums: &crate::humanizer::PreparationBounds,
    training: &HumanizerTraining,
) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let hf_repo = hf_repo.trim();
    if hf_repo.is_empty() {
        return Err(Error("HUMANIZER_HF_REPO cannot be empty".to_string()));
    }
    let targets_bytes = std::fs::read(targets_path)?;
    let key = digest(&targets_bytes);
    let targets_uri = format!(
        "stado://probierz/inputs/transcript-label-trainer/humanizer-model/{key}/targets.jsonl"
    );
    let ster_options = stated_ster_options(&training.ster_options)?;
    let run_key = digest(format!("{key}\nster={ster_options}\n").as_bytes());
    let output_uri =
        format!("stado://probierz/artifacts/models/echo/lukasz-humanizer-cydonia-24b-lora/{run_key}");
    let stado = stado_bin();
    upload(&stado, &targets_uri, targets_path, "application/x-ndjson")?;
    let source_ref = repo_ref()?;
    let work_root = crate::placement::resolve_placement()
        .training_root
        .join("humanizer-model")
        .join("jobs");
    let work_root = shell_quote(&work_root.to_string_lossy());
    let hf_repo = shell_quote(hf_repo);
    let command = format!(
        "set -euo pipefail; work={work_root}/echo-humanizer-{run_key}; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:-$HOME/.stado/bin/stado}}\"; \
         \"$stado\" storage get '{targets_uri}' \"$work/targets.jsonl\"; \
         HUMANIZER_HF_REPO={hf_repo} HUMANIZER_WORK_DIR=\"$work\" \
         HUMANIZER_WORKERS={workers} HUMANIZER_ATTEMPTS={attempts} \
         HUMANIZER_MIN_SEMANTIC_FIDELITY={} HUMANIZER_MIN_VOICE_MATCH={} \
         HUMANIZER_MIN_PASS_RATE={} HUMANIZER_MAX_BOILERPLATE_RATE={} \
         HUMANIZER_MIN_VOICE_GAIN={} HUMANIZER_MIN_SEMANTIC_DELTA={} \
         HUMANIZER_MIN_TRAIN_ROWS={} HUMANIZER_MIN_VALIDATION_ROWS={} HUMANIZER_MIN_TEST_ROWS={} \
         HUMANIZER_MIN_LENGTH_RATIO={} HUMANIZER_MAX_LENGTH_RATIO={} \
         HUMANIZER_TEST_SHARE={} HUMANIZER_VALIDATION_SHARE={} \
         HUMANIZER_STER_OPTIONS={} \
         ./training/humanizer-model/run.sh \"$work/targets.jsonl\"",
        gate.min_semantic_fidelity,
        gate.min_voice_match,
        gate.min_pass_rate,
        gate.max_boilerplate_rate,
        gate.min_voice_gain,
        gate.min_semantic_delta,
        minimums.train,
        minimums.validation,
        minimums.test,
        minimums.length.min,
        minimums.length.max,
        minimums.held_out.test,
        minimums.held_out.validation,
        shell_quote(ster_options),
    );
    let run_id = format!("echo-humanizer-{run_key}-{source_ref}");
    let args = vec![
        OsString::from("submit"),
        OsString::from("--run-id"),
        OsString::from(run_id),
        OsString::from("--pinned-host"),
        OsString::from(compute_target),
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
        signing_secret()?,
        OsString::from("--secret-env"),
        bearer_secret()?,
        OsString::from("--secret-env"),
        secret_env("HF_TOKEN", "TLT_HF_TOKEN_ROLE", "the Hugging Face token")?,
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
