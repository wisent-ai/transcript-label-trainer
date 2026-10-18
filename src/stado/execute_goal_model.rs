use super::*;

/// Submit the reviewed goal dataset to one exclusive Stado GPU target.
/// `ster_options` are the `ster tune sft` settings it trains with; its
/// quantized evaluation lets llama-server size itself, and its final
/// `goal-audit` sends as many calls at once as Brama measured the route to
/// carry.
pub fn execute_goal_model(
    dataset_path: &Path,
    compute_target: &str,
    ster_options: &str,
) -> Result<GoalModelJob> {
    let compute_target = compute_target.trim();
    if compute_target.is_empty() {
        return Err(Error("--compute-target cannot be empty".to_string()));
    }
    let ster_options = stated_ster_options(ster_options)?;
    let dataset_bytes = crate::util::read_bytes(dataset_path)?;
    let key = crate::util::sha256_hex(&dataset_bytes);
    let dataset_uri =
        format!("stado://probierz/inputs/transcript-label-trainer/goal-model/{key}.jsonl");
    let source_ref = repo_ref()?;
    let run_key =
        crate::util::sha256_hex(format!("{key}\nster={ster_options}\nsource={source_ref}\n"));
    let output_uri = format!("stado://probierz/artifacts/models/jeden/goal-qwen3-4b/{run_key}");
    let stado = PathBuf::from(crate::brama::stado_bin()?);
    upload(&stado, &dataset_uri, dataset_path, "application/x-ndjson")?;
    let work = job_work_dir(&stado, compute_target, &format!("jeden-goal-{run_key}"))?;
    // The script copies its dataset into the work directory, so the job
    // fetches it to inputs/ there rather than onto that copy.
    let command = format!(
        "set -euo pipefail; work={work}; \
         mkdir -p \"$work/inputs\"; stado=\"${{STADO_BIN:?Stado exports STADO_BIN to every job it runs, and this job has none}}\"; \
         \"$stado\" storage get '{dataset_uri}' \"$work/inputs/reviewed-goals.jsonl\"; \
         export GOAL_MODEL_WORK_DIR=\"$work\"; export GOAL_STER_OPTIONS={}; \
         ./training/goal-model/run.sh \"$work/inputs/reviewed-goals.jsonl\"",
        shell_quote(&ster_options),
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
    let status = super::repository::follow(&stado, &id)?;
    Ok(GoalModelJob {
        job_id: id,
        output_uri,
        status,
    })
}

/// Submit reviewed Oko lifecycle splits to one exclusive Stado GPU target.
/// `ster_options` are the `ster tune sft` settings the job trains with; its
/// final audit tolerates no wrong decision and runs at the route's measured
/// allowance.
pub fn execute_lifecycle_model(
    train_path: &Path,
    eval_path: &Path,
    compute_target: &str,
    brama_url: &str,
    ster_options: &str,
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
    let train_bytes = crate::util::read_bytes(train_path)?;
    let eval_bytes = crate::util::read_bytes(eval_path)?;
    let key = crate::util::sha256_hex(format!(
        "train={}\neval={}\n",
        crate::util::sha256_hex(&train_bytes),
        crate::util::sha256_hex(&eval_bytes)
    ));
    let base = format!("stado://probierz/inputs/transcript-label-trainer/lifecycle-model/{key}");
    let train_uri = format!("{base}/reviewed-train.jsonl");
    let eval_uri = format!("{base}/reviewed-eval.jsonl");
    // Source, data and settings jointly identify both the workspace and output.
    let source_ref = repo_ref()?;
    let run_key =
        crate::util::sha256_hex(format!("{key}\nster={ster_options}\nsource={source_ref}\n"));
    let ster_options = shell_quote(&ster_options);
    let output_uri = format!("stado://probierz/artifacts/models/oko/lifecycle-qwen3-4b/{run_key}");
    let stado = PathBuf::from(crate::brama::stado_bin()?);
    upload(&stado, &train_uri, train_path, "application/x-ndjson")?;
    upload(&stado, &eval_uri, eval_path, "application/x-ndjson")?;
    let work = job_work_dir(&stado, compute_target, &format!("oko-lifecycle-{run_key}"))?;
    // As for the goal job, inputs/ keeps the fetched splits apart from the
    // copies the script makes in its work directory.
    let command = format!(
        "set -euo pipefail; work={work}; \
         mkdir -p \"$work/inputs\"; stado=\"${{STADO_BIN:?Stado exports STADO_BIN to every job it runs, and this job has none}}\"; \
         \"$stado\" storage get '{train_uri}' \"$work/inputs/reviewed-train.jsonl\"; \
         \"$stado\" storage get '{eval_uri}' \"$work/inputs/reviewed-eval.jsonl\"; \
         export LIFECYCLE_MODEL_WORK_DIR=\"$work\"; \
         export BRAMA_URL={brama_url}; export LIFECYCLE_STER_OPTIONS={ster_options}; \
         ./training/lifecycle-model/run.sh \
         \"$work/inputs/reviewed-train.jsonl\" \"$work/inputs/reviewed-eval.jsonl\""
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
    let status = super::repository::follow(&stado, &id)?;
    Ok(GoalModelJob {
        job_id: id,
        output_uri,
        status,
    })
}

/// How the humanizer job trains its adapter: the `ster tune sft` settings.
/// Its evaluation lets llama-server size itself and each answer run until the
/// model ends it, and scores by chrF with sacrebleu's documented settings.
pub struct HumanizerTraining {
    pub ster_options: String,
}

/// Submit the masked personal-voice corpus to one exclusive Stado GPU target.
/// `training` is how it trains the adapter; the job's Brama preparation and
/// audit ask each question once, as many at a time as each route's measured
/// allowance. Its audit holds the adapter to the base it was trained from.
pub fn execute_humanizer_model(
    targets_path: &Path,
    compute_target: &str,
    hf_repo: &str,
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
    let targets_bytes = crate::util::read_bytes(targets_path)?;
    let key = crate::util::sha256_hex(&targets_bytes);
    let targets_uri = format!(
        "stado://probierz/inputs/transcript-label-trainer/humanizer-model/{key}/targets.jsonl"
    );
    let ster_options = stated_ster_options(&training.ster_options)?;
    let source_ref = repo_ref()?;
    let run_key = crate::util::sha256_hex(serde_json::to_vec(&(
        &key,
        &ster_options,
        &source_ref,
        hf_repo,
    ))?);
    let output_uri = format!(
        "stado://probierz/artifacts/models/echo/lukasz-humanizer-cydonia-24b-lora/{run_key}"
    );
    let stado = PathBuf::from(crate::brama::stado_bin()?);
    upload(&stado, &targets_uri, targets_path, "application/x-ndjson")?;
    let work = job_work_dir(&stado, compute_target, &format!("echo-humanizer-{run_key}"))?;
    let hf_repo = shell_quote(hf_repo);
    let command = format!(
        "set -euo pipefail; work={work}; \
         mkdir -p \"$work\"; stado=\"${{STADO_BIN:?Stado exports STADO_BIN to every job it runs, and this job has none}}\"; \
         \"$stado\" storage get '{targets_uri}' \"$work/targets.jsonl\"; \
         HUMANIZER_HF_REPO={hf_repo} HUMANIZER_WORK_DIR=\"$work\" \
         HUMANIZER_STER_OPTIONS={} \
         ./training/humanizer-model/run.sh \"$work/targets.jsonl\"",
        shell_quote(&ster_options),
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
    let status = super::repository::follow(&stado, id)?;
    Ok(GoalModelJob {
        job_id: id.to_string(),
        output_uri,
        status,
    })
}
