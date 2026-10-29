//! Real generic training dispatch, replay and staged-input source isolation.
//! The worker's declared training root, which holds every job's work
//! directory under jobs/, must be readable on this qualification host.
mod support;

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use support::*;

fn inspect(
    evidence: &mut Evidence,
    stado: &str,
    stdout: &str,
    source: &str,
    staging: &Path,
    model: &Path,
    spec: &Path,
    name: &str,
    label: &str,
) -> Result<Value> {
    let id = field(stdout, "Job ID: ")?;
    let status = evidence.succeeds(stado, &["machine", "status", id], None)?;
    let status: Value = serde_json::from_str(&status)?;
    let job = &status["job"];
    require(
        job["job_id"] == id && job["terminal"] == true,
        format!("job is not terminal: {status}"),
    )?;
    require(
        job["error"].is_null(),
        format!("remote training failed: {status}"),
    )?;
    let run = job["run_id"].as_str().ok_or("job has no run_id")?;
    require(
        run.starts_with("tlt-run-") && run.ends_with(source),
        format!("job does not retain its source: {run}"),
    )?;
    require(
        !run.contains('/') && !run.contains('\\'),
        "run identity is not one path component",
    )?;
    let workspace = staging.join(format!("tlt-{run}"));
    require(
        fs::read(workspace.join("job.yaml"))? == fs::read(spec)?,
        "worker staged a different job specification",
    )?;
    let dataset = workspace.join("dataset.json");
    let dataset_sha = digest(&dataset)?;
    let persisted: Value = serde_yaml::from_slice(&fs::read(model.join("job.yaml"))?)?;
    require(
        persisted["name"] == name && persisted["model"] == "tfidf-logreg",
        format!("persisted trained job differs: {persisted}"),
    )?;
    let metrics_bytes = fs::read(model.join("metrics.json"))?;
    let metrics: Value = serde_json::from_slice(&metrics_bytes)?;
    require(
        metrics["session_text"] == "whole",
        "trained model is not a whole-session model",
    )?;
    let model_sha = digest(&model.join("model.json"))?;
    fs::write(
        evidence.root.join(format!("{label}-metrics.json")),
        &metrics_bytes,
    )?;
    fs::copy(
        model.join("job.yaml"),
        evidence.root.join(format!("{label}-job.yaml")),
    )?;
    Ok(
        json!({"job": job, "workspace": workspace, "dataset_sha256": dataset_sha,
        "model_sha256": model_sha, "metrics_sha256": digest(&model.join("metrics.json"))?,
        "spec_sha256": digest(&model.join("job.yaml"))?}),
    )
}

fn journey(evidence: &mut Evidence) -> Result<()> {
    let source = required("TLT_TEST_SOURCE_REVISION")?;
    let alternate = required("TLT_TEST_ALTERNATE_SOURCE_REVISION")?;
    require(
        source != alternate,
        "source isolation requires two different real revisions",
    )?;
    require(
        evidence
            .succeeds("git", &["rev-parse", "HEAD"], None)?
            .trim()
            == source,
        "qualification source differs from HEAD",
    )?;
    require(
        evidence
            .succeeds("git", &["status", "--porcelain"], None)?
            .trim()
            .is_empty(),
        "commit source and qualification before running",
    )?;
    for revision in [&source, &alternate] {
        let resolved = evidence.succeeds(
            "git",
            &["rev-parse", &format!("{revision}^{{commit}}")],
            None,
        )?;
        require(
            resolved.trim() == revision,
            "a full existing source revision is required",
        )?;
    }
    let stado = required("STADO_BIN")?;
    let target = required("TLT_TEST_COMPUTE_TARGET")?;
    let training = PathBuf::from(required("TLT_TEST_WORKER_TRAINING_ROOT")?).canonicalize()?;
    let input = PathBuf::from(required("TLT_TEST_JOB_SPEC")?).canonicalize()?;
    let input_sha = digest(&input)?;
    let mut spec: Value = serde_yaml::from_slice(&fs::read(&input)?)?;
    require(
        spec["model"] == "tfidf-logreg",
        "this journey qualifies the real tfidf-logreg backend",
    )?;
    let name = format!("dispatch-qualification-{}", rand::random::<u128>());
    spec["name"] = json!(name);
    let declared = evidence.succeeds(
        &stado,
        &[
            "registry",
            "pull",
            "--path",
            &format!("targets.{target}.training.models_dir"),
        ],
        None,
    )?;
    require(
        Path::new(declared.trim()).canonicalize()? == training,
        "qualification training root differs from the target's Stado declaration",
    )?;
    let spec_path = evidence.root.join("job.yaml");
    fs::write(&spec_path, serde_yaml::to_string(&spec)?)?;
    let model = training.join("models").join(&name);
    require(
        !model.try_exists()?,
        format!("refusing existing model path {}", model.display()),
    )?;
    evidence.report["source_revision"] = json!(source);
    evidence.report["alternate_source_revision"] = json!(alternate);
    evidence.report["binary_sha256"] = json!(digest(Path::new(BINARY))?);
    evidence.report["input_sha256"] = json!(input_sha);
    evidence.report["owned_model_path"] = json!(model);
    let staging = training.join("jobs");
    evidence.report["worker_staging_root"] = json!(staging);
    evidence.save()?;
    let args = [
        "run",
        spec_path.to_str().ok_or("job path is not UTF-8")?,
        "--compute-target",
        &target,
    ];
    for invalid in ["not-a-commit", "", " \t\n"] {
        let (accepted, stdout, stderr) = evidence.run(BINARY, &args, Some(invalid))?;
        require(
            !accepted
                && stderr.contains("TLT_REPO_REF")
                && stderr.contains("trainer source commit"),
            format!("invalid source {invalid:?} did not produce its causal refusal: {stderr}"),
        )?;
        require(
            !stdout.contains("Job ID:") && !model.try_exists()?,
            "refused source started training",
        )?;
    }
    let first = evidence.succeeds(BINARY, &args, Some(&source))?;
    let first = inspect(
        evidence, &stado, &first, &source, &staging, &model, &spec_path, &name, "first",
    )?;
    evidence.report["observations"]["first"] = first.clone();
    evidence.save()?;
    let replay = evidence.succeeds(BINARY, &args, Some(&source))?;
    let replay = inspect(
        evidence, &stado, &replay, &source, &staging, &model, &spec_path, &name, "replay",
    )?;
    require(
        first["job"]["job_id"] == replay["job"]["job_id"],
        "replay submitted a second job",
    )?;
    for key in [
        "workspace",
        "dataset_sha256",
        "model_sha256",
        "metrics_sha256",
        "spec_sha256",
    ] {
        require(first[key] == replay[key], format!("replay changed {key}"))?;
    }
    evidence.report["observations"]["replay"] = replay;
    evidence.save()?;
    let other = evidence.succeeds(BINARY, &args, Some(&alternate))?;
    let other = inspect(
        evidence,
        &stado,
        &other,
        &alternate,
        &staging,
        &model,
        &spec_path,
        &name,
        "alternate",
    )?;
    require(
        first["job"]["job_id"] != other["job"]["job_id"],
        "another source reused the job",
    )?;
    require(
        first["workspace"] != other["workspace"],
        "another source reused staged input paths",
    )?;
    require(
        first["dataset_sha256"] == other["dataset_sha256"],
        "source change used different training data",
    )?;
    let first_workspace = Path::new(
        first["workspace"]
            .as_str()
            .ok_or("workspace is not a path")?,
    );
    require(
        digest(&first_workspace.join("dataset.json"))? == first["dataset_sha256"],
        "another source changed the first staged dataset",
    )?;
    require(
        fs::read(first_workspace.join("job.yaml"))? == fs::read(&spec_path)?,
        "another source changed the first staged specification",
    )?;
    require(
        digest(&input)? == input_sha,
        "qualification rewrote the supplied job",
    )?;
    evidence.report["observations"]["alternate"] = other;
    // Jobs and their staged inputs are retained as execution evidence. Only the
    // uniquely named model created by these now-terminal jobs is removed here.
    fs::remove_dir_all(&model)?;
    require(!model.try_exists()?, "owned model remains after cleanup")?;
    evidence.report["model_removed"] = json!(true);
    Ok(())
}

fn main() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".build")
        .join(format!("run-dispatch-{}", rand::random::<u128>()));
    fs::create_dir_all(root.parent().ok_or("report has no parent")?)?;
    fs::create_dir(&root)?;
    let mut evidence = Evidence {
        root,
        report: json!({"commands": [], "observations": {}, "verdict": "failed"}),
    };
    evidence.save()?;
    let result = journey(&mut evidence);
    match &result {
        Ok(()) => evidence.report["verdict"] = json!("passed"),
        Err(error) => evidence.report["error"] = json!(error.to_string()),
    }
    evidence.report["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    evidence.save()?;
    eprintln!("generic dispatch evidence: {}", evidence.root.display());
    result
}
