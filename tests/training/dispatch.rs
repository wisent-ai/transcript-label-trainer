//! Real lifecycle training, source-isolated publication and retained Stado replay.
//! Requires the declared GPU, reviewed datasets, Ster and Brama; never substitutes providers.
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
    label: &str,
) -> Result<Value> {
    let id = field(stdout, "Stado job: ")?;
    let uri = field(stdout, "model artifact: ")?;
    let status = evidence.succeeds(stado, &["machine", "status", id], None)?;
    let status: Value = serde_json::from_str(&status)?;
    let job = &status["job"];
    require(
        job["job_id"] == id && job["terminal"] == true,
        format!("job is not terminal: {status}"),
    )?;
    require(
        job["error"].is_null(),
        format!("remote job failed: {status}"),
    )?;
    require(
        job["output_uri"] == uri,
        format!("persisted output URI differs: {status}"),
    )?;
    let run_id = job["run_id"].as_str().ok_or("job has no run_id")?;
    require(
        run_id.ends_with(source),
        format!("job identity does not retain source: {run_id}"),
    )?;
    let key = uri.rsplit('/').next().ok_or("output URI has no key")?;
    require(
        job["command"]
            .as_str()
            .ok_or("job has no command")?
            .contains(&format!("/oko-lifecycle-{key}")),
        "persisted workspace does not use the artifact identity",
    )?;
    let manifest_path = evidence.root.join(format!("{label}-manifest.json"));
    evidence.succeeds(
        stado,
        &[
            "storage",
            "get",
            &format!("{uri}/model-manifest.json"),
            manifest_path.to_str().ok_or("manifest path is not UTF-8")?,
        ],
        None,
    )?;
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    require(
        manifest["qualified"] == true,
        format!("actual trained model did not qualify: {manifest}"),
    )?;
    for name in ["final-judge.json", "metrics.json", "metrics-gguf.json"] {
        let signature = &manifest["files"][name];
        let path = evidence.root.join(format!("{label}-{name}"));
        evidence.succeeds(
            stado,
            &[
                "storage",
                "get",
                &format!("{uri}/{name}"),
                path.to_str().ok_or("evidence path is not UTF-8")?,
            ],
            None,
        )?;
        require(
            signature["sha256"] == digest(&path)?,
            format!("published {name} checksum differs"),
        )?;
        require(
            signature["bytes"] == fs::metadata(&path)?.len(),
            format!("published {name} length differs"),
        )?;
    }
    let judge: Value = serde_json::from_slice(&fs::read(
        evidence.root.join(format!("{label}-final-judge.json")),
    )?)?;
    require(
        judge["passed"] == true,
        "independent final judge did not pass",
    )?;
    Ok(json!({"job": job, "manifest_sha256": digest(&manifest_path)?}))
}

fn journey(evidence: &mut Evidence) -> Result<()> {
    let source = required("TLT_TEST_SOURCE_REVISION")?;
    let alternate = required("TLT_TEST_ALTERNATE_SOURCE_REVISION")?;
    require(
        source != alternate,
        "source-isolation qualification needs distinct real revisions",
    )?;
    let head = evidence.succeeds("git", &["rev-parse", "HEAD"], None)?;
    require(
        head.trim() == source,
        "candidate source differs from the declared revision",
    )?;
    require(
        evidence
            .succeeds("git", &["status", "--porcelain"], None)?
            .trim()
            .is_empty(),
        "commit candidate and journey before qualification",
    )?;
    for revision in [&source, &alternate] {
        let resolved = evidence.succeeds(
            "git",
            &["rev-parse", &format!("{revision}^{{commit}}")],
            None,
        )?;
        require(
            resolved.trim() == revision,
            "state a full existing source commit",
        )?;
    }
    let stado = required("STADO_BIN")?;
    let target = required("TLT_TEST_COMPUTE_TARGET")?;
    let brama = required("TLT_TEST_BRAMA_URL")?;
    let options = required("TLT_TEST_STER_OPTIONS")?;
    let train = PathBuf::from(required("TLT_TEST_TRAIN_DATASET")?).canonicalize()?;
    let eval = PathBuf::from(required("TLT_TEST_EVAL_DATASET")?).canonicalize()?;
    evidence.report["source_revision"] = json!(source);
    evidence.report["alternate_source_revision"] = json!(alternate);
    evidence.report["binary_sha256"] = json!(digest(Path::new(BINARY))?);
    evidence.report["train_sha256"] = json!(digest(&train)?);
    evidence.report["eval_sha256"] = json!(digest(&eval)?);
    evidence.save()?;
    let args = [
        "lifecycle-model",
        train.to_str().ok_or("train path is not UTF-8")?,
        eval.to_str().ok_or("eval path is not UTF-8")?,
        "--compute-target",
        &target,
        "--brama-url",
        &brama,
        "--ster-options",
        &options,
    ];
    for invalid in ["not-a-commit", "", " \t\n"] {
        let (accepted, stdout, stderr) = evidence.run(BINARY, &args, Some(invalid))?;
        require(
            !accepted
                && stderr.contains("TLT_REPO_REF")
                && stderr.contains("trainer source commit"),
            format!("missing source-admission refusal for {invalid:?}: {stderr}"),
        )?;
        require(!stdout.contains("Job ID:"), "invalid source was submitted")?;
    }
    let first = evidence.succeeds(BINARY, &args, Some(&source))?;
    let first = inspect(evidence, &stado, &first, &source, "first")?;
    let replay = evidence.succeeds(BINARY, &args, Some(&source))?;
    let replay = inspect(evidence, &stado, &replay, &source, "replay")?;
    require(
        first["job"]["job_id"] == replay["job"]["job_id"],
        "identical request created a second job",
    )?;
    require(
        first["manifest_sha256"] == replay["manifest_sha256"],
        "replay changed the published model",
    )?;
    let other = evidence.succeeds(BINARY, &args, Some(&alternate))?;
    let other = inspect(evidence, &stado, &other, &alternate, "alternate")?;
    require(
        first["job"]["job_id"] != other["job"]["job_id"],
        "different source reused the job",
    )?;
    require(
        first["job"]["output_uri"] != other["job"]["output_uri"],
        "different source reused model output",
    )?;
    require(
        first["job"]["command"] != other["job"]["command"],
        "different source reused staging workspace",
    )?;
    require(
        evidence.report["train_sha256"] == digest(&train)?,
        "training input changed",
    )?;
    require(
        evidence.report["eval_sha256"] == digest(&eval)?,
        "evaluation input changed",
    )?;
    evidence.report["observations"] = json!({"first": first, "replay": replay, "alternate": other});
    Ok(())
}

fn main() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".build")
        .join(format!("lifecycle-dispatch-{}", rand::random::<u128>()));
    fs::create_dir_all(root.parent().ok_or("report has no parent")?)?;
    fs::create_dir(&root)?;
    let mut evidence = Evidence {
        root,
        report: json!({"commands": [], "verdict": "failed"}),
    };
    evidence.save()?;
    let result = journey(&mut evidence);
    match &result {
        Ok(()) => evidence.report["verdict"] = json!("passed"),
        Err(error) => evidence.report["error"] = json!(error.to_string()),
    }
    evidence.report["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    evidence.save()?;
    eprintln!("lifecycle dispatch evidence: {}", evidence.root.display());
    result
}
