//! Every training run states its backend's settings. The real executable
//! refuses `train` without a tfidf-logreg setting by its key and flag, refuses
//! a flag of the other backend, and refuses a job whose `training` section
//! holds a setting out of range, each before any label is read. Every command,
//! its exit status and output go to the case's report.json under `.build/`.
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");

struct Run {
    root: PathBuf,
    report: Value,
}

impl Run {
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("training-{case}-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        Self { root, report: json!({"case": case, "commands": [], "outcome": "failed"}) }
    }

    /// Runs the executable with both roots inside the case directory, so no
    /// operator data is read or written.
    fn run(&mut self, args: &[&str]) -> Output {
        let root = self.root.to_str().unwrap().to_string();
        let mut full = vec!["--training-root", root.as_str(), "--storage-root", root.as_str()];
        full.extend_from_slice(args);
        let output = Command::new(BINARY).args(&full).output().unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": full,
            "exit_status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }));
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap()).unwrap();
        output
    }

    fn passed(mut self) {
        self.report["outcome"] = json!("passed");
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap()).unwrap();
        eprintln!("training evidence: {}", self.root.display());
    }
}

/// One, the smallest positive whole number, as the text a flag carries.
fn one() -> String {
    NonZeroUsize::MIN.to_string()
}

/// A share above all of a whole: one more than one.
fn above_all() -> String {
    NonZeroUsize::MIN.saturating_add(NonZeroUsize::MIN.get()).to_string()
}

#[test]
fn train_without_a_tfidf_setting_is_refused_by_its_key_and_flag() {
    let mut run = Run::start("missing-setting");
    let answer = run.run(&["train", "--aspect", "topic", "--no-eval-split"]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains("training.ngram_max (--ngram-max) is required: tfidf-logreg assumes no value for it"),
        "{stderr}"
    );
    assert!(!run.root.join("models").exists(), "no artifact directory is written");
    run.passed();
}

#[test]
fn a_fine_tune_flag_on_a_tfidf_run_is_refused_by_name() {
    let mut run = Run::start("stray-flag");
    let floor = one();
    let answer = run.run(&[
        "train",
        "--aspect",
        "topic",
        "--no-eval-split",
        "--epochs",
        &floor,
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("--epochs does not apply to model 'tfidf-logreg'"), "{stderr}");
    run.passed();
}

#[test]
fn a_job_with_a_line_search_share_above_all_is_refused_by_its_key() {
    let mut run = Run::start("job-share");
    let (one, above) = (one(), above_all());
    let job = run.root.join("job.yaml");
    fs::write(
        &job,
        format!(
            "name: topic-under-test\ntask: classify the topic\nevaluator: manual\nmodel: tfidf-logreg\n\
             scope:\n  aspect: topic\neval_split: false\ntraining:\n\
             \x20 ngram_max: {one}\n  lowercase: true\n  sublinear_tf: true\n  smooth_idf: true\n\
             \x20 min_df: {one}\n  max_df: {one}\n  c: {one}\n  max_iter: {one}\n  tol: {one}\n\
             \x20 lbfgs_memory: {one}\n  armijo_c1: {above}\n  max_backtracks: {one}\n"
        ),
    )
    .unwrap();
    let answer = run.run(&["run", job.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "training.armijo_c1 (--armijo-c1) must be a share above none and below all, got {above}"
        )),
        "{stderr}"
    );
    run.passed();
}

#[test]
fn a_setting_flag_beside_the_preset_is_refused_by_name() {
    let mut run = Run::start("preset-beside-flag");
    let floor = one();
    let answer = run.run(&[
        "train",
        "--aspect",
        "topic",
        "--no-eval-split",
        "--training",
        "scikit-learn",
        "--max-iter",
        &floor,
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains("--max-iter cannot be stated beside --training scikit-learn"),
        "{stderr}"
    );
    assert!(!run.root.join("models").exists(), "no artifact directory is written");
    run.passed();
}

#[test]
fn a_job_naming_an_unknown_preset_is_refused_with_the_presets_that_exist() {
    let mut run = Run::start("job-unknown-preset");
    let job = run.root.join("job.yaml");
    fs::write(
        &job,
        "name: topic-under-test\ntask: classify the topic\nevaluator: manual\nmodel: tfidf-logreg\n\
         scope:\n  aspect: topic\neval_split: false\ntraining: sklearn\n",
    )
    .unwrap();
    let answer = run.run(&["run", job.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains("unknown training preset 'sklearn': the presets are scikit-learn"),
        "{stderr}"
    );
    run.passed();
}

#[test]
fn the_preset_on_a_fine_tune_is_refused() {
    let mut run = Run::start("preset-on-fine-tune");
    let answer = run.run(&[
        "train",
        "--aspect",
        "topic",
        "--no-eval-split",
        "--model",
        "distilbert-base-multilingual-cased",
        "--training",
        "scikit-learn",
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(
        stderr.contains("training preset 'scikit-learn' applies to model 'tfidf-logreg'"),
        "{stderr}"
    );
    run.passed();
}
