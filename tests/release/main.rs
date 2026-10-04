//! `release-publish` sends a model's weights where the model runs, and refuses
//! a destination that is not the model's before it downloads a byte. The real
//! executable is asked to publish from a job output that does not exist: a
//! refusal that names the destination proves the check ran first, because the
//! download it would otherwise attempt fails with a different sentence.
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");
const SOURCE: &str = "stado://probierz/artifacts/models/oko/lifecycle-qwen3-4b/absent";

struct Run {
    root: PathBuf,
    report: Value,
}

impl Run {
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("release-{case}-{}", hex::encode(rand::random::<[u8; 8]>())));
        fs::create_dir_all(&root).unwrap();
        Self { root, report: json!({"case": case, "commands": [], "outcome": "failed"}) }
    }

    fn publish(&mut self, args: &[&str]) -> Output {
        let output = Command::new(BINARY)
            .arg("release-publish")
            .arg(SOURCE)
            .args(args)
            .env_remove("HF_TOKEN")
            .output()
            .unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": args,
            "exit_status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }));
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap())
            .unwrap();
        output
    }

    fn passed(mut self) {
        self.report["outcome"] = json!("passed");
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap())
            .unwrap();
        eprintln!("release evidence: {}", self.root.display());
    }
}

#[test]
fn a_served_model_without_a_private_repository_is_refused_before_any_download() {
    let mut run = Run::start("lifecycle-no-repo");
    let output = run.publish(&["--model", "lifecycle"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("the lifecycle model is served from a GPU host")
            && stderr.contains("--repo OWNER/REPOSITORY")
            && stderr.contains("the fleet's release store is not a model store"),
        "the refusal names where the weights go: {stderr}"
    );
    assert!(!stderr.contains("downloading"), "nothing was downloaded: {stderr}");
    run.passed();
}

#[test]
fn a_desktop_model_given_a_private_repository_is_refused() {
    let mut run = Run::start("goal-with-repo");
    let output = run.publish(&["--model", "goal", "--repo", "owner/private-goal"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("the goal model ships through the release channel")
            && stderr.contains("--repo owner/private-goal"),
        "the refusal names the destination the model does take: {stderr}"
    );
    assert!(!stderr.contains("downloading"), "nothing was downloaded: {stderr}");
    run.passed();
}
