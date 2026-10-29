//! `info` lists every trained aspect and its artifacts from the training
//! root. An aspect directory the trainer cannot list must refuse the command
//! with that directory and the cause; skipped, a trained aspect or a newer
//! hf-* run went missing and an older artifact was marked active. The real
//! executable runs against an isolated training root under `.build/`; every
//! command, its exit status and output go to the case's report.json.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");

struct Run {
    root: PathBuf,
    report: Value,
}

impl Run {
    fn command(&mut self, program: &str, args: &[&str]) -> Output {
        let output = Command::new(program).args(args).output().unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "program": program,
            "args": args,
            "exit_status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }));
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&self.report).unwrap(),
        )
        .unwrap();
        output
    }

    fn info(&mut self) -> Output {
        let training = self.root.join("training");
        let storage = self.root.join("storage");
        let args = [
            "--training-root",
            path(&training),
            "--storage-root",
            path(&storage),
            "info",
        ];
        let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        self.command(BINARY, &args)
    }
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn an_aspect_directory_that_cannot_be_listed_is_refused_with_its_path() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".build")
        .join(format!("model-listing-{}", std::process::id()));
    let aspect = root.join("training").join("models").join("listing-probe");
    fs::create_dir_all(&aspect).unwrap();
    fs::create_dir_all(root.join("storage")).unwrap();
    let mut run = Run {
        root: root.clone(),
        report: json!({"case": "model-listing", "commands": [], "outcome": "failed"}),
    };

    let readable = run.info();
    assert!(
        readable.status.success(),
        "a readable training root is listed: {}",
        String::from_utf8_lossy(&readable.stderr)
    );
    assert!(
        String::from_utf8_lossy(&readable.stdout).contains("listing-probe"),
        "the aspect is listed"
    );

    let locked = run.command("chmod", &["a-r", path(&aspect)]);
    assert!(
        locked.status.success(),
        "{}",
        String::from_utf8_lossy(&locked.stderr)
    );
    let refused = run.info();
    let restored = run.command("chmod", &["u+r", path(&aspect)]);
    assert!(
        restored.status.success(),
        "{}",
        String::from_utf8_lossy(&restored.stderr)
    );

    assert!(
        !refused.status.success(),
        "an unlistable aspect directory must refuse info"
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains(&format!("{} cannot be listed", aspect.display())),
        "the refusal must name the directory and its cause: {stderr}"
    );
    run.report["outcome"] = json!("passed");
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&run.report).unwrap(),
    )
    .unwrap();
    eprintln!("model listing evidence: {}", root.display());
}
