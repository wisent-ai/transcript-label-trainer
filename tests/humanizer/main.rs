//! The humanizer trains with Ster on its prepared splits and is measured on
//! served models. The real executable turns a prepared split into the Ster
//! example set `ster tune sft` reads, refuses a row it cannot turn into one,
//! and refuses to evaluate a model that does not exist before starting a
//! server. Every command, its exit status and output go to the case's
//! report.json under `.build/`.
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
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("humanizer-{case}-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        Self { root, report: json!({"case": case, "commands": [], "outcome": "failed"}) }
    }

    fn run(&mut self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let output = Command::new(BINARY).args(args).envs(env.iter().copied()).output().unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": args,
            "env": env.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>(),
            "exit_status": output.status.code(),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
        }));
        self.write();
        output
    }

    fn write(&self) {
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap()).unwrap();
    }

    fn passed(mut self) {
        self.report["outcome"] = json!("passed");
        self.write();
        eprintln!("humanizer evidence: {}", self.root.display());
    }
}

fn prepared(id: &str, turns: &[(&str, &str)]) -> Value {
    let messages: Vec<Value> = turns.iter().map(|(role, content)| json!({"role": role, "content": content})).collect();
    json!({"id": id, "messages": messages, "metadata": {"session_id": "session-under-test"}})
}

fn write_rows(path: &Path, rows: &[Value]) {
    let lines: String = rows.iter().map(|row| format!("{row}\n")).collect();
    fs::write(path, lines).unwrap();
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn a_prepared_split_becomes_ster_examples() {
    let mut run = Run::start("examples");
    let rows = run.root.join("train.jsonl");
    let output = run.root.join("examples.json");
    write_rows(
        &rows,
        &[prepared(
            "row-voice",
            &[
                ("system", "Rewrite the text in the author's own voice."),
                ("user", "Please be advised that the deployment has been completed successfully."),
                ("assistant", "deploy's done, all green"),
            ],
        )],
    );

    let answer = run.run(&["humanizer-examples", "--rows", path(&rows), "--output", path(&output)], &[]);
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));
    let written: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        written,
        json!({"examples": [{
            "system": "Rewrite the text in the author's own voice.",
            "prompt": "Please be advised that the deployment has been completed successfully.",
            "completion": "deploy's done, all green",
        }]}),
        "the row's system prompt, the generic source as the prompt, the user's target as the completion"
    );
    run.passed();
}

#[test]
fn a_row_without_its_target_is_refused_by_id_and_nothing_is_written() {
    let mut run = Run::start("examples-no-target");
    let rows = run.root.join("train.jsonl");
    let output = run.root.join("examples.json");
    write_rows(&rows, &[prepared("row-untargeted", &[("system", "Rewrite."), ("user", "Some generic text.")])]);

    let answer = run.run(&["humanizer-examples", "--rows", path(&rows), "--output", path(&output)], &[]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("row \"row-untargeted\" must hold exactly one assistant turn"), "{stderr}");
    assert!(!output.exists(), "no example set is written");
    run.passed();
}

#[test]
fn an_evaluation_of_a_missing_student_is_refused_before_a_server_starts() {
    let mut run = Run::start("evaluate-no-student");
    let rows = run.root.join("test.jsonl");
    write_rows(&rows, &[prepared("row-voice", &[("system", "Rewrite."), ("user", "Generic text."), ("assistant", "my text")])]);
    let base = run.root.join("base-f16.gguf");
    fs::write(&base, "a file standing where the base model is expected").unwrap();
    let student = run.root.join("absent-student.gguf");
    let (predictions, log) = (run.root.join("predictions.jsonl"), run.root.join("server.log"));
    let (metrics, server) = (run.root.join("metrics.json"), run.root.join("absent-llama-server"));

    let answer = run.run(
        &[
            "humanizer-evaluate-gguf",
            "--base",
            path(&base),
            "--student",
            path(&student),
            "--dataset",
            path(&rows),
            "--predictions",
            path(&predictions),
            "--metrics",
            path(&metrics),
            "--server",
            path(&server),
            "--server-log",
            path(&log),
            "--base-model",
            "TheDrummer/Cydonia-24B-v4.3",
            "--base-revision",
            "db0426d39d4bd4a6d34fdc71db97569da68f55e1",
            "--parallel",
            "1",
            "--slot-context",
            "4096",
            "--gpu-layers",
            "all",
            "--max-tokens",
            "512",
            "--chrf-order",
            "3",
        ],
        &[],
    );
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains(&format!("the student model {} does not exist", student.display())), "{stderr}");
    assert!(!predictions.exists() && !log.exists(), "no prediction is written and no server started");
    run.passed();
}
