//! The goal model is trained and measured on the chat Jeden serves. The real
//! executable turns reviewed rows into the Ster example set `ster tune sft`
//! reads, keeping the gold rows out; refuses a file it cannot train on;
//! refuses to evaluate a model that does not exist before starting anything;
//! and `goal-model` refuses a job that states no Ster settings before it
//! curates a row. Every command, its exit status and output go to the case's
//! report.json under `.build/`.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");
const SYSTEM_PROMPT: &str = include_str!("../../training/goal-model/goal-system-prompt.txt");

struct Run {
    root: PathBuf,
    report: Value,
}

impl Run {
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("goal-{case}-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        Self { root, report: json!({"case": case, "commands": [], "outcome": "failed"}) }
    }

    fn run(&mut self, args: &[&str]) -> Output {
        let output = Command::new(BINARY).args(args).output().unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": args,
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
        eprintln!("goal evidence: {}", self.root.display());
    }
}

fn row(session: &str, message: &str, goal: Option<&str>, gold: bool) -> Value {
    json!({"session_id": session, "runtime": "omp", "message": message, "goal": goal, "goal_source": "teacher", "gold": gold})
}

fn write_rows(path: &Path, rows: &[Value]) {
    let lines: String = rows.iter().map(|row| format!("{row}\n")).collect();
    fs::write(path, lines).unwrap();
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn teacher_rows_become_the_served_chat_and_gold_rows_stay_out() {
    let mut run = Run::start("examples");
    let rows = run.root.join("reviewed-goals.jsonl");
    let output = run.root.join("examples.json");
    let reviewed = [
        row("session-task", "please fix the login redirect loop", Some("Fix login redirect loop"), false),
        row("session-chat", "thanks, that works", None, false),
        row("session-gold", "add dark mode to settings", Some("Add settings dark mode"), true),
    ];
    write_rows(&rows, &reviewed);

    let answer = run.run(&["goal-examples", "--rows", path(&rows), "--output", path(&output)]);
    let stdout = String::from_utf8_lossy(&answer.stdout);
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));
    let counts: Value = serde_json::from_str(&stdout).unwrap();
    let gold = reviewed.iter().filter(|row| row["gold"] == json!(true)).count();
    assert_eq!(counts["held_out"], json!(gold), "every gold row is counted as held out: {stdout}");
    assert_eq!(counts["examples"], json!(reviewed.len() - gold), "every other row is an example: {stdout}");

    let written: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(
        written,
        json!({"examples": [
            {"system": SYSTEM_PROMPT.trim(), "prompt": "<user>please fix the login redirect loop</user>", "completion": "<goal>Fix login redirect loop</goal>"},
            {"system": SYSTEM_PROMPT.trim(), "prompt": "<user>thanks, that works</user>", "completion": "<goal/>"},
        ]}),
        "each teacher row as Jeden's chat, a row without a goal answered <goal/>, the gold row absent"
    );
    run.passed();
}

#[test]
fn a_file_of_only_gold_rows_is_refused_and_nothing_is_written() {
    let mut run = Run::start("examples-only-gold");
    let rows = run.root.join("reviewed-goals.jsonl");
    let output = run.root.join("examples.json");
    write_rows(&rows, &[row("session-gold", "add dark mode to settings", Some("Add settings dark mode"), true)]);

    let answer = run.run(&["goal-examples", "--rows", path(&rows), "--output", path(&output)]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("reviewed rows is held-out gold; none is left to train on"), "{stderr}");
    assert!(!output.exists(), "no example set is written");
    run.passed();
}

#[test]
fn an_empty_file_is_refused_by_name() {
    let mut run = Run::start("examples-empty");
    let rows = run.root.join("reviewed-goals.jsonl");
    fs::write(&rows, "").unwrap();
    let output = run.root.join("examples.json");

    let answer = run.run(&["goal-examples", "--rows", path(&rows), "--output", path(&output)]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains(&format!("{} holds no reviewed row", rows.display())), "{stderr}");
    run.passed();
}

#[test]
fn an_evaluation_of_a_model_that_does_not_exist_is_refused_before_a_server_starts() {
    let mut run = Run::start("evaluate-no-model");
    let rows = run.root.join("reviewed-goals.jsonl");
    write_rows(&rows, &[row("session-gold", "add dark mode to settings", Some("Add settings dark mode"), true)]);
    let model = run.root.join("absent.gguf");
    let predictions = run.root.join("predictions.jsonl");
    let (metrics, server, log) = (run.root.join("metrics-gguf.json"), run.root.join("absent-llama-server"), run.root.join("server.log"));

    let answer = run.run(&[
        "goal-evaluate-gguf",
        "--model",
        path(&model),
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
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains(&format!("the served model {} does not exist", model.display())), "{stderr}");
    assert!(!predictions.exists() && !log.exists(), "no prediction is written and no server started");
    run.passed();
}

#[test]
fn a_goal_job_without_ster_settings_is_refused_before_a_row_is_curated() {
    let mut run = Run::start("model-no-ster-options");
    let answer = run.run(&[
        "goal-model",
        "--compute-target",
        "target-under-test",
        "--limit",
        "1",
        "--ster-options",
        " ",
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("--ster-options cannot be empty"), "the refusal names the missing settings: {stderr}");
    assert!(answer.stdout.is_empty(), "no curation summary was printed: {}", String::from_utf8_lossy(&answer.stdout));
    run.passed();
}
