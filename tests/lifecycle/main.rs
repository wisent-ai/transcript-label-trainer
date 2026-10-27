//! The lifecycle model is trained on the chat it is served. The real
//! executable turns reviewed rows into the Ster example set `ster tune sft`
//! reads, refuses a row it cannot turn into one, and `lifecycle-model` refuses
//! to submit a job that states no Ster settings before it reads or uploads a
//! byte. Every command, its exit status and output go to the case's
//! report.json under `.build/`.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");
const SYSTEM_PROMPT: &str = include_str!("../../training/lifecycle-model/lifecycle-system-prompt.txt");

struct Run {
    root: PathBuf,
    report: Value,
}

impl Run {
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("lifecycle-{case}-{}", std::process::id()));
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
        eprintln!("lifecycle evidence: {}", self.root.display());
    }
}

/// Oko's envelope offering a new goal and one open goal, as the user turn sends it.
fn envelope() -> String {
    json!({
        "text": "keep going on the login fix",
        "candidates": [
            {"ref": "NEW_GOAL", "title": "a new goal"},
            {"ref": "goal-login", "title": "Fix the login redirect"},
        ],
    })
    .to_string()
}

/// One reviewed row answered by `answer` (no assistant turn at all when `None`).
fn reviewed_row(id: &str, answer: Option<&Value>) -> Value {
    let mut messages = vec![
        json!({"role": "system", "content": "the reviewer's copy of the prompt"}),
        json!({"role": "user", "content": envelope()}),
    ];
    messages.extend(answer.map(|answer| json!({"role": "assistant", "content": answer.to_string()})));
    json!({"id": id, "messages": messages})
}

fn write_rows(path: &Path, rows: &[Value]) {
    let lines: String = rows.iter().map(|row| format!("{row}\n")).collect();
    fs::write(path, lines).unwrap();
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn reviewed_rows_become_the_served_chat_answered_by_the_reviewed_decision() {
    let mut run = Run::start("examples");
    let rows = run.root.join("reviewed.jsonl");
    let output = run.root.join("examples.json");
    let decision = json!({"action": "continueCurrent", "goal_ref": "goal-login", "title": "kept out", "lifecycle_evidence": "none"});
    write_rows(&rows, &[reviewed_row("row-continue", Some(&decision))]);

    let answer = run.run(&["lifecycle-examples", "--rows", path(&rows), "--output", path(&output)]);
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));

    let mut written: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    // The completion is the decision as JSON text; read it back as JSON so
    // the comparison is of the decision, not of its spacing.
    for example in written["examples"].as_array_mut().unwrap() {
        example["completion"] = serde_json::from_str(example["completion"].as_str().unwrap()).unwrap();
    }
    assert_eq!(
        written,
        json!({"examples": [{
            "system": SYSTEM_PROMPT.trim(),
            "prompt": envelope(),
            "completion": {"action": "continueCurrent", "goal_ref": "goal-login", "title": "", "lifecycle_evidence": "none"},
        }]}),
        "one example per row: the served system prompt (not the reviewer's copy), the user envelope as sent, and the reviewed decision with its title blanked"
    );
    run.passed();
}

#[test]
fn a_row_without_a_reviewed_decision_is_refused_by_id_and_nothing_is_written() {
    let mut run = Run::start("examples-unreviewed");
    let rows = run.root.join("reviewed.jsonl");
    let output = run.root.join("examples.json");
    write_rows(&rows, &[reviewed_row("row-unreviewed", None)]);

    let answer = run.run(&["lifecycle-examples", "--rows", path(&rows), "--output", path(&output)]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("row-unreviewed must contain one reviewed assistant decision"), "{stderr}");
    assert!(!output.exists(), "no example set is written");
    run.passed();
}

#[test]
fn a_decision_outside_the_contract_is_refused_by_id() {
    let mut run = Run::start("examples-contract");
    let rows = run.root.join("reviewed.jsonl");
    let output = run.root.join("examples.json");
    let decision = json!({"action": "finishGoal", "goal_ref": "goal-login", "title": "", "lifecycle_evidence": "none"});
    write_rows(&rows, &[reviewed_row("row-finish", Some(&decision))]);

    let answer = run.run(&["lifecycle-examples", "--rows", path(&rows), "--output", path(&output)]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("row-finish finishGoal lacks explicit completion evidence"), "{stderr}");
    assert!(!output.exists(), "no example set is written");
    run.passed();
}

#[test]
fn a_blank_reviewed_identity_is_refused_before_writing_either_split() {
    let mut run = Run::start("split-blank-id");
    let train = run.root.join("reviewed-train.jsonl");
    let eval = run.root.join("reviewed-eval.jsonl");
    let output_train = run.root.join("train.jsonl");
    let output_eval = run.root.join("eval.jsonl");
    write_rows(&train, &[json!({"id": " ", "split_day": "day-under-test"})]);
    write_rows(&eval, &[]);

    let answer = run.run(&[
        "lifecycle-split-by-day",
        "--train", path(&train),
        "--eval", path(&eval),
        "--eval-day", "day-under-test",
        "--output-train", path(&output_train),
        "--output-eval", path(&output_eval),
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("a reviewed row has no nonblank string id"), "{stderr}");
    assert!(stderr.contains(&train.display().to_string()), "{stderr}");
    assert!(!output_train.exists() && !output_eval.exists(), "neither output was written");
    run.passed();
}

#[test]
fn reviewed_rows_are_resplit_by_day_and_persisted_in_disjoint_outputs() {
    let mut run = Run::start("split-by-day");
    let train = run.root.join("reviewed-train.jsonl");
    let eval = run.root.join("reviewed-eval.jsonl");
    let output_train = run.root.join("train.jsonl");
    let output_eval = run.root.join("eval.jsonl");
    let schema: Value = serde_json::from_str(include_str!(
        "../../training/lifecycle-model/lifecycle-output-schema.json"
    )).unwrap();
    let actions = schema["oneOf"].as_array().unwrap();
    let mut rows = Vec::new();
    for branch in actions {
        let action = branch["properties"]["action"]["const"].as_str().unwrap();
        for day in ["earlier", "held-out"] {
            let mut row = reviewed_row(
                &format!("{day}-{action}"),
                Some(&json!({"action": action})),
            );
            row["split_day"] = json!(day);
            rows.push(row);
        }
    }
    write_rows(&train, &rows);
    write_rows(&eval, &[]);

    let answer = run.run(&[
        "lifecycle-split-by-day",
        "--train", path(&train),
        "--eval", path(&eval),
        "--eval-day", "held-out",
        "--output-train", path(&output_train),
        "--output-eval", path(&output_eval),
    ]);
    assert!(answer.status.success(), "{}", String::from_utf8_lossy(&answer.stderr));
    for (output, day, split) in [
        (&output_train, "earlier", "train"),
        (&output_eval, "held-out", "eval"),
    ] {
        let written = fs::read_to_string(output).unwrap();
        let persisted: Vec<Value> = written.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(persisted.len(), actions.len(), "{}", output.display());
        assert!(persisted.iter().all(|row|
            row["split_day"] == day && row["split"] == split
                && row["id"].as_str().unwrap().starts_with(day)
        ), "{}", output.display());
    }
    run.passed();
}

#[test]
fn a_lifecycle_job_without_ster_settings_is_refused_before_anything_is_read() {
    let mut run = Run::start("model-no-ster-options");
    let absent = run.root.join("absent.jsonl");
    let answer = run.run(&[
        "lifecycle-model",
        path(&absent),
        path(&absent),
        "--compute-target",
        "target-under-test",
        "--brama-url",
        "http://brama.invalid",
        "--ster-options",
        " ",
    ]);
    let stderr = String::from_utf8_lossy(&answer.stderr);
    assert!(!answer.status.success(), "{stderr}");
    assert!(stderr.contains("--ster-options cannot be empty"), "the refusal names the missing settings: {stderr}");
    assert!(!stderr.contains("absent.jsonl"), "the datasets were never read: {stderr}");
    run.passed();
}
