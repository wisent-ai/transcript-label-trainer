//! Every command that reaches Brama first resolves where Brama is and reads
//! its signing secret. The real executable runs `humanizer-audit` with an
//! isolated HOME, and each way that resolution fails must be refused with
//! its own cause: a BRAMA_URL set but empty is named instead of being read as
//! unset, a stado that cannot run names the program, and the real Stado's
//! refusal of the placement or of a credential role is carried in the
//! answer, not replaced by a generic "no Brama endpoint". Every command, its
//! exit status and output go to the case's report.json under `.build/`.
//!
//! TLT_TEST_STADO names the real stado executable; it is required.
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::{json, Value};

const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");

struct Run {
    root: PathBuf,
    home: PathBuf,
    report: Value,
}

impl Run {
    fn start(case: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".build")
            .join(format!("brama-{case}-{}", std::process::id()));
        let home = root.join("home");
        fs::create_dir_all(&home).unwrap();
        let row = json!({"id": "case-one", "source": "draft", "target": "voice", "base": "base", "student": "student"});
        fs::write(root.join("predictions.jsonl"), format!("{row}\n")).unwrap();
        Self { root, home, report: json!({"case": case, "commands": [], "outcome": "failed"}) }
    }

    /// `humanizer-audit` with an identity present and every Brama address
    /// removed, so the run stops at resolving Brama before any request.
    fn audit(&mut self, env: &[(&str, String)], without: &[&str]) -> Output {
        let predictions = self.root.join("predictions.jsonl");
        let output = self.root.join("audit.json");
        let args = [
            "humanizer-audit".to_string(),
            predictions.display().to_string(),
            "--output".to_string(),
            output.display().to_string(),
        ];
        let mut command = Command::new(BINARY);
        command
            .args(&args)
            .env("HOME", &self.home)
            .env("WISENT_APP_AGENT_ID", "transcript-label-trainer-test")
            .env_remove("BRAMA_URL");
        for name in without {
            command.env_remove(name);
        }
        for (name, value) in env {
            command.env(name, value);
        }
        let answer = command.output().unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": args,
            "env": env.iter().map(|(name, value)| json!({"name": name, "value": value})).collect::<Vec<_>>(),
            "exit_status": answer.status.code(),
            "stdout": String::from_utf8_lossy(&answer.stdout),
            "stderr": String::from_utf8_lossy(&answer.stderr),
        }));
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap()).unwrap();
        answer
    }

    fn passed(mut self) {
        self.report["outcome"] = json!("passed");
        fs::write(self.root.join("report.json"), serde_json::to_vec_pretty(&self.report).unwrap()).unwrap();
        eprintln!("brama evidence: {}", self.root.display());
    }
}

fn stado() -> String {
    std::env::var("TLT_TEST_STADO").expect("TLT_TEST_STADO names the real stado executable")
}

fn secrets() -> Vec<(&'static str, String)> {
    vec![
        ("WISENT_APP_AGENT_AUTH_SECRET", "unsent-test-secret".to_string()),
        ("BRAMA_TOKEN", "unsent-test-token".to_string()),
    ]
}

fn refusal(answer: &Output) -> String {
    assert!(!answer.status.success(), "resolution must be refused: {}", String::from_utf8_lossy(&answer.stdout));
    String::from_utf8_lossy(&answer.stderr).into_owned()
}

#[test]
fn an_empty_brama_url_is_refused_by_name() {
    let mut run = Run::start("empty-brama-url");
    let mut env = secrets();
    env.push(("BRAMA_URL", String::new()));
    let stderr = refusal(&run.audit(&env, &[]));
    assert!(
        stderr.contains("BRAMA_URL is set but empty"),
        "the refusal must name the empty variable: {stderr}"
    );
    assert!(!stderr.contains("no Brama endpoint"), "an empty variable is not an absent endpoint: {stderr}");
    run.passed();
}

#[test]
fn a_stado_that_cannot_run_is_named() {
    let mut run = Run::start("stado-missing");
    let missing = run.root.join("no-stado-here");
    let mut env = secrets();
    env.push(("TLT_STADO_BIN", missing.display().to_string()));
    let stderr = refusal(&run.audit(&env, &[]));
    assert!(stderr.contains("Stado could not place Brama"), "{stderr}");
    assert!(
        stderr.contains(&format!("{} could not be run", missing.display())),
        "the refusal must name the program that could not run: {stderr}"
    );
    run.passed();
}

#[test]
fn stados_refusal_of_the_placement_is_carried() {
    let mut run = Run::start("stado-placement");
    let mut env = secrets();
    env.push(("TLT_STADO_BIN", stado()));
    let stderr = refusal(&run.audit(&env, &[]));
    let Some((_, cause)) = stderr.split_once("service directory connect --json brama` ") else {
        panic!("the refusal must quote the Stado command: {stderr}");
    };
    assert!(
        cause.starts_with("was refused: ") || cause.starts_with("answered "),
        "the refusal must carry what Stado answered: {stderr}"
    );
    assert!(!cause.trim_start_matches("was refused: ").trim().is_empty(), "Stado's reason must not be empty: {stderr}");
    run.passed();
}

#[test]
fn stados_refusal_of_a_credential_role_is_carried() {
    let mut run = Run::start("stado-role");
    let env = vec![
        ("TLT_STADO_BIN", stado()),
        ("TLT_BRAMA_AGENT_ROLE", "transcript-label-trainer-test-absent-role#secret".to_string()),
    ];
    let stderr = refusal(&run.audit(&env, &["WISENT_APP_AGENT_AUTH_SECRET"]));
    let quoted = "credentials get --role transcript-label-trainer-test-absent-role --field secret` was refused: ";
    let Some((_, reason)) = stderr.split_once(quoted) else {
        panic!("the refusal must quote the Stado command: {stderr}");
    };
    assert!(!reason.trim().is_empty(), "Stado's reason must be carried: {stderr}");
    assert!(!stderr.contains("run it to read Stado's reason"), "{stderr}");
    run.passed();
}
