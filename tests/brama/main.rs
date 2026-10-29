//! Every command that reaches a model service first asks where it is
//! (`BRAMA_URL`, else `stado capabilities resolve inference --consumer
//! transcript-label-trainer`) and, for Brama, reads its signing secret. The
//! real executable runs `humanizer-audit` with an isolated HOME, and each way
//! that resolution fails must be refused with its own cause: a BRAMA_URL set
//! but empty is named instead of being read as unset, a stado that cannot run
//! names the program, and the real Stado's refusal of every alternative or
//! of a credential role is carried in the answer, not replaced by a generic
//! "no Brama endpoint". When Brama does not answer and an endpoint is
//! declared in its place, that endpoint is the one asked, for its own model.
//!
//! A command that asks a typed question (a label, a verdict, an audit) asks
//! it on Brama's decision endpoint, which answers only its two decision
//! aliases; a command naming any other alias for that question is refused
//! before it reads a session or a prediction. Every command, its exit status
//! and output go to the case's report.json under `.build/`.
//!
//! TLT_TEST_STADO names the real stado executable; it is required.
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::mpsc;

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
        Self {
            root,
            home,
            report: json!({"case": case, "commands": [], "outcome": "failed"}),
        }
    }

    /// The real executable with `args`, an identity present and every Brama
    /// address removed, so a run stops at resolving Brama before any request.
    fn invoke(&mut self, args: &[String], env: &[(&str, String)], without: &[&str]) -> Output {
        let mut command = Command::new(BINARY);
        command
            .args(args)
            .env("HOME", &self.home)
            .env("WISENT_APP_AGENT_ID", "transcript-label-trainer-test")
            .env_remove("BRAMA_URL");
        for name in without {
            command.env_remove(name);
        }
        for (name, value) in env {
            command.env(name, value);
        }
        let answer = stado_wait::output(&mut command).unwrap();
        self.report["commands"].as_array_mut().unwrap().push(json!({
            "args": args,
            "env": env.iter().map(|(name, value)| json!({"name": name, "value": value})).collect::<Vec<_>>(),
            "exit_status": answer.status.code(),
            "stdout": String::from_utf8_lossy(&answer.stdout),
            "stderr": String::from_utf8_lossy(&answer.stderr),
        }));
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&self.report).unwrap(),
        )
        .unwrap();
        answer
    }

    /// `humanizer-audit` over the case's one prediction.
    fn audit(&mut self, env: &[(&str, String)], without: &[&str]) -> Output {
        let args = [
            "humanizer-audit".to_string(),
            self.root.join("predictions.jsonl").display().to_string(),
            "--output".to_string(),
            self.root.join("audit.json").display().to_string(),
        ];
        self.invoke(&args, env, without)
    }

    fn passed(mut self) {
        self.report["outcome"] = json!("passed");
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&self.report).unwrap(),
        )
        .unwrap();
        eprintln!("brama evidence: {}", self.root.display());
    }
}

fn stado() -> String {
    std::env::var("TLT_TEST_STADO").expect("TLT_TEST_STADO names the real stado executable")
}

fn secrets() -> Vec<(&'static str, String)> {
    vec![
        (
            "WISENT_APP_AGENT_AUTH_SECRET",
            "unsent-test-secret".to_string(),
        ),
        ("BRAMA_TOKEN", "unsent-test-token".to_string()),
    ]
}

fn refusal(answer: &Output) -> String {
    assert!(
        !answer.status.success(),
        "resolution must be refused: {}",
        String::from_utf8_lossy(&answer.stdout)
    );
    String::from_utf8_lossy(&answer.stderr).into_owned()
}

fn strings(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| arg.to_string()).collect()
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
    assert!(
        !stderr.contains("no Brama endpoint"),
        "an empty variable is not an absent endpoint: {stderr}"
    );
    run.passed();
}

#[test]
fn a_stado_that_cannot_run_is_named() {
    let mut run = Run::start("stado-missing");
    let missing = run.root.join("no-stado-here");
    let mut env = secrets();
    env.push(("TLT_STADO_BIN", missing.display().to_string()));
    let stderr = refusal(&run.audit(&env, &[]));
    assert!(stderr.contains("Stado handed out none"), "{stderr}");
    assert!(
        stderr.contains(&format!("{} could not be run", missing.display())),
        "the refusal must name the program that could not run: {stderr}"
    );
    run.passed();
}

#[test]
fn stados_refusal_of_every_alternative_is_carried() {
    let mut run = Run::start("stado-placement");
    let mut env = secrets();
    env.push(("TLT_STADO_BIN", stado()));
    let stderr = refusal(&run.audit(&env, &[]));
    // The isolated home places Brama nowhere and declares no endpoint, so
    // Stado names both alternatives and why each did not answer.
    let Some((_, named)) = stderr.split_once("no inference alternative answers: ") else {
        panic!("the refusal must carry Stado's alternatives: {stderr}");
    };
    assert!(
        named.starts_with("brama at "),
        "Brama is named first: {stderr}"
    );
    assert!(
        named.contains("model-endpoint at "),
        "the declared endpoint alternative is named too: {stderr}"
    );
    run.passed();
}

#[test]
fn stados_refusal_of_a_credential_role_is_carried() {
    let mut run = Run::start("stado-role");
    // A Brama address is given, so the run goes on to read Brama's identity.
    let env = vec![
        ("TLT_STADO_BIN", stado()),
        ("BRAMA_URL", "http://127.0.0.1:1".to_string()),
        (
            "TLT_BRAMA_AGENT_ROLE",
            "transcript-label-trainer-test-absent-role#secret".to_string(),
        ),
    ];
    let stderr = refusal(&run.audit(&env, &["WISENT_APP_AGENT_AUTH_SECRET"]));
    let quoted = "credentials get --role transcript-label-trainer-test-absent-role --field secret` was refused: ";
    let Some((_, reason)) = stderr.split_once(quoted) else {
        panic!("the refusal must quote the Stado command: {stderr}");
    };
    assert!(
        !reason.trim().is_empty(),
        "Stado's reason must be carried: {stderr}"
    );
    assert!(
        !stderr.contains("run it to read Stado's reason"),
        "{stderr}"
    );
    run.passed();
}

#[test]
fn a_teacher_that_cannot_answer_a_typed_choice_is_refused_before_any_session() {
    let mut run = Run::start("autolabel-chat-alias");
    let args = strings(&[
        "autolabel",
        "--aspect",
        "tasktype",
        "--values",
        "bugfix,feature",
        "--brama-model",
        "chat-only-alias",
    ]);
    let stderr = refusal(&run.invoke(&args, &secrets(), &[]));
    assert!(
        stderr.contains("`chat-only-alias` cannot answer a typed choice"),
        "the refusal must name the alias that cannot answer: {stderr}"
    );
    assert!(
        stderr.contains("decision-model") && stderr.contains("best-decision-model"),
        "the refusal must name the aliases that can: {stderr}"
    );
    assert!(
        !stderr.contains("Brama unreachable"),
        "no request may be sent for a refused alias: {stderr}"
    );
    run.passed();
}

#[test]
fn an_empty_teacher_is_refused_by_its_flag() {
    let mut run = Run::start("autolabel-empty-alias");
    let args = strings(&[
        "autolabel",
        "--aspect",
        "tasktype",
        "--values",
        "bugfix,feature",
        "--brama-model",
        " ",
    ]);
    let stderr = refusal(&run.invoke(&args, &secrets(), &[]));
    assert!(
        stderr.contains("--brama-model is empty"),
        "the refusal must name the empty flag: {stderr}"
    );
    run.passed();
}

#[test]
fn a_goal_audit_by_a_chat_alias_is_refused_before_reading_predictions() {
    let mut run = Run::start("goal-audit-chat-alias");
    let absent = run.root.join("no-predictions-here.jsonl");
    let output = run.root.join("goal-audit.json");
    let args = vec![
        "goal-audit".to_string(),
        absent.display().to_string(),
        "--output".to_string(),
        output.display().to_string(),
        "--brama-model".to_string(),
        "chat-only-alias".to_string(),
    ];
    let stderr = refusal(&run.invoke(&args, &secrets(), &[]));
    assert!(
        stderr.contains("`chat-only-alias` cannot answer a typed choice"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("cannot read goal predictions"),
        "the alias is refused before the input is read: {stderr}"
    );
    assert!(!output.exists(), "a refused audit writes no record");
    run.passed();
}

/// An OpenAI-compatible service on loopback answering every request with one
/// chat completion, reporting each request line and body it received.
fn endpoint() -> (String, mpsc::Receiver<(String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (sender, received) = mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let mut length = 0;
            let mut header = String::new();
            while reader.read_line(&mut header).unwrap() > 2 {
                if let Some((name, value)) = header.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                }
                header.clear();
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let answer = json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]})
                .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                answer.len()
            )
            .unwrap();
            let _ = sender.send((
                request.trim().to_owned(),
                String::from_utf8_lossy(&body).into_owned(),
            ));
        }
    });
    (origin, received)
}

#[test]
fn an_endpoint_declared_in_bramas_place_is_asked_when_brama_does_not_answer() {
    let mut run = Run::start("stado-alternative");
    let (origin, received) = endpoint();
    // The isolated home places Brama nowhere; the endpoint declared in its
    // place answers, and the audit asks it for its own model, unsigned.
    let env = vec![
        ("TLT_STADO_BIN", stado()),
        ("STADO_INFERENCE_ENDPOINT_URL", origin.clone()),
        (
            "STADO_INFERENCE_ENDPOINT_MODEL",
            "endpoint-model".to_string(),
        ),
    ];
    run.audit(
        &env,
        &[
            "WISENT_APP_AGENT_AUTH_SECRET",
            "BRAMA_TOKEN",
            "WISENT_APP_AGENT_ID",
        ],
    );
    let asked: Vec<(String, String)> = received.try_iter().collect();
    assert!(
        !asked.iter().any(|(line, _)| line.contains("/v1/aliases")),
        "Brama's own measurement is not asked of the endpoint: {asked:?}"
    );
    let chats: Vec<&String> = asked
        .iter()
        .filter(|(line, _)| line.starts_with("POST /v1/chat/completions "))
        .map(|(_, body)| body)
        .collect();
    assert!(
        !chats.is_empty(),
        "the endpoint itself was asked: {asked:?}"
    );
    for body in chats {
        let request: Value = serde_json::from_str(body).unwrap();
        assert_eq!(request["model"], "endpoint-model", "{request}");
    }
    let record = fs::read_to_string(run.home.join(".stado/capabilities.json")).unwrap();
    assert!(
        record.contains("model-endpoint"),
        "Stado records the endpoint as the one that served: {record}"
    );
    run.passed();
}
