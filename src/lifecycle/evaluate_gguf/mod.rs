//! `lifecycle-evaluate-gguf`: the quantized lifecycle model measured the way
//! production serves it — `llama-server` answering Oko's chat contract with
//! decoding constrained to the checked-in decision schema, narrowed per
//! request to that request's candidate references.
//!
//! Nothing here waits by the clock. The server is ready when its health
//! answer is ready; between probes the evaluator reads the server's next log
//! line, so it waits on the server's own progress, and a server that exits
//! first is a failure naming its status and log. Each request waits for its
//! answer; a request that fails is a failure naming its row, never retried on
//! a guessed schedule. The answer's length is bounded by the schema's grammar,
//! not by a token budget.

use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Map};

use super::*;

/// What one evaluation run is given; every field is the caller's.
pub struct GgufEvaluation {
    pub model: PathBuf,
    pub dataset: PathBuf,
    pub predictions: PathBuf,
    pub metrics: PathBuf,
    pub server: PathBuf,
    pub server_log: PathBuf,
    pub output_schema: PathBuf,
    pub parallel: usize,
    pub slot_context: usize,
    pub gpu_layers: String,
}

/// The served alias every request names.
fn alias(model: &Path) -> String {
    model.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default()
}

/// An HTTP client that waits for the server's answer: reqwest's blocking
/// client otherwise gives up after its own 30 s.
fn waiting_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder().timeout(None).build().map_err(|error| Error(format!("the HTTP client could not be built: {error}")))
}

/// A loopback port the operating system has free right now.
fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?.local_addr()?.port())
}

/// The checked-in schema with `goal_ref` narrowed from its pattern to `refs`.
fn narrowed(schema: &Value, refs: &[String]) -> Result<Value> {
    let variants = schema["oneOf"].as_array().ok_or("the output schema has no oneOf")?;
    let mut narrowed = Vec::with_capacity(variants.len());
    for variant in variants {
        let mut variant = variant.clone();
        let goal_ref = variant
            .pointer_mut("/properties/goal_ref")
            .and_then(Value::as_object_mut)
            .ok_or("an output schema variant has no goal_ref property")?;
        if goal_ref.remove("pattern").is_some() {
            goal_ref.insert("enum".to_string(), json!(refs));
        }
        narrowed.push(variant);
    }
    Ok(json!({ "oneOf": narrowed }))
}

/// The candidate references of `row` the schema's pattern variants may name:
/// every candidate except the references the schema itself fixes (a new goal).
fn existing_refs(row: &TrainingRow, schema: &Value) -> Result<Vec<String>> {
    let fixed: Vec<&str> = schema["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|variant| variant.pointer("/properties/goal_ref/const").and_then(Value::as_str))
        .collect();
    let envelope = input_envelope(row)?;
    let candidates = envelope["candidates"].as_array().ok_or_else(|| Error(format!("{} has no candidates", row.id)))?;
    Ok(candidates
        .iter()
        .filter_map(|candidate| candidate["ref"].as_str())
        .filter(|reference| !fixed.contains(reference))
        .map(str::to_string)
        .collect())
}

/// Whether `value` satisfies one narrowed variant: exactly its required
/// string properties, each matching its `const` or `enum`.
fn satisfies(value: &Map<String, Value>, variant: &Value) -> bool {
    let Some(properties) = variant["properties"].as_object() else { return false };
    if value.len() != properties.len() || !value.keys().all(|key| properties.contains_key(key)) {
        return false;
    }
    properties.iter().all(|(key, rule)| {
        let Some(actual) = value.get(key) else { return false };
        if !actual.is_string() || rule.get("pattern").is_some() {
            return false;
        }
        rule.get("const").map_or(true, |constant| constant == actual)
            && rule.get("enum").and_then(Value::as_array).map_or(true, |allowed| allowed.contains(actual))
    })
}

/// The decision in `text` when it satisfies one variant of `schema`.
fn decision(text: &str, schema: &Value) -> Option<Decision> {
    let value = parse_json_object(text).ok()?;
    let object = value.as_object()?;
    schema["oneOf"].as_array()?.iter().any(|variant| satisfies(object, variant)).then(|| serde_json::from_value(value.clone()).ok())?
}

/// The reference decision of `row`, its title blanked: titles belong to Oko's title model.
fn target(row: &TrainingRow, schema: &Value) -> Result<Decision> {
    let content = &row.messages.iter().find(|message| message.role == "assistant").ok_or_else(|| Error(format!("{} has no assistant message", row.id)))?.content;
    let mut value = parse_json_object(content)?;
    value["title"] = json!("");
    decision(&value.to_string(), schema).ok_or_else(|| Error(format!("{} has a reference decision outside the output schema", row.id)))
}

/// Start the server and return once its health answer is ready.
fn start_server(run: &GgufEvaluation, port: u16) -> Result<Child> {
    let mut child = Command::new(&run.server)
        .arg("--model")
        .arg(&run.model)
        .args(["--host", &Ipv4Addr::LOCALHOST.to_string(), "--port", &port.to_string(), "--alias", &alias(&run.model)])
        .args(["--ctx-size", &(run.slot_context * run.parallel).to_string(), "--gpu-layers", &run.gpu_layers])
        .args(["--parallel", &run.parallel.to_string(), "--no-webui"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error(format!("{} could not start: {error}", run.server.display())))?;
    let mut log = fs::File::create(&run.server_log)?;
    let stderr = child.stderr.take().ok_or("llama-server has no stderr")?;
    let stdout = child.stdout.take().ok_or("llama-server has no stdout")?;
    let mut copy_out = fs::File::create(run.server_log.with_extension("stdout.log"))?;
    thread::spawn(move || std::io::copy(&mut BufReader::new(stdout), &mut copy_out));
    let health = format!("http://{}:{port}/health", Ipv4Addr::LOCALHOST);
    let client = waiting_client()?;
    let mut lines = BufReader::new(stderr).lines();
    loop {
        if client.get(&health).send().map(|response| response.status().is_success()).unwrap_or(false) {
            thread::spawn(move || {
                for line in lines.map_while(std::result::Result::ok) {
                    let _ = writeln!(log, "{line}");
                }
            });
            return Ok(child);
        }
        match lines.next() {
            Some(line) => writeln!(log, "{}", line?)?,
            None => {
                let status = child.wait()?;
                return Err(Error(format!("llama-server exited {status} before it was ready; its log is {}", run.server_log.display())));
            }
        }
    }
}

/// Ask the server for `row`'s decision; `(raw answer, decision when it satisfies the schema)`.
fn classify(client: &reqwest::blocking::Client, endpoint: &str, alias: &str, schema: &Value, row: &TrainingRow) -> Result<(String, Option<Decision>)> {
    let user = &row.messages.iter().find(|message| message.role == "user").ok_or_else(|| Error(format!("{} has no user message", row.id)))?.content;
    let narrowed = narrowed(schema, &existing_refs(row, schema)?)?;
    let body = json!({
        "model": alias,
        "messages": [{ "role": "system", "content": SYSTEM_PROMPT.trim() }, { "role": "user", "content": user }],
        "temperature": 0,
        "stream": false,
        "chat_template_kwargs": { "enable_thinking": false },
        "response_format": { "type": "json_schema", "json_schema": { "name": "oko_goal_lifecycle", "strict": true, "schema": narrowed } },
    });
    let response = client.post(endpoint).json(&body).send().map_err(|error| Error(format!("{} inference failed: {error}", row.id)))?;
    let status = response.status();
    let answer: Value = response.json().map_err(|error| Error(format!("{} answered {status} without JSON: {error}", row.id)))?;
    let raw = answer.pointer("/choices/0/message/content").and_then(Value::as_str).ok_or_else(|| Error(format!("{} answered {status} without a message: {answer}", row.id)))?.trim().to_string();
    let parsed = decision(&raw, &narrowed);
    Ok((raw, parsed))
}

mod scoring;
pub use scoring::evaluate_gguf;
