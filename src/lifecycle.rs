//! Oko lifecycle-model dataset curation and independent semantic audits.
//!
//! Input rows are masked Oko training envelopes. Brama owns every model call;
//! this module replaces historical silver answers with reviewed decisions and
//! records the exact route used for provenance.

use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::brama::{BramaClient, Message};
use crate::util::{now_iso, Error, Result};

const SYSTEM_PROMPT: &str = include_str!("../training/lifecycle-model/lifecycle-system-prompt.txt");
const WORKERS: usize = 16;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TrainingRow {
    id: String,
    #[serde(default)]
    split_day: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    metadata: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Decision {
    action: String,
    goal_ref: String,
    title: String,
    lifecycle_evidence: String,
}

fn read_rows(path: &Path) -> Result<Vec<TrainingRow>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            serde_json::from_str(&line).map_err(Error::from)
        })
        .collect()
}

fn parse_json_object(answer: &str) -> Result<Value> {
    let trimmed = answer.trim();
    let candidate = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        trimmed
    } else {
        let start = trimmed
            .find('{')
            .ok_or_else(|| Error("lifecycle reviewer returned no JSON object".to_string()))?;
        let end = trimmed
            .rfind('}')
            .ok_or_else(|| Error("lifecycle reviewer returned incomplete JSON".to_string()))?;
        &trimmed[start..=end]
    };
    serde_json::from_str(candidate).map_err(Error::from)
}

fn input_envelope(row: &TrainingRow) -> Result<Value> {
    let content = row
        .messages
        .iter()
        .find(|message| message.role == "user")
        .ok_or_else(|| Error(format!("{} has no user message", row.id)))?
        .content
        .trim();
    serde_json::from_str(content)
        .map_err(|error| Error(format!("{} has invalid user envelope: {error}", row.id)))
}

fn validate_decision(row: &TrainingRow, value: Value) -> Result<Decision> {
    let mut decision: Decision = serde_json::from_value(value)
        .map_err(|error| Error(format!("{} has invalid reviewer decision: {error}", row.id)))?;
    let envelope = input_envelope(row)?;
    let refs = envelope
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|candidate| candidate.get("ref").and_then(Value::as_str))
        .collect::<Vec<_>>();
    if !["startGoal", "continueCurrent", "finishGoal", "ignore"].contains(&decision.action.as_str())
    {
        return Err(Error(format!(
            "{} has unknown action {}",
            row.id, decision.action
        )));
    }
    if !refs.contains(&decision.goal_ref.as_str()) {
        return Err(Error(format!(
            "{} has unknown goal_ref {}",
            row.id, decision.goal_ref
        )));
    }
    if !["none", "explicit_open", "explicit_completion"]
        .contains(&decision.lifecycle_evidence.as_str())
    {
        return Err(Error(format!(
            "{} has unknown lifecycle_evidence {}",
            row.id, decision.lifecycle_evidence
        )));
    }
    decision.title.clear();
    if decision.action == "startGoal" {
        if decision.goal_ref != "NEW_GOAL" {
            return Err(Error(format!("{} has invalid startGoal ref", row.id)));
        }
    } else if decision.goal_ref == "NEW_GOAL" {
        return Err(Error(format!(
            "{} uses NEW_GOAL for {}",
            row.id, decision.action
        )));
    }
    if decision.action == "finishGoal" && decision.lifecycle_evidence != "explicit_completion" {
        return Err(Error(format!(
            "{} finishGoal lacks explicit completion evidence",
            row.id
        )));
    }
    if decision.lifecycle_evidence == "explicit_completion" && decision.action != "finishGoal" {
        return Err(Error(format!(
            "{} completion evidence has non-finish action",
            row.id
        )));
    }
    Ok(decision)
}

fn classify(client: &BramaClient, model: &str, row: &TrainingRow) -> Result<Decision> {
    let envelope = input_envelope(row)?;
    let request = [
        Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::to_string(&envelope)?,
        },
    ];
    let mut last = Error("lifecycle reviewer did not run".to_string());
    for attempt in 0..3 {
        match client
            .chat(model, &request)
            .and_then(|answer| parse_json_object(&answer))
            .and_then(|value| validate_decision(row, value))
        {
            Ok(decision) => return Ok(decision),
            Err(error) => last = error,
        }
        thread::sleep(Duration::from_secs(1 << attempt));
    }
    Err(Error(format!("{}: {last}", row.id)))
}

fn reviewed_row(
    mut row: TrainingRow,
    decision: Decision,
    model: &str,
    split: &str,
) -> Result<Value> {
    row.messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.trim().to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: serde_json::to_string(&input_envelope(&row)?)?,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: serde_json::to_string(&decision)?,
        },
    ];
    let mut value = serde_json::to_value(row)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error("serialized lifecycle row is not an object".to_string()))?;
    object.insert("split".to_string(), Value::String(split.to_string()));
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Default::default()));
    let metadata = metadata
        .as_object_mut()
        .ok_or_else(|| Error("lifecycle metadata is not an object".to_string()))?;
    metadata.insert("reviewedBy".to_string(), Value::String(model.to_string()));
    metadata.insert("reviewedAt".to_string(), Value::String(now_iso()));
    metadata.insert(
        "contract".to_string(),
        Value::String("oko-goal-lifecycle-v1".to_string()),
    );
    Ok(value)
}
fn reviewed_ids(path: &Path) -> Result<HashSet<String>> {
    if !path.is_file() {
        return Ok(HashSet::new());
    }
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            let value: Value = serde_json::from_str(&line)?;
            value
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| Error("reviewed lifecycle row has no id".to_string()))
        })
        .collect()
}

pub fn review_dataset(
    input: &Path,
    output: &Path,
    split: &str,
    model: &str,
    limit: Option<usize>,
) -> Result<Value> {
    if !["train", "eval"].contains(&split) {
        return Err(Error("--split must be train or eval".to_string()));
    }
    let mut rows = read_rows(input)?;
    if let Some(limit) = limit {
        rows.truncate(limit);
    }
    if rows.is_empty() {
        return Err(Error("lifecycle dataset input is empty".to_string()));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let existing_ids = reviewed_ids(output)?;
    let existing = existing_ids.len();
    rows.retain(|row| !existing_ids.contains(&row.id));
    if rows.is_empty() {
        return Ok(serde_json::json!({
            "input": input,
            "output": output,
            "split": split,
            "review_model": model,
            "rows": existing,
            "resumed": existing,
            "contract": "oko-goal-lifecycle-v1"
        }));
    }
    let client = BramaClient::from_env()?;
    let rows = Arc::new(rows);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<Option<Result<Value>>>>> =
        Arc::new(Mutex::new((0..rows.len()).map(|_| None).collect()));
    let workers = env::var("LIFECYCLE_REVIEW_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(WORKERS)
        .min(rows.len());
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let client = client.clone();
        let rows = Arc::clone(&rows);
        let next = Arc::clone(&next);
        let results = Arc::clone(&results);
        let model = model.to_string();
        let split = split.to_string();
        handles.push(thread::spawn(move || loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= rows.len() {
                break;
            }
            let row = rows[index].clone();
            let result = classify(&client, &model, &row)
                .and_then(|decision| reviewed_row(row, decision, &model, &split));
            results.lock().expect("lifecycle result lock")[index] = Some(result);
        }));
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| Error("lifecycle review worker panicked".to_string()))?;
    }
    let file = OpenOptions::new().create(true).append(true).open(output)?;
    let mut writer = BufWriter::new(file);
    let mut counts = serde_json::Map::new();
    let mut written = 0usize;
    let mut failures = Vec::new();
    for result in results.lock().expect("lifecycle result lock").iter_mut() {
        let Some(result) = result.take() else {
            failures.push("lifecycle review produced a missing row".to_string());
            continue;
        };
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                failures.push(error.to_string());
                continue;
            }
        };
        let action = value
            .pointer("/messages/2/content")
            .and_then(Value::as_str)
            .and_then(|content| serde_json::from_str::<Value>(content).ok())
            .and_then(|decision| {
                decision
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "invalid".to_string());
        let count = counts.entry(action).or_insert(Value::from(0));
        *count = Value::from(count.as_u64().unwrap_or(0) + 1);
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
        written += 1;
    }
    writer.flush()?;
    if !failures.is_empty() {
        return Err(Error(format!(
            "lifecycle review failed for {} row(s); preserved {} newly reviewed row(s) in {}: {}",
            failures.len(),
            written,
            output.display(),
            failures[0]
        )));
    }
    Ok(serde_json::json!({
        "input": input,
        "output": output,
        "split": split,
        "review_model": model,
        "rows": existing + written,
        "resumed": existing,
        "actions": counts,
        "contract": "oko-goal-lifecycle-v1"
    }))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AuditPrediction {
    id: String,
    input: Value,
    target: Decision,
    prediction: Option<Decision>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AuditDecision {
    verdict: String,
    dangerous_finish: bool,
}

fn read_predictions(path: &Path) -> Result<Vec<AuditPrediction>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            serde_json::from_str(&line).map_err(Error::from)
        })
        .collect()
}

fn retryable_audit_error(error: &Error) -> bool {
    error.0.contains("\"retryable\":true")
        || ["HTTP 429", "HTTP 503", "HTTP 504"]
            .iter()
            .any(|status| error.0.contains(status))
}

fn audit_one(
    client: &BramaClient,
    model: &str,
    prediction: &AuditPrediction,
) -> Result<AuditDecision> {
    let request = [
        Message {
            role: "system".to_string(),
            content: format!(
                "You are the independent final judge for Oko's goal-lifecycle model. \
                 Read the contract and the masked input. Decide whether the student decision \
                 is semantically safe and defensible. A different candidate reference is wrong \
                 when it points at a different goal. Any inferred completion without explicit \
                 successful-completion evidence is dangerous. Start-goal title wording is not \
                 lifecycle correctness: Oko replaces it with its separate title model, so ignore \
                 title differences when the student title satisfies the output contract. The \
                 reference decision is evidence, not authority. Return only strict JSON: \
                 {{\"verdict\":\"student-sensible|student-wrong|unjudgeable\",\
                 \"dangerous_finish\":false}}.\n\n{SYSTEM_PROMPT}"
            ),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::to_string(&serde_json::json!({
                "input": prediction.input,
                "reference": prediction.target,
                "student": prediction.prediction,
            }))?,
        },
    ];
    let mut last = Error("lifecycle audit did not run".to_string());
    for attempt in 0..3 {
        match client
            .chat(model, &request)
            .and_then(|answer| parse_json_object(&answer))
            .and_then(|value| serde_json::from_value::<AuditDecision>(value).map_err(Error::from))
        {
            Ok(decision)
                if ["student-sensible", "student-wrong", "unjudgeable"]
                    .contains(&decision.verdict.as_str()) =>
            {
                return Ok(decision);
            }
            Ok(_) => last = Error(format!("{} has an unknown audit verdict", prediction.id)),
            Err(error) => last = error,
        }
        if attempt == 2 || !retryable_audit_error(&last) {
            break;
        }
        thread::sleep(Duration::from_secs(1 << attempt));
    }
    Err(Error(format!("{}: {last}", prediction.id)))
}

pub fn audit_predictions(input: &Path, output: &Path, model: &str) -> Result<Value> {
    let predictions = read_predictions(input)?;
    if predictions.is_empty() {
        return Err(Error("lifecycle predictions input is empty".to_string()));
    }
    let client = BramaClient::from_env()?;
    let predictions = Arc::new(predictions);
    let next = Arc::new(AtomicUsize::new(0));
    let aborted = Arc::new(AtomicBool::new(false));
    let results: Arc<Mutex<Vec<Option<Result<AuditDecision>>>>> =
        Arc::new(Mutex::new((0..predictions.len()).map(|_| None).collect()));
    // The local Brama route accounts each request as two concurrency units and the
    // operator plan currently exposes four. More workers make a healthy route
    // deterministically return 429 and abort the entire final audit.
    let workers = usize::from(2_u8).min(predictions.len());
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let client = client.clone();
        let predictions = Arc::clone(&predictions);
        let next = Arc::clone(&next);
        let aborted = Arc::clone(&aborted);
        let results = Arc::clone(&results);
        let model = model.to_string();
        handles.push(thread::spawn(move || loop {
            if aborted.load(Ordering::Relaxed) {
                break;
            }
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= predictions.len() {
                break;
            }
            let result = audit_one(&client, &model, &predictions[index]);
            let failed = result.is_err();
            results.lock().expect("lifecycle audit result lock")[index] = Some(result);
            if failed {
                aborted.store(true, Ordering::Relaxed);
            }
        }));
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| Error("lifecycle audit worker panicked".to_string()))?;
    }

    let mut records = Vec::with_capacity(predictions.len());
    let mut sensible = 0usize;
    let mut wrong = 0usize;
    let mut unjudgeable = 0usize;
    let mut dangerous_finish = 0usize;
    let mut failures = Vec::new();
    for (index, result) in results
        .lock()
        .expect("lifecycle audit result lock")
        .iter_mut()
        .enumerate()
    {
        match result.take() {
            Some(Ok(decision)) => {
                match decision.verdict.as_str() {
                    "student-sensible" => sensible += 1,
                    "student-wrong" => wrong += 1,
                    _ => unjudgeable += 1,
                }
                dangerous_finish += usize::from(decision.dangerous_finish);
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "decision": decision,
                }));
            }
            Some(Err(error)) => {
                failures.push(error.to_string());
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "error": error.to_string(),
                }));
            }
            None => {
                let error = "audit skipped after an earlier audit failure".to_string();
                failures.push(error.clone());
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "error": error,
                }));
            }
        }
    }
    let maximum_wrong = predictions.len() / 50;
    let passed =
        wrong <= maximum_wrong && unjudgeable == 0 && dangerous_finish == 0 && failures.is_empty();
    let report = serde_json::json!({
        "created_at": now_iso(),
        "review_model": model,
        "input": input,
        "counts": {
            "total": predictions.len(),
            "student_sensible": sensible,
            "student_wrong": wrong,
            "unjudgeable": unjudgeable,
            "dangerous_finish": dangerous_finish,
            "audit_errors": failures.len(),
        },
        "thresholds": {
            "maximum_student_wrong": maximum_wrong,
            "maximum_unjudgeable": 0,
            "maximum_dangerous_finish": 0,
            "maximum_audit_errors": 0,
        },
        "passed": passed,
        "failures": failures,
        "records": records,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("json.{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&report)?)?;
    fs::rename(temporary, output)?;
    Ok(report)
}
