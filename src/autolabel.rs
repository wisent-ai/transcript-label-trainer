//! Automatic labeling by a Brama teacher — operator-mandated zero-touch.
//!
//! For each lake session with no label on the aspect yet, a Brama-routed model
//! classifies the reconstructed session text into one allowed value, and the
//! result is applied immediately through the lake CLI
//! (`label add --source brama:<model>`). The lake validates sessions and owns
//! the write; what there is not, by design, is a human review step.
//!
//! Human labels are never overwritten: a session already labeled with the
//! aspect by ANY source is skipped, so reruns are idempotent.
use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::util::Result;
use crate::{brama, jobs, lake};

pub const AUTOLABEL_NOTE: &str = "autolabel";

fn record(session_id: &str, key: &str, value: String) -> Value {
    let mut entry = Map::new();
    entry.insert("session_id".to_string(), Value::String(session_id.to_string()));
    entry.insert(key.to_string(), Value::String(value));
    Value::Object(entry)
}

/// Label every unlabeled session for an aspect via the Brama teacher.
///
/// Returns the summary (labeled / skipped_labeled / failed counts with
/// per-session results and failures).
pub fn autolabel(
    aspect: &str,
    values: &[String],
    brama_model: Option<&str>,
    limit: Option<i64>,
    runtime: Option<&str>,
) -> Result<Value> {
    let model_id = brama_model
        .filter(|value| !value.is_empty())
        .unwrap_or(brama::DEFAULT_MODEL)
        .to_string();
    let source = format!("brama:{model_id}");
    let client = brama::BramaClient::from_env()?;

    // Latest label per session, any source: a session anyone has already
    // labeled on this aspect is never relabeled.
    let already: HashSet<String> =
        lake::load_labels(aspect)?.into_iter().map(|label| label.session_id).collect();
    let mut sessions = lake::all_sessions()?;
    if let Some(runtime) = runtime {
        sessions.retain(|session| session.runtime.as_deref() == Some(runtime));
    }
    let skipped_labeled =
        sessions.iter().filter(|session| already.contains(&session.session_id)).count();
    let mut targets: Vec<String> = sessions
        .into_iter()
        .filter(|session| !already.contains(&session.session_id))
        .map(|session| session.session_id)
        .collect();
    if let Some(limit) = limit {
        targets.truncate(limit.max(0) as usize);
    }

    let texts = lake::session_texts(&targets)?;
    let mut results: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut no_text = 0usize;
    for session_id in &targets {
        let Some(entry) = texts.get(session_id) else {
            no_text += 1;
            continue;
        };
        if entry.text.trim().is_empty() {
            no_text += 1;
            continue;
        }
        let prompt = brama::build_prompt(aspect, values, &entry.text);
        let answer = match client.chat(&model_id, &prompt) {
            Ok(answer) => answer,
            Err(error) => {
                failures.push(record(session_id, "error", brama::truncate_chars(&error.0, 200)));
                continue;
            }
        };
        let Some((value, _exact)) = brama::parse_answer(&answer, values) else {
            failures.push(record(
                session_id,
                "error",
                format!(
                    "unparseable answer: {}",
                    jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                ),
            ));
            continue;
        };
        // The lake CLI validates the session and owns the write; that boundary
        // stays. A refusal fails this one session and nothing else.
        if let Err(error) = lake::label_add(session_id, aspect, &value, &source, AUTOLABEL_NOTE) {
            failures.push(record(session_id, "error", brama::truncate_chars(&error.0, 200)));
            continue;
        }
        results.push(record(session_id, "value", value));
    }

    let mut summary = Map::new();
    summary.insert("aspect".to_string(), Value::String(aspect.to_string()));
    summary.insert("brama_model".to_string(), Value::String(model_id));
    summary.insert("source".to_string(), Value::String(source));
    summary.insert("labeled".to_string(), Value::Number(results.len().into()));
    summary.insert("skipped_labeled".to_string(), Value::Number(skipped_labeled.into()));
    summary.insert("skipped_no_text".to_string(), Value::Number(no_text.into()));
    summary.insert("failed".to_string(), Value::Number(failures.len().into()));
    summary.insert("results".to_string(), Value::Array(results));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}
