//! Automatic labeling by a Brama teacher — operator-mandated zero-touch.
//!
//! For each lake session with no label on the aspect yet, Brama's decision
//! endpoint chooses one allowed value for the reconstructed session text. With
//! `--best`, Brama's independent `best` route must first call that proposal
//! sensible; only then is it applied through the lake CLI. The lake validates
//! sessions and owns the write; there is no human staging queue.
//!
//! Human labels are never overwritten: a session already labeled with the
//! aspect by ANY source is skipped, so reruns are idempotent.
use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::bail;
use crate::util::{Error, Result};
use crate::{brama, lake};

pub const AUTOLABEL_NOTE: &str = "autolabel";
const BEST_REVIEW_NOTE: &str = "autolabel; reviewed=best";
const SENSIBLE: &str = "sensible";
const NONSENSICAL: &str = "nonsensical";

fn record(session_id: &str, key: &str, value: String) -> Value {
    let mut entry = Map::new();
    entry.insert(
        "session_id".to_string(),
        Value::String(session_id.to_string()),
    );
    entry.insert(key.to_string(), Value::String(value));
    Value::Object(entry)
}

/// Label every unlabeled session for an aspect via the Brama teacher.
///
/// Returns the summary (labeled / skipped_labeled / failed counts with
/// per-session results and failures). The teacher is `brama_model` when it
/// is given and Brama's `best` otherwise; it answers through the decision
/// endpoint, so an alias that cannot answer a typed choice is refused before
/// any session is read.
pub fn autolabel(
    aspect: &str,
    values: &[String],
    brama_model: Option<&str>,
    best_review: bool,
    limit: Option<i64>,
    runtime: Option<&str>,
) -> Result<Value> {
    let model_id = match brama_model {
        Some(model) if model.trim().is_empty() => {
            bail!("--brama-model is empty; name a Brama alias or leave it out")
        }
        Some(model) => model.trim().to_string(),
        None => brama::DEFAULT_MODEL.to_string(),
    };
    let teacher = brama::decision_alias(&model_id)?;
    let limit = match limit {
        Some(limit) => Some(usize::try_from(limit).map_err(|error| {
            Error(format!(
                "--limit must be a count of sessions, not {limit}: {error}"
            ))
        })?),
        None => None,
    };
    let source = format!("brama:{model_id}");
    let client = brama::BramaClient::from_env()?;
    let labels: Vec<(&str, &str)> = values
        .iter()
        .map(|value| (value.as_str(), value.as_str()))
        .collect();

    // Latest label per session, any source: a session anyone has already
    // labeled on this aspect is never relabeled.
    let already: HashSet<String> = lake::load_labels(aspect)?
        .into_iter()
        .map(|label| label.session_id)
        .collect();
    let mut sessions = lake::all_sessions()?;
    if let Some(runtime) = runtime {
        sessions.retain(|session| session.runtime.as_deref() == Some(runtime));
    }
    let skipped_labeled = sessions
        .iter()
        .filter(|session| already.contains(&session.session_id))
        .count();
    let mut targets: Vec<String> = sessions
        .into_iter()
        .filter(|session| !already.contains(&session.session_id))
        .map(|session| session.session_id)
        .collect();
    if let Some(limit) = limit {
        targets.truncate(limit);
    }

    let texts = lake::session_texts(&targets)?;
    let mut results: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    let mut review_failed = Vec::<String>::new();
    let mut no_text = Vec::<String>::new();
    for session_id in &targets {
        let Some(entry) = texts
            .get(session_id)
            .filter(|entry| !entry.text.trim().is_empty())
        else {
            no_text.push(session_id.clone());
            continue;
        };
        let instructions = format!(
            "Classify this coding-agent session transcript for the aspect `{aspect}`: choose the \
             one allowed value that describes it."
        );
        let value = match client.choose(teacher, &entry.text, &instructions, &labels) {
            Ok(value) => value,
            Err(error) => {
                failures.push(record(session_id, "error", error.0.clone()));
                continue;
            }
        };
        let mut review = None;
        if best_review {
            let instructions = format!(
                "The proposed label for the aspect `{aspect}` (allowed values: {}) is `{value}`. \
                 Is it a sensible reading of this coding-agent session transcript?",
                values.join(", ")
            );
            let verdicts = [
                (
                    SENSIBLE,
                    "the proposed label is a sensible reading of the transcript for the aspect",
                ),
                (
                    NONSENSICAL,
                    "the proposed label is not a sensible reading of the transcript for the aspect",
                ),
            ];
            let verdict = match client.choose(
                brama::BEST_DECISION_MODEL,
                &entry.text,
                &instructions,
                &verdicts,
            ) {
                Ok(verdict) => verdict,
                Err(error) => {
                    review_failed.push(session_id.clone());
                    failures.push(record(
                        session_id,
                        "error",
                        format!("final label review failed: {}", error.0),
                    ));
                    continue;
                }
            };
            if verdict == NONSENSICAL {
                let mut rejected_label = Map::new();
                rejected_label.insert(
                    "session_id".to_string(),
                    Value::String(session_id.to_string()),
                );
                rejected_label.insert("value".to_string(), Value::String(value));
                rejected_label.insert("best_review".to_string(), Value::String(verdict));
                rejected.push(Value::Object(rejected_label));
                continue;
            }
            review = Some(verdict);
        }
        // The lake CLI validates the session and owns the write; that boundary
        // stays. A refusal fails this one session and nothing else.
        let note = if best_review {
            BEST_REVIEW_NOTE
        } else {
            AUTOLABEL_NOTE
        };
        if let Err(error) = lake::label_add(session_id, aspect, &value, &source, note) {
            failures.push(record(session_id, "error", error.0.clone()));
            continue;
        }
        let mut labeled = Map::new();
        labeled.insert(
            "session_id".to_string(),
            Value::String(session_id.to_string()),
        );
        labeled.insert("value".to_string(), Value::String(value));
        if let Some(review) = review {
            labeled.insert("best_review".to_string(), Value::String(review));
        }
        results.push(Value::Object(labeled));
    }

    let mut summary = Map::new();
    summary.insert("aspect".to_string(), Value::String(aspect.to_string()));
    summary.insert("brama_model".to_string(), Value::String(model_id));
    summary.insert(
        "decision_model".to_string(),
        Value::String(teacher.to_string()),
    );
    summary.insert("source".to_string(), Value::String(source));
    summary.insert("labeled".to_string(), serde_json::json!(results.len()));
    summary.insert(
        "skipped_labeled".to_string(),
        serde_json::json!(skipped_labeled),
    );
    summary.insert(
        "skipped_no_text".to_string(),
        serde_json::json!(no_text.len()),
    );
    summary.insert("failed".to_string(), serde_json::json!(failures.len()));
    summary.insert(
        "best_review".to_string(),
        serde_json::json!({
            "enabled": best_review,
            "model": if best_review {
                Value::String(brama::BEST_DECISION_MODEL.to_string())
            } else {
                Value::Null
            },
            "accepted": results.len(),
            "rejected_nonsensical": rejected.len(),
            "failed": review_failed.len(),
            "sensible": !best_review || (rejected.is_empty() && review_failed.is_empty()),
        }),
    );
    summary.insert("results".to_string(), Value::Array(results));
    summary.insert("rejected".to_string(), Value::Array(rejected));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}
