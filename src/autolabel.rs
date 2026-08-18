//! Automatic labeling by a Brama teacher — operator-mandated zero-touch.
//!
//! For each lake session with no label on the aspect yet, a Brama-routed model
//! classifies the reconstructed session text into one allowed value. With
//! `--best`, Brama's independent `best` route must first call that proposal
//! sensible; only then is it applied through the lake CLI. The lake validates
//! sessions and owns the write; there is no human staging queue.
//!
//! Human labels are never overwritten: a session already labeled with the
//! aspect by ANY source is skipped, so reruns are idempotent.
use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::util::Result;
use crate::{brama, jobs, lake};

pub const AUTOLABEL_NOTE: &str = "autolabel";
const BEST_REVIEW_NOTE: &str = "autolabel; reviewed=best";
const LABEL_REVIEW_VALUES: [&str; 2] = ["sensible", "nonsensical"];

fn label_review_prompt(
    aspect: &str,
    values: &[String],
    proposed: &str,
    text: &str,
) -> Vec<brama::Message> {
    vec![
        brama::Message {
            role: "system".to_string(),
            content: "You audit proposed aspect labels for coding-agent session \
                      transcripts. Answer with exactly one word: sensible or \
                      nonsensical."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}\nAllowed values: {}\nProposed label: {proposed}\n\n\
                 Is that proposed label a sensible reading of this transcript \
                 for the declared aspect? Answer sensible or nonsensical.\n\n\
                 Transcript:\n{text}",
                values.join(", ")
            ),
        },
    ]
}

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
/// per-session results and failures).
pub fn autolabel(
    aspect: &str,
    values: &[String],
    brama_model: Option<&str>,
    best_review: bool,
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
        targets.truncate(limit.max(0) as usize);
    }

    let texts = lake::session_texts(&targets)?;
    let mut results: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    let mut review_failed = 0usize;
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
                failures.push(record(
                    session_id,
                    "error",
                    brama::truncate_chars(&error.0, 200),
                ));
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
        let mut review = None;
        if best_review {
            let prompt = label_review_prompt(aspect, values, &value, &entry.text);
            let answer = match client.chat(brama::BEST_MODEL, &prompt) {
                Ok(answer) => answer,
                Err(error) => {
                    review_failed += 1;
                    failures.push(record(
                        session_id,
                        "error",
                        format!(
                            "final label review failed: {}",
                            brama::truncate_chars(&error.0, 160)
                        ),
                    ));
                    continue;
                }
            };
            let Some((verdict, _exact)) = brama::parse_answer(&answer, &LABEL_REVIEW_VALUES[..])
            else {
                review_failed += 1;
                failures.push(record(
                    session_id,
                    "error",
                    format!(
                        "unparseable final label review: {}",
                        jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                    ),
                ));
                continue;
            };
            if verdict == LABEL_REVIEW_VALUES[1] {
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
            failures.push(record(
                session_id,
                "error",
                brama::truncate_chars(&error.0, 200),
            ));
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
    summary.insert("source".to_string(), Value::String(source));
    summary.insert("labeled".to_string(), Value::Number(results.len().into()));
    summary.insert(
        "skipped_labeled".to_string(),
        Value::Number(skipped_labeled.into()),
    );
    summary.insert("skipped_no_text".to_string(), Value::Number(no_text.into()));
    summary.insert("failed".to_string(), Value::Number(failures.len().into()));
    summary.insert(
        "best_review".to_string(),
        serde_json::json!({
            "enabled": best_review,
            "model": if best_review {
                Value::String(brama::BEST_MODEL.to_string())
            } else {
                Value::Null
            },
            "accepted": results.len(),
            "rejected_nonsensical": rejected.len(),
            "failed": review_failed,
            "sensible": !best_review || (rejected.is_empty() && review_failed == 0),
        }),
    );
    summary.insert("results".to_string(), Value::Array(results));
    summary.insert("rejected".to_string(), Value::Array(rejected));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}
