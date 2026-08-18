//! Aspect discovery — a Brama teacher reads recent masked sessions and
//! proposes aspect dimensions grounded in what the user asked for and how the
//! agent answered.
//!
//! This writes nothing. The lake's labeler owns the aspect vocabulary, so the
//! output is a proposal document: each proposal names the aspect, its allowed
//! values, why it matters, which sampled sessions ground it, and the exact
//! `autolabel`/`train` commands that would turn it into a model. With
//! `--best`, Brama's independent `-best` route must call a proposal sensible
//! before it is kept; the audit failing is an exit-status matter, not a
//! silently shorter list.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::util::Result;
use crate::{brama, lake};

/// Sessions shown to the teacher per call: enough contrast to see a dimension,
/// small enough that every transcript excerpt stays readable.
const CHUNK_SESSIONS: usize = 5;
/// Per-session excerpt cap inside a teacher prompt.
const SESSION_CHARS: usize = 4000;
/// A proposal list is a few hundred tokens of JSON, not one word.
const PROPOSAL_MAX_TOKENS: u32 = 900;
const DEFAULT_SESSION_LIMIT: usize = 60;
const DEFAULT_MAX_ASPECTS: usize = 8;
const REVIEW_VALUES: [&str; 2] = ["sensible", "nonsensical"];
/// Values kept per merged aspect; a dimension needing more is a free-text
/// field, not a classification aspect.
const MAX_VALUES: usize = 8;
const MAX_EVIDENCE: usize = 12;

fn teacher_prompt(excerpts: &[(String, String)]) -> Vec<brama::Message> {
    let mut transcripts = String::new();
    for (session_id, text) in excerpts {
        transcripts.push_str(&format!("=== session {session_id} ===\n{text}\n\n"));
    }
    vec![
        brama::Message {
            role: "system".to_string(),
            content: "You design label taxonomies for coding-agent session \
                      transcripts. From the sessions shown you propose ASPECTS: \
                      independent dimensions worth classifying every session on, \
                      grounded in what the user actually asked for and how the \
                      agent actually answered — recurring behaviors, failure \
                      modes, user reactions. Rules: an aspect must be judgeable \
                      from a transcript alone; 2-6 mutually exclusive values in \
                      lowercase-kebab-case; no aspect that would be true of \
                      every session or of almost none; no restating of obvious \
                      metadata (language, length, runtime). Answer with strict \
                      JSON only: an array of at most 4 objects, each \
                      {\"aspect\": \"kebab-case-name\", \"values\": [..], \
                      \"description\": \"one sentence\", \
                      \"evidence\": [\"session ids from the input\"]}."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Propose aspects grounded in these sessions.\n\n{transcripts}"
            ),
        },
    ]
}

fn review_prompt(aspect: &str, values: &[String], description: &str) -> Vec<brama::Message> {
    vec![
        brama::Message {
            role: "system".to_string(),
            content: "You audit proposed aspect dimensions for coding-agent \
                      session transcripts. A sensible aspect is judgeable from \
                      a transcript alone, has mutually exclusive values, and \
                      would divide real sessions rather than land on one value \
                      always. Answer with exactly one word: sensible or \
                      nonsensical."
                .to_string(),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}\nValues: {}\nDescription: {description}\n\n\
                 Is this a sensible aspect to classify every session on? \
                 Answer sensible or nonsensical.",
                values.join(", ")
            ),
        },
    ]
}

fn kebab(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for character in name.trim().to_lowercase().chars() {
        if character.is_ascii_alphanumeric() {
            out.push(character);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// The teacher's JSON, with a possible ```json fence stripped.
fn parse_proposals(answer: &str) -> Option<Vec<Value>> {
    let trimmed = answer
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<Value>(trimmed)
        .ok()?
        .as_array()
        .cloned()
}

struct Merged {
    values: Vec<String>,
    description: String,
    support: usize,
    evidence: Vec<String>,
}

/// Propose aspect dimensions from recent sessions via the Brama teacher.
pub fn discover(
    limit: Option<i64>,
    brama_model: Option<&str>,
    best_review: bool,
    max_aspects: Option<i64>,
    runtime: Option<&str>,
) -> Result<Value> {
    let model_id = brama_model
        .filter(|value| !value.is_empty())
        .unwrap_or(brama::DEFAULT_MODEL)
        .to_string();
    let client = brama::BramaClient::from_env()?;

    let mut sessions = lake::all_sessions()?;
    if let Some(runtime) = runtime {
        sessions.retain(|session| session.runtime.as_deref() == Some(runtime));
    }
    let session_limit = limit
        .map(|value| value.max(0) as usize)
        .unwrap_or(DEFAULT_SESSION_LIMIT);
    let mut targets: Vec<String> = sessions
        .into_iter()
        .map(|session| session.session_id)
        .collect();
    if targets.len() > session_limit {
        // The lake lists oldest first; the newest sessions are the ones the
        // operator's current habits are visible in.
        targets = targets.split_off(targets.len() - session_limit);
    }
    let texts = lake::session_texts(&targets)?;

    let mut merged: BTreeMap<String, Merged> = BTreeMap::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut chunks = 0usize;
    let mut sampled = 0usize;
    let excerpts: Vec<(String, String)> = targets
        .iter()
        .filter_map(|session_id| {
            let entry = texts.get(session_id)?;
            let text = entry.text.trim();
            if text.is_empty() {
                return None;
            }
            Some((
                session_id.clone(),
                brama::truncate_chars(text, SESSION_CHARS),
            ))
        })
        .collect();
    for chunk in excerpts.chunks(CHUNK_SESSIONS) {
        chunks += 1;
        sampled += chunk.len();
        let prompt = teacher_prompt(chunk);
        let answer = match client.chat_limited(&model_id, &prompt, PROPOSAL_MAX_TOKENS) {
            Ok(answer) => answer,
            Err(error) => {
                failures.push(json!({
                    "chunk": chunks,
                    "error": brama::truncate_chars(&error.0, 200),
                }));
                continue;
            }
        };
        let Some(proposals) = parse_proposals(&answer) else {
            failures.push(json!({
                "chunk": chunks,
                "error": format!(
                    "unparseable proposals: {}",
                    brama::truncate_chars(&answer, 120)
                ),
            }));
            continue;
        };
        for proposal in proposals {
            let Some(aspect) = proposal.get("aspect").and_then(Value::as_str) else {
                continue;
            };
            let aspect = kebab(aspect);
            if aspect.is_empty() {
                continue;
            }
            let values: Vec<String> = proposal
                .get("values")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(kebab)
                        .filter(|value| !value.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            if values.len() < 2 {
                continue;
            }
            let description = proposal
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let evidence: Vec<String> = proposal
                .get("evidence")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let entry = merged.entry(aspect).or_insert_with(|| Merged {
                values: Vec::new(),
                description: description.clone(),
                support: 0,
                evidence: Vec::new(),
            });
            entry.support += 1;
            for value in values {
                if !entry.values.contains(&value) && entry.values.len() < MAX_VALUES {
                    entry.values.push(value);
                }
            }
            for session_id in evidence {
                if !entry.evidence.contains(&session_id) && entry.evidence.len() < MAX_EVIDENCE {
                    entry.evidence.push(session_id);
                }
            }
            if entry.description.is_empty() {
                entry.description = description;
            }
        }
    }

    let mut ranked: Vec<(String, Merged)> = merged.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .support
            .cmp(&left.1.support)
            .then_with(|| left.0.cmp(&right.0))
    });
    let keep = max_aspects
        .map(|value| value.max(0) as usize)
        .unwrap_or(DEFAULT_MAX_ASPECTS);
    ranked.truncate(keep);

    let mut proposals: Vec<Value> = Vec::new();
    let mut rejected: Vec<Value> = Vec::new();
    let mut review_failed = 0usize;
    for (aspect, entry) in ranked {
        let mut review = None;
        if best_review {
            let prompt = review_prompt(&aspect, &entry.values, &entry.description);
            match client.chat(brama::BEST_MODEL, &prompt) {
                Ok(answer) => match brama::parse_answer(&answer, &REVIEW_VALUES[..]) {
                    Some((verdict, _exact)) if verdict == REVIEW_VALUES[1] => {
                        rejected.push(json!({
                            "aspect": aspect,
                            "values": entry.values,
                            "best_review": verdict,
                        }));
                        continue;
                    }
                    Some((verdict, _exact)) => review = Some(verdict),
                    None => {
                        review_failed += 1;
                        failures.push(json!({
                            "aspect": aspect,
                            "error": format!(
                                "unparseable best review: {}",
                                brama::truncate_chars(&answer, 80)
                            ),
                        }));
                        continue;
                    }
                },
                Err(error) => {
                    review_failed += 1;
                    failures.push(json!({
                        "aspect": aspect,
                        "error": format!(
                            "best review failed: {}",
                            brama::truncate_chars(&error.0, 160)
                        ),
                    }));
                    continue;
                }
            }
        }
        let values_csv = entry.values.join(",");
        let mut proposal = Map::new();
        proposal.insert("aspect".to_string(), Value::String(aspect.clone()));
        proposal.insert(
            "values".to_string(),
            Value::Array(entry.values.iter().cloned().map(Value::String).collect()),
        );
        proposal.insert(
            "description".to_string(),
            Value::String(entry.description),
        );
        proposal.insert("support".to_string(), Value::Number(entry.support.into()));
        proposal.insert(
            "evidence_sessions".to_string(),
            Value::Array(entry.evidence.into_iter().map(Value::String).collect()),
        );
        if let Some(review) = review {
            proposal.insert("best_review".to_string(), Value::String(review));
        }
        proposal.insert(
            "next".to_string(),
            json!([
                format!(
                    "transcript-label-trainer autolabel --aspect {aspect} --values {values_csv} --best"
                ),
                format!("transcript-label-trainer train --aspect {aspect}"),
            ]),
        );
        proposals.push(Value::Object(proposal));
    }

    let mut summary = Map::new();
    summary.insert("brama_model".to_string(), Value::String(model_id));
    summary.insert(
        "sessions_sampled".to_string(),
        Value::Number(sampled.into()),
    );
    summary.insert("chunks".to_string(), Value::Number(chunks.into()));
    summary.insert(
        "best_review".to_string(),
        json!({
            "enabled": best_review,
            "model": if best_review {
                Value::String(brama::BEST_MODEL.to_string())
            } else {
                Value::Null
            },
            "accepted": proposals.len(),
            "rejected_nonsensical": rejected.len(),
            "failed": review_failed,
            "sensible": !best_review || (rejected.is_empty() && review_failed == 0),
        }),
    );
    summary.insert("proposals".to_string(), Value::Array(proposals));
    summary.insert("rejected".to_string(), Value::Array(rejected));
    summary.insert("failures".to_string(), Value::Array(failures));
    Ok(Value::Object(summary))
}
