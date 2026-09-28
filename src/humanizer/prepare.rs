//! Inverse style-transfer pairs from masked authored messages: a Brama teacher
//! rewrites each target as generic AI prose, the anchor contract and an
//! independent Brama review keep only faithful pairs, and each session lands
//! in exactly one of the train, validation and test splits.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::anchors::valid_source;
use super::calls::{ask, fan_out, json_object, read_jsonl, ERROR_EXCERPT};
use super::prompts::{REVIEW_PROMPT, SYSTEM_PROMPT, TEACHER_PROMPT};
use super::{TargetRow, REPORT_SCHEMA_VERSION};
use crate::brama::{truncate_chars, BramaClient};
use crate::util::{Error, Result};

/// Rows prepared between two progress lines.
const PROGRESS_EVERY: usize = 50;

/// Output budget for the teacher: twice the target's characters, never below
/// what a short message needs nor above one long message.
const TEACHER_MIN_TOKENS: usize = 192;
const TEACHER_MAX_TOKENS: usize = 1024;

/// The review answer is four booleans.
const REVIEW_MAX_TOKENS: u32 = 96;

/// Session buckets: one of ten goes to test, one to validation, the rest to
/// train, keyed on the session so no conversation spans two splits.
const SPLIT_BUCKETS: u32 = 10;

/// The fewest accepted rows each split needs before training may start.
const SPLIT_MINIMUMS: [(&str, usize); 3] = [("train", 700), ("validation", 70), ("test", 70)];

const REVIEW_FIELDS: [&str; 4] = ["faithful", "generic_ai", "same_language", "usable"];

fn parse_review(answer: &str) -> Result<serde_json::Map<String, Value>> {
    let review = json_object(answer, "review")?;
    let shaped = review.len() == REVIEW_FIELDS.len()
        && REVIEW_FIELDS
            .iter()
            .all(|field| review.get(*field).is_some_and(Value::is_boolean));
    match shaped {
        true => Ok(review),
        false => crate::bail!("review has invalid shape: {}", Value::Object(review)),
    }
}

fn split_of(session_id: &str) -> &'static str {
    let digest = Sha256::digest(session_id.as_bytes());
    let head = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    match head % SPLIT_BUCKETS {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}

/// One accepted pair, or the reason it was refused.
fn pair(
    row: &TargetRow,
    client: &BramaClient,
    teacher: &str,
    reviewer: &str,
) -> Result<Value, String> {
    let target = row.target.trim();
    let budget = (target.chars().count() * 2).clamp(TEACHER_MIN_TOKENS, TEACHER_MAX_TOKENS);
    let excerpt = |error: Error| format!("error:{}", truncate_chars(&error.0, ERROR_EXCERPT));
    let source = ask(
        client,
        teacher,
        TEACHER_PROMPT,
        target.to_string(),
        budget as u32,
        |answer| Ok(answer.to_string()),
    )
    .map_err(excerpt)?;
    if !valid_source(target, &source) {
        return Err("source_contract".to_string());
    }
    let question = json!({"source": source, "target": target}).to_string();
    let review = ask(
        client,
        reviewer,
        REVIEW_PROMPT,
        question,
        REVIEW_MAX_TOKENS,
        parse_review,
    )
    .map_err(excerpt)?;
    if review.get("usable") != Some(&Value::Bool(true)) {
        return Err("review_rejected".to_string());
    }
    Ok(json!({
        "id": row.id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "session_id": row.session_id,
            "runtime": row.runtime,
            "split": split_of(&row.session_id),
            "teacher_model": teacher,
            "review_model": reviewer,
            "review": review,
        },
    }))
}

fn jsonl(values: &[&Value]) -> Result<String> {
    let mut text = String::new();
    for value in values {
        text.push_str(&serde_json::to_string(value)?);
        text.push('\n');
    }
    Ok(text)
}

/// Prepare `input` (JSONL of `{id, session_id, runtime, target}`) into
/// `train.jsonl`, `validation.jsonl`, `test.jsonl` and `preparation.json`
/// under `output_dir`. Nothing is written unless every split reaches its
/// minimum.
pub fn prepare_dataset(
    input: &Path,
    output_dir: &Path,
    teacher: &str,
    reviewer: &str,
    workers: usize,
) -> Result<Value> {
    let bytes = fs::read(input)
        .map_err(|error| Error(format!("cannot read {}: {error}", input.display())))?;
    let targets: Vec<TargetRow> = read_jsonl(input)?;
    let client = BramaClient::from_env()?;
    let outcomes = fan_out(&targets, workers, "prepared", PROGRESS_EVERY, |row| {
        pair(row, &client, teacher, reviewer)
    });
    let mut accepted = Vec::new();
    let mut rejected: BTreeMap<String, usize> = BTreeMap::new();
    for outcome in outcomes {
        match outcome {
            Ok(value) => accepted.push(value),
            Err(reason) => *rejected.entry(reason).or_default() += 1,
        }
    }
    accepted.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    let mut splits = serde_json::Map::new();
    let mut files = Vec::new();
    for (name, minimum) in SPLIT_MINIMUMS {
        let rows: Vec<&Value> = accepted
            .iter()
            .filter(|row| row["metadata"]["split"] == name)
            .collect();
        if rows.len() < minimum {
            crate::bail!(
                "{name} has {} accepted rows; need {minimum} (rejected: {})",
                rows.len(),
                json!(rejected)
            );
        }
        splits.insert(name.to_string(), json!(rows.len()));
        files.push((output_dir.join(format!("{name}.jsonl")), jsonl(&rows)?));
    }
    fs::create_dir_all(output_dir)?;
    for (path, text) in files {
        fs::write(&path, text)
            .map_err(|error| Error(format!("cannot write {}: {error}", path.display())))?;
    }
    let report = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": "transcript-lake:masked-user-events",
        "target_rows": targets.len(),
        "accepted_rows": accepted.len(),
        "splits": splits,
        "rejected": rejected,
        "teacher_model": teacher,
        "review_model": reviewer,
        "system_prompt_sha256": hex::encode(Sha256::digest(SYSTEM_PROMPT.as_bytes())),
        "input_sha256": hex::encode(Sha256::digest(&bytes)),
    });
    let path = output_dir.join("preparation.json");
    fs::write(&path, serde_json::to_string_pretty(&report)? + "\n")
        .map_err(|error| Error(format!("cannot write {}: {error}", path.display())))?;
    Ok(report)
}
