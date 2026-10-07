//! Inverse style-transfer pairs from masked authored messages: a Brama teacher
//! rewrites each target as generic AI prose, the anchor contract and an
//! independent Brama review keep only faithful pairs, and each session lands
//! in exactly one of the train, validation and test splits.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::anchors::{valid_source, LengthRatio};
use super::calls::{ask, fan_out, json_object, read_jsonl};
use super::prompts::{REVIEW_PROMPT, SYSTEM_PROMPT, TEACHER_PROMPT};
use super::{TargetRow, REPORT_SCHEMA_VERSION};
use crate::brama::BramaClient;
use crate::util::{Error, Result};

/// What preparation holds the teacher's pairs and splits to, every value the
/// caller's to state: the fewest accepted rows each split needs before
/// training may start, how long a generated source may be against its
/// target, and the shares of sessions held out for test and validation.
pub struct PreparationBounds {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
    pub length: LengthRatio,
    pub held_out: HeldOut,
}

/// The shares of sessions that go to test and to validation; the rest go to
/// train. Each is a share of all sessions, and together they leave train a
/// positive share.
pub struct HeldOut {
    pub test: f64,
    pub validation: f64,
}

impl HeldOut {
    fn check(&self) -> Result<()> {
        let total = self.test + self.validation;
        // Both are shares, and the sign of a non-negative total is one:
        // train keeps what the two leave, so together they stay below it.
        let valid = [self.test, self.validation]
            .iter()
            .all(|share| share.is_finite() && !share.is_sign_negative())
            && total < total.signum();
        match valid {
            true => Ok(()),
            false => crate::bail!(
                "--test-share {} and --validation-share {} must be shares from 0 to 1 that together leave train a share",
                self.test,
                self.validation
            ),
        }
    }
}

impl PreparationBounds {
    fn named(&self) -> [(&'static str, usize); 3] {
        [("train", self.train), ("validation", self.validation), ("test", self.test)]
    }
}

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

/// The split of a session: its digest read as a point between none and all,
/// below the test share in test, below both shares in validation, else train.
fn split_of(session_id: &str, held_out: &HeldOut) -> &'static str {
    let digest = Sha256::digest(session_id.as_bytes());
    let head = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let point = f64::from(head) / f64::from(u32::MAX);
    if point < held_out.test {
        "test"
    } else if point < held_out.test + held_out.validation {
        "validation"
    } else {
        "train"
    }
}

/// One accepted pair, or the reason it was refused.
fn pair(
    row: &TargetRow,
    client: &BramaClient,
    teacher: &str,
    reviewer: &str,
    attempts: usize,
    bounds: &PreparationBounds,
) -> Result<Value, String> {
    let target = row.target.trim();
    let excerpt = |error: Error| format!("error:{}", error.0);
    let source = ask(client, teacher, attempts, TEACHER_PROMPT, target.to_string(), |answer| {
        Ok(answer.to_string())
    })
    .map_err(excerpt)?;
    if !valid_source(target, &source, &bounds.length) {
        return Err("source_contract".to_string());
    }
    let question = json!({"source": source, "target": target}).to_string();
    let review = ask(client, reviewer, attempts, REVIEW_PROMPT, question, parse_review)
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
            "split": split_of(&row.session_id, &bounds.held_out),
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
    attempts: usize,
    minimums: &PreparationBounds,
) -> Result<Value> {
    minimums.held_out.check()?;
    let bytes = fs::read(input)
        .map_err(|error| Error(format!("cannot read {}: {error}", input.display())))?;
    let targets: Vec<TargetRow> = read_jsonl(input)?;
    let client = BramaClient::from_env()?;
    let outcomes = fan_out(&targets, workers, "prepared", |row| {
        pair(row, &client, teacher, reviewer, attempts, minimums)
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
    for (name, minimum) in minimums.named() {
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
        "split_minimums": minimums.named().into_iter().map(|(name, minimum)| (name.to_string(), json!(minimum))).collect::<serde_json::Map<_, _>>(),
        "source_length_ratio": {"min": minimums.length.min, "max": minimums.length.max},
        "held_out_shares": {"test": minimums.held_out.test, "validation": minimums.held_out.validation},
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
