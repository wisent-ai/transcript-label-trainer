//! The humanizer trained with Ster and measured on the model it publishes.
//!
//! `humanizer-examples` writes a prepared split (each row the system prompt,
//! the generic source and the user's own target) as the Ster example set
//! `ster tune sft` trains on. `humanizer-evaluate-gguf` asks the base and the
//! merged student — both full-precision GGUFs, so the student answers exactly
//! as the base with the published adapter attached — for every test row
//! through `crate::serving`, one server after the other, and writes the
//! `predictions.jsonl` `humanizer-audit` judges and the `metrics.json` the
//! manifest reads. Each answer is scored against the target and the source by
//! chrF as sacrebleu computes it by default, and by its length against the
//! target's. Each answer runs until the model ends it, as llama-server does by
//! default; nothing here sets a token budget.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::{MODEL_CONTRACT, REPORT_SCHEMA_VERSION};
use crate::serving::{Server, Serving};
use crate::util::{Error, Result};

/// One prepared row's three turns: system prompt, source, target.
struct Turns {
    id: String,
    system: String,
    source: String,
    target: String,
    metadata: Value,
}

fn turns(row: &Value) -> Result<Turns> {
    let id = row.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
    let content = |role: &str| -> Result<String> {
        let mut found = row
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some(role));
        match (found.next(), found.next()) {
            (Some(message), None) => message
                .get("content")
                .and_then(Value::as_str)
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
                .ok_or_else(|| Error(format!("row {id:?} has an empty {role} turn"))),
            _ => Err(Error(format!("row {id:?} must hold exactly one {role} turn"))),
        }
    };
    Ok(Turns {
        system: content("system")?,
        source: content("user")?,
        target: content("assistant")?,
        metadata: row.get("metadata").cloned().unwrap_or(Value::Null),
        id,
    })
}

/// Every prepared row of `path`, refusing a file without one.
fn split_rows(path: &Path) -> Result<Vec<Turns>> {
    let text = std::fs::read_to_string(path).map_err(|error| Error(format!("{}: {error}", path.display())))?;
    let rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: Value = serde_json::from_str(line).map_err(|error| Error(format!("{}: {error}", path.display())))?;
            turns(&row)
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Err(Error(format!("{} holds no prepared row", path.display())));
    }
    Ok(rows)
}

/// Write `bytes` to `output` whole, through a sibling file renamed into place.
fn write_whole(output: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, output)?;
    Ok(())
}

/// Write the prepared split `rows` as the Ster supervised example set.
pub fn export_examples(rows: &Path, output: &Path) -> Result<Value> {
    let examples: Vec<Value> = split_rows(rows)?
        .into_iter()
        .map(|row| json!({ "system": row.system, "prompt": row.source, "completion": row.target }))
        .collect();
    let count = examples.len();
    write_whole(output, &serde_json::to_vec_pretty(&json!({ "examples": examples }))?)?;
    Ok(json!({ "examples": count, "output": output.display().to_string() }))
}

/// chrF's settings, sacrebleu's documented defaults: character n-grams of
/// every order up to six with whitespace removed and case kept, and an
/// F-score weighting recall twice as much as precision.
// https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py (char_order=6)
const CHAR_ORDER: usize = 6;
// https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py (beta=2)
const BETA: f64 = 2.0;
/// Where those settings are documented, recorded with the metrics.
const CHRF_SOURCE: &str = "https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py";

/// Each distinct character n-gram of `chars` of length `order` with how
/// often it occurs; empty when the text is shorter than one n-gram.
fn grams(chars: &[char], order: usize) -> Vec<(&[char], usize)> {
    let mut all: Vec<&[char]> = chars.windows(order).collect();
    all.sort();
    all.chunk_by(|left, right| left == right).filter_map(|run| run.first().map(|gram| (*gram, run.len()))).collect()
}

/// chrF of `candidate` against `reference` (Popović 2015, as sacrebleu
/// computes it): precision and recall of shared character n-grams, each
/// averaged over the orders both texts are long enough for, combined into
/// their F-score with recall weighted `BETA` times. Nothing in common, or a
/// text with no n-gram at all, scores zero.
fn chrf(reference: &str, candidate: &str) -> f64 {
    let strip = |text: &str| text.chars().filter(|c| !c.is_whitespace()).collect::<Vec<char>>();
    let (reference, candidate) = (strip(reference), strip(candidate));
    let scores: Vec<(f64, f64)> = (std::num::NonZeroUsize::MIN.get()..=CHAR_ORDER)
        .filter_map(|order| {
            let (wanted, got) = (grams(&reference, order), grams(&candidate, order));
            if wanted.is_empty() || got.is_empty() {
                return None;
            }
            let total = |counts: &[(&[char], usize)]| counts.iter().map(|(_, count)| *count).sum::<usize>() as f64;
            let overlap: usize = wanted
                .iter()
                .map(|(gram, have)| got.iter().filter(|(other, _)| other == gram).map(|(_, count)| (*have).min(*count)).sum::<usize>())
                .sum();
            Some((overlap as f64 / total(&got), overlap as f64 / total(&wanted)))
        })
        .collect();
    let orders = scores.len().max(std::num::NonZeroUsize::MIN.get()) as f64;
    let precision = scores.iter().map(|(precision, _)| precision).sum::<f64>() / orders;
    let recall = scores.iter().map(|(_, recall)| recall).sum::<f64>() / orders;
    let weight = BETA * BETA;
    let denominator = weight * precision + recall;
    if !denominator.is_normal() {
        // Precision and recall are both zero: so is their product.
        return precision * recall;
    }
    // F-beta, https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py: (1 + beta^2) P R / (beta^2 P + R)
    (1.0 + weight) * precision * recall / denominator
}

fn score(row: &Turns, candidate: &str) -> Value {
    json!({
        "target_chrf": chrf(&row.target, candidate),
        "source_chrf": chrf(&row.source, candidate),
        "length_ratio": candidate.chars().count() as f64 / row.target.chars().count() as f64,
    })
}

/// The mean of every score of `scores`, key by key.
fn means(scores: &[Value]) -> Value {
    let mut sums: Map<String, Value> = Map::new();
    for score in scores {
        for (key, value) in score.as_object().into_iter().flatten() {
            let sum = sums.get(key).and_then(Value::as_f64).unwrap_or_default() + value.as_f64().unwrap_or_default();
            sums.insert(key.clone(), json!(sum));
        }
    }
    let count = scores.len() as f64;
    Value::Object(sums.into_iter().map(|(key, sum)| (key, json!(sum.as_f64().unwrap_or_default() / count))).collect())
}

/// What one humanizer evaluation is given; every field is the caller's.
pub(crate) struct HumanizerEvaluation {
    pub(crate) base: Serving,
    pub(crate) student: Serving,
    pub(crate) dataset: PathBuf,
    pub(crate) predictions: PathBuf,
    pub(crate) metrics: PathBuf,
    pub(crate) base_model: String,
    pub(crate) base_revision: String,
}

/// Ask the model `serving` serves for every row, in order; the first failed
/// request ends the run naming its row.
fn answers(serving: &Serving, rows: &[Turns], which: &str) -> Result<Vec<String>> {
    let server = Server::start(serving)?;
    let answered = rows
        .iter()
        .map(|row| {
            let answer = server.answer(&row.id, &row.system, &row.source, None);
            println!("{which} humanizer answer for {}: {}", row.id, if answer.is_ok() { "received" } else { "failed" });
            answer
        })
        .collect();
    server.stop();
    answered
}

/// Ask the base and the student for every test row, write the predictions
/// and the metrics, and answer the metrics.
pub(crate) fn evaluate_gguf(run: &HumanizerEvaluation) -> Result<Value> {
    let rows = split_rows(&run.dataset)?;
    for (which, serving) in [("base", &run.base), ("student", &run.student)] {
        if !serving.model.is_file() {
            return Err(Error(format!("the {which} model {} does not exist", serving.model.display())));
        }
    }
    let base = answers(&run.base, &rows, "base")?;
    let student = answers(&run.student, &rows, "student")?;
    let mut lines = String::new();
    let (mut base_scores, mut student_scores) = (Vec::new(), Vec::new());
    for ((row, base), student) in rows.iter().zip(&base).zip(&student) {
        let (base_score, student_score) = (score(row, base), score(row, student));
        let prediction = json!({
            "id": row.id,
            "source": row.source,
            "target": row.target,
            "base": base,
            "student": student,
            "base_metrics": base_score,
            "student_metrics": student_score,
            "metadata": row.metadata,
        });
        lines.push_str(&format!("{prediction}\n"));
        base_scores.push(base_score);
        student_scores.push(student_score);
    }
    write_whole(&run.predictions, lines.as_bytes())?;
    let (base_means, student_means) = (means(&base_scores), means(&student_scores));
    let gain = student_means["target_chrf"].as_f64().unwrap_or_default() - base_means["target_chrf"].as_f64().unwrap_or_default();
    let metrics = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract": MODEL_CONTRACT,
        "base_model": run.base_model,
        "base_revision": run.base_revision,
        "test_sha256": hex::encode(Sha256::digest(std::fs::read(&run.dataset)?)),
        "test_rows": rows.len(),
        "chrf": { "char_order": CHAR_ORDER, "beta": BETA, "whitespace": false, "lowercase": false, "source": CHRF_SOURCE },
        "evaluation_surface": "base and merged student as full-precision GGUFs through llama-server, one after the other",
        "base": base_means,
        "student": student_means,
        "target_chrf_gain": gain,
    });
    write_whole(&run.metrics, (serde_json::to_string_pretty(&metrics)? + "\n").as_bytes())?;
    Ok(metrics)
}
