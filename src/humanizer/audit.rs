//! Independent Brama audit of the base and trained humanizer on held-out
//! cases: every case is judged, the full record is kept, and the trained
//! adapter qualifies only when it beats the base on voice without losing
//! meaning.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::calls::{ask, fan_out, json_object, read_jsonl};
use super::prompts::JUDGE_PROMPT;
use super::{MODEL_CONTRACT, REPORT_SCHEMA_VERSION};
use crate::brama::BramaClient;
use crate::util::{Error, Result};

const CANDIDATES: [&str; 2] = ["base", "student"];
const SCORES: [&str; 2] = ["semantic_fidelity", "voice_match"];
const VERDICTS: [&str; 2] = ["ai_boilerplate", "passed"];

#[derive(Deserialize)]
struct Prediction {
    id: String,
    source: Value,
    target: Value,
    base: Value,
    student: Value,
}

fn parse_verdict(answer: &str) -> Result<Map<String, Value>> {
    let verdict = json_object(answer, "judge output")?;
    if verdict.len() != CANDIDATES.len()
        || !CANDIDATES.iter().all(|name| verdict.contains_key(*name))
    {
        crate::bail!("judge output has invalid candidates");
    }
    for name in CANDIDATES {
        let Some(result) = verdict[name].as_object() else {
            crate::bail!("judge output has invalid {name} shape");
        };
        let fields = SCORES.len() + VERDICTS.len();
        let complete = SCORES
            .iter()
            .chain(VERDICTS.iter())
            .all(|key| result.contains_key(*key));
        if result.len() != fields || !complete {
            crate::bail!("judge output has invalid {name} shape");
        }
        for score in SCORES {
            let valid = result[score]
                .as_f64()
                .is_some_and(|value| (0.0..=1.0).contains(&value));
            if !valid {
                crate::bail!("judge output has invalid {name}.{score}");
            }
        }
        if !VERDICTS.iter().all(|key| result[*key].is_boolean()) {
            crate::bail!("judge output has invalid {name} verdict");
        }
    }
    Ok(verdict)
}

fn judge(row: &Prediction, client: &BramaClient, model: &str) -> Result<Value> {
    let question = json!({
        "source": row.source,
        "target": row.target,
        "base": row.base,
        "student": row.student,
    })
    .to_string();
    let verdict = ask(client, model, JUDGE_PROMPT, question, parse_verdict)
        .map_err(|error| Error(format!("{}: {error}", row.id)))?;
    Ok(json!({"id": row.id, "verdict": verdict}))
}

fn mean(records: &[Value], candidate: &str, field: &str) -> f64 {
    let total: f64 = records
        .iter()
        .map(|record| &record["verdict"][candidate][field])
        .map(|value| {
            value
                .as_f64()
                .unwrap_or(f64::from(u8::from(value.as_bool() == Some(true))))
        })
        .sum();
    total / records.len() as f64
}

fn aggregate(records: &[Value], candidate: &str) -> Value {
    json!({
        "semantic_fidelity": mean(records, candidate, "semantic_fidelity"),
        "voice_match": mean(records, candidate, "voice_match"),
        "ai_boilerplate_rate": mean(records, candidate, "ai_boilerplate"),
        "pass_rate": mean(records, candidate, "passed"),
    })
}

/// Judge every row of `predictions` (JSONL of `{id, source, target, base,
/// student}`) through `model` and write the full record to `output`. The
/// returned summary carries `passed`; an incomplete audit is an error and
/// writes nothing. The adapter passes when the judge measures it better than
/// the base it was trained from on the same held-out cases: closer to the
/// author's voice, no less faithful to the meaning, passed no less often and
/// called AI boilerplate no more often. The base's own scores are the bound;
/// nobody states one.
pub fn audit_outputs(
    predictions: &Path,
    output: &Path,
    model: &str,
) -> Result<Value> {
    let rows: Vec<Prediction> = read_jsonl(predictions)?;
    let client = BramaClient::from_env()?;
    let workers = client.allowance(model)?.get();
    let outcomes = fan_out(&rows, workers, "audited", |row| {
        judge(row, &client, model)
    });
    let mut records = Vec::new();
    let mut failures: BTreeMap<String, usize> = BTreeMap::new();
    for outcome in outcomes {
        match outcome {
            Ok(record) => records.push(record),
            Err(error) => {
                *failures
                    .entry(error.0.clone())
                    .or_default() += 1
            }
        }
    }
    if !failures.is_empty() {
        crate::bail!("audit incomplete: {}", json!(failures));
    }
    records.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    let base = aggregate(&records, "base");
    let student = aggregate(&records, "student");
    let score = |value: &Value, field: &str| value[field].as_f64().unwrap_or(f64::NAN);
    let voice_gain = score(&student, "voice_match") - score(&base, "voice_match");
    let semantic_delta = score(&student, "semantic_fidelity") - score(&base, "semantic_fidelity");
    let better = |field: &str| score(&student, field) > score(&base, field);
    let no_worse = |field: &str| score(&student, field) >= score(&base, field);
    let passed = better("voice_match")
        && no_worse("semantic_fidelity")
        && no_worse("pass_rate")
        && score(&student, "ai_boilerplate_rate") <= score(&base, "ai_boilerplate_rate");
    let mut report = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract": MODEL_CONTRACT,
        "review_model": model,
        "rows": records.len(),
        "base": base,
        "student": student,
        "voice_match_gain": voice_gain,
        "semantic_fidelity_delta": semantic_delta,
        "gate": "student better than base: voice_match higher, semantic_fidelity and pass_rate no lower, ai_boilerplate_rate no higher",
        "passed": passed,
        "records": records,
    });
    fs::write(output, serde_json::to_string_pretty(&report)? + "\n")
        .map_err(|error| Error(format!("cannot write {}: {error}", output.display())))?;
    if let Some(fields) = report.as_object_mut() {
        fields.remove("records");
    }
    Ok(report)
}
