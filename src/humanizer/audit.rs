//! Independent Brama audit of the base and trained humanizer on held-out
//! cases: every case is judged, the full record is kept, and the trained
//! adapter qualifies only when it beats the base on voice without losing
//! meaning.

use std::ops::RangeInclusive;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::calls::{ask_call, fan_out, read_jsonl};
use super::prompts::JUDGE_PROMPT;
use super::{MODEL_CONTRACT, REPORT_SCHEMA_VERSION};
use crate::brama::BramaClient;
use crate::util::{Error, Result};

/// The scale JUDGE_PROMPT asks the judge to score each candidate on.
const SCORE_SCALE: RangeInclusive<f64> = 0.0..=1.0; // https://github.com/wisent-ai/transcript-label-trainer/blob/main/src/humanizer/prompts.rs ("Score each candidate independently from 0 to 1")

#[derive(Deserialize)]
struct Prediction {
    id: String,
    source: Value,
    target: Value,
    base: Value,
    student: Value,
}

/// The judge's answer, exactly the shape JUDGE_PROMPT asks for.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Verdict {
    base: Judged,
    student: Judged,
}

/// How the judge rated one candidate.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Judged {
    semantic_fidelity: f64,
    voice_match: f64,
    ai_boilerplate: bool,
    passed: bool,
}

/// The function the judge answers through: its arguments are the verdict.
fn verdict_function() -> (&'static str, &'static str, Value) {
    let judged = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["semantic_fidelity", "voice_match", "ai_boilerplate", "passed"],
        "properties": {
            "semantic_fidelity": { "type": "number" },
            "voice_match": { "type": "number" },
            "ai_boilerplate": { "type": "boolean" },
            "passed": { "type": "boolean" },
        },
    });
    (
        "record_verdict",
        "The judge's rating of the base and the student candidate.",
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["base", "student"],
            "properties": { "base": judged, "student": judged },
        }),
    )
}

fn parse_verdict(arguments: serde_json::Map<String, Value>) -> Result<Verdict> {
    let verdict: Verdict = serde_json::from_value(Value::Object(arguments.clone())).map_err(|error| {
        Error(format!("judge output is not the record_verdict object ({error}): {}", Value::Object(arguments)))
    })?;
    for (name, judged) in [("base", &verdict.base), ("student", &verdict.student)] {
        for (score, value) in [
            ("semantic_fidelity", judged.semantic_fidelity),
            ("voice_match", judged.voice_match),
        ] {
            if !SCORE_SCALE.contains(&value) {
                crate::bail!("judge output scores {name}.{score} {value}, outside {SCORE_SCALE:?}");
            }
        }
    }
    Ok(verdict)
}

/// One judged case.
struct Record {
    id: String,
    verdict: Verdict,
}

fn judge(row: &Prediction, client: &BramaClient, model: &str) -> Result<Record> {
    let question = json!({
        "source": row.source,
        "target": row.target,
        "base": row.base,
        "student": row.student,
    })
    .to_string();
    let verdict = ask_call(client, model, JUDGE_PROMPT, question, verdict_function(), parse_verdict)
        .map_err(|error| Error(format!("{}: {error}", row.id)))?;
    Ok(Record {
        id: row.id.clone(),
        verdict,
    })
}

/// One candidate's measures over every judged case: the mean of each score
/// and the share of cases each verdict holds true for.
#[derive(Serialize)]
struct Measures {
    semantic_fidelity: f64,
    voice_match: f64,
    ai_boilerplate_rate: f64,
    pass_rate: f64,
}

impl Measures {
    /// `judged` holds at least one rating.
    fn of(judged: &[&Judged]) -> Self {
        let cases = judged.len() as f64;
        let mean = |score: fn(&Judged) -> f64| judged.iter().map(|one| score(one)).sum::<f64>() / cases;
        let rate = |held: fn(&Judged) -> bool| judged.iter().filter(|one| held(one)).count() as f64 / cases;
        Self {
            semantic_fidelity: mean(|one| one.semantic_fidelity),
            voice_match: mean(|one| one.voice_match),
            ai_boilerplate_rate: rate(|one| one.ai_boilerplate),
            pass_rate: rate(|one| one.passed),
        }
    }
}

/// Judge every row of `predictions` (JSONL of `{id, source, target, base,
/// student}`) through `model` and write the full record to `output`. The
/// returned summary carries `passed`; an incomplete audit is an error and
/// writes nothing, and so is a predictions file without a row. The adapter
/// passes when the judge measures it better than the base it was trained from
/// on the same held-out cases: closer to the author's voice, no less faithful
/// to the meaning, passed no less often and called AI boilerplate no more
/// often. The base's own scores are the bound; nobody states one.
pub fn audit_outputs(
    predictions: &Path,
    output: &Path,
    model: &str,
) -> Result<Value> {
    let rows: Vec<Prediction> = read_jsonl(predictions)?;
    if rows.is_empty() {
        crate::bail!("{} holds no predictions to audit", predictions.display());
    }
    let client = BramaClient::from_env()?;
    let workers = client.allowance(model)?;
    let outcomes = fan_out(&rows, workers, "audited", |row| {
        judge(row, &client, model)
    });
    let mut records = Vec::new();
    let mut failures = Vec::new();
    for outcome in outcomes {
        match outcome {
            Ok(record) => records.push(record),
            Err(error) => failures.push(error.0),
        }
    }
    if !failures.is_empty() {
        let counted = crate::model::class_counts(failures.iter().map(String::as_str));
        crate::bail!("audit incomplete: {}", json!(counted));
    }
    records.sort_by(|left, right| left.id.cmp(&right.id));
    let base = Measures::of(&records.iter().map(|record| &record.verdict.base).collect::<Vec<_>>());
    let student =
        Measures::of(&records.iter().map(|record| &record.verdict.student).collect::<Vec<_>>());
    let passed = student.voice_match > base.voice_match
        && student.semantic_fidelity >= base.semantic_fidelity
        && student.pass_rate >= base.pass_rate
        && student.ai_boilerplate_rate <= base.ai_boilerplate_rate;
    let mut report = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract": MODEL_CONTRACT,
        "review_model": model,
        "rows": records.len(),
        "voice_match_gain": student.voice_match - base.voice_match,
        "semantic_fidelity_delta": student.semantic_fidelity - base.semantic_fidelity,
        "base": base,
        "student": student,
        "gate": "student better than base: voice_match higher, semantic_fidelity and pass_rate no lower, ai_boilerplate_rate no higher",
        "passed": passed,
        "records": records
            .iter()
            .map(|record| json!({"id": record.id, "verdict": record.verdict}))
            .collect::<Vec<_>>(),
    });
    crate::util::write_json_pretty(output, &report)?;
    if let Some(fields) = report.as_object_mut() {
        fields.remove("records");
    }
    Ok(report)
}
