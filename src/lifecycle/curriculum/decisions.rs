//! `lifecycle-decisions` and `lifecycle-examples`: reviewed lifecycle rows as
//! the two documents Ster reads. Labelled decisions are what `ster decisions
//! benchmark` measures; supervised examples are what `ster tune sft` trains
//! the served model on, each one the chat Oko sends — the lifecycle system
//! prompt, the row's user envelope — answered by the reviewed decision.
//!
//! Each row's one reviewed decision is checked against the decision contract
//! (`validate_decision`). Its state is the masked input envelope Oko sends;
//! its questions are the ones `training/lifecycle-model/decision/questions.json`
//! declares, with goal_ref's options being the row's own candidate
//! references described by their titles; its answers are the reviewed
//! decision's action, goal_ref and lifecycle_evidence. The title is no
//! question: the contract keeps it empty.

use serde_json::{json, Map};

use super::super::*;

const QUESTIONS: &str = include_str!("../../../training/lifecycle-model/decision/questions.json");

/// What one run converts and writes.
pub struct DecisionExport<'a> {
    pub rows: &'a Path,
    pub output: &'a Path,
}

fn choice(question: &Value, criteria: Value) -> Value {
    json!({ "type": "choice", "instructions": question["instructions"], "criteria": criteria })
}

/// goal_ref's options: every candidate reference of the envelope, described
/// by its candidate's title. A candidate without a title is refused, since an
/// option nobody can read cannot be chosen by reading the state.
fn candidate_criteria(row: &TrainingRow, envelope: &Value) -> Result<Value> {
    let mut criteria = Map::new();
    for candidate in envelope["candidates"].as_array().into_iter().flatten() {
        let reference = candidate["ref"]
            .as_str()
            .ok_or_else(|| Error(format!("{} has a candidate without a ref", row.id)))?;
        let title = candidate["title"]
            .as_str()
            .filter(|title| !title.trim().is_empty())
            .ok_or_else(|| Error(format!("{} candidate {reference} has no title", row.id)))?;
        criteria.insert(reference.to_string(), json!(title));
    }
    if criteria.is_empty() {
        return Err(Error(format!("{} offers no candidate", row.id)));
    }
    Ok(Value::Object(criteria))
}

/// The row's one reviewed answer: refused when it has none or several.
fn reviewed_answer(row: &TrainingRow) -> Result<&str> {
    let mut answers = row
        .messages
        .iter()
        .filter(|message| message.role == "assistant");
    match (answers.next(), answers.next()) {
        (Some(answer), None) => Ok(&answer.content),
        _ => Err(Error(format!(
            "{} must contain one reviewed assistant decision",
            row.id
        ))),
    }
}

/// One reviewed row as one Ster labelled example.
fn example(row: &TrainingRow, declared: &Value) -> Result<Value> {
    let decision = validate_decision(row, parse_json_object(reviewed_answer(row)?)?)?;
    let envelope = input_envelope(row)?;
    Ok(json!({
        "state": serde_json::to_string(&envelope)?,
        "questions": {
            "action": choice(&declared["action"], declared["action"]["criteria"].clone()),
            "goal_ref": choice(&declared["goal_ref"], candidate_criteria(row, &envelope)?),
            "lifecycle_evidence": choice(&declared["lifecycle_evidence"], declared["lifecycle_evidence"]["criteria"].clone()),
        },
        "answers": {
            "action": decision.action,
            "goal_ref": decision.goal_ref,
            "lifecycle_evidence": decision.lifecycle_evidence,
        },
    }))
}

/// One reviewed row as one Ster supervised example: the conversation Oko
/// serves (`classify` in `evaluate_gguf` sends the same system prompt and user
/// envelope) answered by the reviewed decision, its title blanked, as the JSON
/// object the output schema constrains the served answer to.
fn served_example(row: &TrainingRow) -> Result<Value> {
    let decision = validate_decision(row, parse_json_object(reviewed_answer(row)?)?)?;
    let user = &row
        .messages
        .iter()
        .find(|message| message.role == "user")
        .ok_or_else(|| Error(format!("{} has no user message", row.id)))?
        .content;
    Ok(json!({
        "system": SYSTEM_PROMPT.trim(),
        "prompt": user,
        "completion": serde_json::to_string(&decision)?,
    }))
}

/// Read every reviewed row, refusing a file without one.
fn reviewed_rows(rows: &Path) -> Result<Vec<TrainingRow>> {
    let read = read_rows(rows).map_err(|error| Error(format!("{}: {error}", rows.display())))?;
    if read.is_empty() {
        return Err(Error(format!("{} holds no reviewed row", rows.display())));
    }
    Ok(read)
}

/// Write `{"examples": examples}` to `output` whole, through a sibling file
/// renamed into place, and answer the count.
fn write_examples(output: &Path, examples: Vec<Value>) -> Result<Value> {
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    let count = examples.len();
    fs::write(&temporary, serde_json::to_vec_pretty(&json!({ "examples": examples }))?)?;
    fs::rename(&temporary, output)?;
    Ok(json!({ "examples": count, "output": output.display().to_string() }))
}

/// Convert every reviewed row, write the labelled set and answer the count.
pub fn export_decisions(run: &DecisionExport) -> Result<Value> {
    let declared: Value = serde_json::from_str(QUESTIONS)?;
    let examples = reviewed_rows(run.rows)?
        .iter()
        .map(|row| example(row, &declared))
        .collect::<Result<Vec<_>>>()?;
    write_examples(run.output, examples)
}

/// Convert every reviewed row, write the supervised set and answer the count.
pub fn export_examples(run: &DecisionExport) -> Result<Value> {
    let examples = reviewed_rows(run.rows)?
        .iter()
        .map(served_example)
        .collect::<Result<Vec<_>>>()?;
    write_examples(run.output, examples)
}
