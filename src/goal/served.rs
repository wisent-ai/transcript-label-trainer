//! The goal model trained and measured on the chat it is served: Jeden sends
//! the goal system prompt and the user's message wrapped as `<user>…</user>`,
//! and the model answers `<goal>…</goal>` or `<goal/>`.
//!
//! `goal-examples` writes the reviewed teacher rows as the Ster example set
//! `ster tune sft` trains on; the held-out gold rows never enter it.
//! `goal-evaluate-gguf` asks the quantized model, served by `crate::serving`,
//! for every gold row and writes the predictions `goal-audit` judges, so the
//! audit judges what Jeden will run rather than the trainer's own generation.
//! Decoding is constrained to the answer's two shapes by a grammar, so the
//! answer's length is bounded by the shape, not by a token budget.

use std::path::{Path, PathBuf};

use serde_json::json;

use super::*;
use crate::serving::{Server, Serving};

/// The two answers the model may give, as a llama.cpp grammar: no task, or one
/// goal on one line.
const ANSWER_GRAMMAR: &str = "root ::= \"<goal/>\" | \"<goal>\" [^<\\n]+ \"</goal>\"";

/// The user turn Jeden sends for `message`.
fn user_turn(message: &str) -> String {
    format!("<user>{message}</user>")
}

/// The reviewed answer of `row`.
fn answer(row: &GoalRow) -> String {
    match row.goal.as_deref().map(str::trim).filter(|goal| !goal.is_empty()) {
        Some(goal) => format!("<goal>{goal}</goal>"),
        None => "<goal/>".to_string(),
    }
}

/// Every reviewed row of `path`, refusing a file without one.
fn reviewed_rows(path: &Path) -> Result<Vec<GoalRow>> {
    let text = fs::read_to_string(path).map_err(|error| Error(format!("{}: {error}", path.display())))?;
    let rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<GoalRow>(line).map_err(|error| Error(format!("{}: {error}", path.display()))))
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Err(Error(format!("{} holds no reviewed row", path.display())));
    }
    Ok(rows)
}

/// Write `bytes` to `output` whole, through a sibling file renamed into place.
fn write_whole(output: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = output.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, output)?;
    Ok(())
}

/// Write the reviewed teacher rows of `rows` (every row not marked gold) as
/// the Ster supervised example set and answer the counts.
pub fn export_examples(rows: &Path, output: &Path) -> Result<Value> {
    let rows = reviewed_rows(rows)?;
    let held_out = rows.iter().filter(|row| row.gold).count();
    let examples: Vec<Value> = rows
        .iter()
        .filter(|row| !row.gold)
        .map(|row| json!({ "system": SYSTEM_PROMPT.trim(), "prompt": user_turn(&row.message), "completion": answer(row) }))
        .collect();
    if examples.is_empty() {
        return Err(Error(format!("every one of the {held_out} reviewed rows is held-out gold; none is left to train on")));
    }
    let count = examples.len();
    write_whole(output, &serde_json::to_vec_pretty(&json!({ "examples": examples }))?)?;
    Ok(json!({ "examples": count, "held_out": held_out, "output": output.display().to_string() }))
}

/// What one goal GGUF evaluation is given; every field is the caller's.
pub(crate) struct GoalEvaluation {
    pub(crate) serving: Serving,
    pub(crate) dataset: PathBuf,
    pub(crate) predictions: PathBuf,
    pub(crate) metrics: PathBuf,
}

/// Ask `server` for every gold row, in order; the first failed request ends
/// the run naming its session.
fn predictions(server: &Server, gold: &[GoalRow]) -> Result<Vec<(String, bool)>> {
    gold.iter()
        .map(|row| {
            let student = server.answer(&row.session_id, SYSTEM_PROMPT.trim(), &user_turn(&row.message), Some(("grammar", json!(ANSWER_GRAMMAR))))?;
            let exact = student == answer(row);
            println!("quantized goal prediction for {}: {}", row.session_id, if exact { "exact" } else { "differs" });
            Ok((student, exact))
        })
        .collect()
}

/// Ask the served model for every held-out gold row of the dataset, write
/// the predictions `goal-audit` reads and the exact-match share, and answer
/// the metrics. A failed request fails the run naming its session.
pub(crate) fn evaluate_gguf(run: &GoalEvaluation) -> Result<Value> {
    let gold: Vec<GoalRow> = reviewed_rows(&run.dataset)?.into_iter().filter(|row| row.gold).collect();
    if gold.is_empty() {
        return Err(Error(format!("{} holds no held-out gold row", run.dataset.display())));
    }
    let server = Server::start(&run.serving)?;
    let answered = predictions(&server, &gold);
    server.stop();
    let answered = answered?;
    let lines: String = gold
        .iter()
        .zip(&answered)
        .map(|(row, (student, _))| {
            let prediction = json!({
                "session_id": row.session_id,
                "message": row.message,
                "goal": row.goal.clone().unwrap_or_default(),
                "student": student,
            });
            format!("{prediction}\n")
        })
        .collect();
    write_whole(&run.predictions, lines.as_bytes())?;
    let exact = answered.iter().filter(|(_, exact)| *exact).count();
    let metrics = json!({
        "gold_rows": gold.len(),
        "exact_match": exact as f64 / gold.len() as f64,
    });
    write_whole(&run.metrics, (serde_json::to_string_pretty(&metrics)? + "\n").as_bytes())?;
    Ok(metrics)
}
