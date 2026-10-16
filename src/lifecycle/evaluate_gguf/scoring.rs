//! The evaluation run: every row asked on one worker per slot the server
//! chose, one progress line per finished row, then the predictions file and
//! the metrics, which report each measured rate and leave out a rate with
//! nothing to measure. Both files are written whole, so a reader never sees
//! half of either.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use serde_json::json;

use super::*;

/// `count` of `total` as a rate; null when there was nothing to measure.
fn rate(count: usize, total: usize) -> Value {
    match NonZeroUsize::new(total) {
        Some(total) => json!(count as f64 / total.get() as f64),
        None => Value::Null,
    }
}

/// What one row's answer measured against its target.
struct Scored {
    /// The action the target took, which groups the row in `by_action`.
    action: String,
    valid: bool,
    action_correct: bool,
    goal_ref_correct: bool,
    evidence_correct: bool,
    joint_correct: bool,
    /// For a prediction that declared an explicit completion, whether it
    /// took the target's action.
    finish: Option<bool>,
}

/// How many of `rows` `hit` holds for.
fn counted<'a>(rows: impl IntoIterator<Item = &'a Scored>, hit: impl Fn(&Scored) -> bool) -> usize {
    rows.into_iter().filter(|row| hit(row)).count()
}

pub fn evaluate_gguf(run: &GgufEvaluation) -> Result<Value> {
    for (what, path) in [
        ("model", &run.model),
        ("dataset", &run.dataset),
        ("server", &run.server),
        ("output schema", &run.output_schema),
    ] {
        if !path.is_file() {
            return Err(Error(format!(
                "the GGUF evaluation {what} {} does not exist",
                path.display()
            )));
        }
    }
    let schema: Value = serde_json::from_str(&crate::util::read_text(&run.output_schema)?)?;
    let rows = read_rows(&run.dataset)?;
    if rows.is_empty() {
        return Err(Error(format!(
            "{} holds no evaluation row",
            run.dataset.display()
        )));
    }
    let targets = rows
        .iter()
        .map(|row| target(row, &schema))
        .collect::<Result<Vec<_>>>()?;
    let server = crate::serving::Server::start(&run.serving())?;
    let queue = Mutex::new(rows.iter().enumerate());
    let answers: Mutex<Vec<Option<Result<(String, Option<Decision>)>>>> =
        Mutex::new(rows.iter().map(|_| None).collect());
    // Scoped workers borrow the one server, the queue and the answers; the
    // scope ends when every worker has answered, before the server is stopped.
    let joined: Result<()> = thread::scope(|scope| {
        let (server, schema, queue, answers) = (&server, &schema, &queue, &answers);
        let workers: Vec<_> = std::iter::repeat_with(|| {
            scope.spawn(move || loop {
                let Some((index, row)) = queue.lock().expect("evaluation queue lock").next() else {
                    break;
                };
                let answer = classify(server, schema, row);
                let mut held = answers.lock().expect("evaluation answers lock");
                held[index] = Some(answer);
                let finished = held.iter().filter(|answer| answer.is_some()).count();
                println!("quantized lifecycle predictions {finished}/{}", held.len());
            })
        })
        .take(server.slots.min(rows.len()))
        .collect();
        for worker in workers {
            worker
                .join()
                .map_err(|_| Error("an evaluation worker panicked".to_string()))?;
        }
        Ok(())
    });
    server.stop();
    joined?;

    let answers = answers.into_inner().expect("evaluation answers lock");
    let mut scored = Vec::with_capacity(rows.len());
    let mut lines = String::new();
    for ((row, target), answer) in rows.iter().zip(&targets).zip(answers) {
        let (raw, prediction) =
            answer.ok_or_else(|| Error(format!("{} was never asked", row.id)))??;
        let action = prediction
            .as_ref()
            .is_some_and(|p| p.action == target.action);
        let goal_ref = prediction
            .as_ref()
            .is_some_and(|p| p.goal_ref == target.goal_ref);
        let evidence = prediction
            .as_ref()
            .is_some_and(|p| p.lifecycle_evidence == target.lifecycle_evidence);
        let joint = action && goal_ref && evidence;
        let record = json!({
            "id": row.id, "target": target, "input": input_envelope(row)?, "prediction": prediction, "raw": raw,
            "valid": prediction.is_some(), "action_correct": action, "goal_ref_correct": goal_ref,
            "evidence_correct": evidence, "joint_correct": joint, "metadata": row.metadata,
        });
        lines.push_str(&record.to_string());
        lines.push('\n');
        scored.push(Scored {
            action: target.action.clone(),
            valid: prediction.is_some(),
            action_correct: action,
            goal_ref_correct: goal_ref,
            evidence_correct: evidence,
            joint_correct: joint,
            finish: prediction
                .as_ref()
                .filter(|p| p.lifecycle_evidence == "explicit_completion")
                .map(|p| p.action == target.action),
        });
    }
    crate::util::write_whole(&run.predictions, lines.as_bytes())?;

    let total = scored.len();
    let finishes: Vec<bool> = scored.iter().filter_map(|row| row.finish).collect();
    let mut by_action: BTreeMap<&str, Value> = BTreeMap::new();
    for row in &scored {
        by_action.entry(row.action.as_str()).or_insert_with(|| {
            let own: Vec<&Scored> = scored
                .iter()
                .filter(|other| other.action == row.action)
                .collect();
            json!({
                "action_correct": counted(own.iter().copied(), |row| row.action_correct),
                "joint_correct": counted(own.iter().copied(), |row| row.joint_correct),
                "rows": own.len(),
            })
        });
    }
    let report = json!({
        "rows": total,
        "model": run.model.display().to_string(),
        "valid_json": rate(counted(&scored, |row| row.valid), total),
        "action_accuracy": rate(counted(&scored, |row| row.action_correct), total),
        "goal_ref_accuracy": rate(counted(&scored, |row| row.goal_ref_correct), total),
        "evidence_accuracy": rate(counted(&scored, |row| row.evidence_correct), total),
        "joint_accuracy": rate(counted(&scored, |row| row.joint_correct), total),
        "finish_precision": rate(finishes.iter().filter(|correct| **correct).count(), finishes.len()),
        "by_action": by_action,
    });
    crate::util::write_json_pretty(&run.metrics, &report)?;
    Ok(report)
}
