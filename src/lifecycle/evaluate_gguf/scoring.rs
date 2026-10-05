//! The evaluation run: every row asked on `parallel` workers, one progress
//! line per finished row, then the predictions file and the metrics, which
//! report each measured rate and leave out a rate with nothing to measure.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;

use super::*;

fn rate(count: usize, total: usize) -> Value {
    if total == 0 { Value::Null } else { json!(count as f64 / total as f64) }
}

pub fn evaluate_gguf(run: &GgufEvaluation) -> Result<Value> {
    for (what, path) in [("model", &run.model), ("dataset", &run.dataset), ("server", &run.server), ("output schema", &run.output_schema)] {
        if !path.is_file() {
            return Err(Error(format!("the GGUF evaluation {what} {} does not exist", path.display())));
        }
    }
    if run.parallel == 0 || run.slot_context == 0 {
        return Err(Error("--parallel and --slot-context must be at least 1".to_string()));
    }
    let schema: Value = serde_json::from_str(&fs::read_to_string(&run.output_schema)?)?;
    let rows = read_rows(&run.dataset)?;
    if rows.is_empty() {
        return Err(Error(format!("{} holds no evaluation row", run.dataset.display())));
    }
    let targets = rows.iter().map(|row| target(row, &schema)).collect::<Result<Vec<_>>>()?;
    let port = free_port()?;
    let mut server = start_server(run, port)?;
    let endpoint = format!("http://{}:{port}/v1/chat/completions", Ipv4Addr::LOCALHOST);
    let served = alias(&run.model);
    let rows = Arc::new(rows);
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let answers: Arc<Mutex<Vec<Option<Result<(String, Option<Decision>)>>>>> = Arc::new(Mutex::new((0..rows.len()).map(|_| None).collect()));
    let client = waiting_client()?;
    let workers: Vec<_> = (0..run.parallel.min(rows.len()))
        .map(|_| {
            let (rows, next, done, answers) = (Arc::clone(&rows), Arc::clone(&next), Arc::clone(&done), Arc::clone(&answers));
            let (client, endpoint, served, schema) = (client.clone(), endpoint.clone(), served.clone(), schema.clone());
            thread::spawn(move || loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some(row) = rows.get(index) else { break };
                let answer = classify(&client, &endpoint, &served, &schema, row);
                answers.lock().expect("evaluation answers lock")[index] = Some(answer);
                println!("quantized lifecycle predictions {}/{}", done.fetch_add(1, Ordering::SeqCst) + 1, rows.len());
            })
        })
        .collect();
    for worker in workers {
        worker.join().map_err(|_| Error("an evaluation worker panicked".to_string()))?;
    }
    let _ = server.kill();
    let _ = server.wait();

    let answers = std::mem::take(&mut *answers.lock().expect("evaluation answers lock"));
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_action: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut lines = String::new();
    for ((row, target), answer) in rows.iter().zip(&targets).zip(answers) {
        let (raw, prediction) = answer.ok_or_else(|| Error(format!("{} was never asked", row.id)))??;
        let action = prediction.as_ref().is_some_and(|p| p.action == target.action);
        let goal_ref = prediction.as_ref().is_some_and(|p| p.goal_ref == target.goal_ref);
        let evidence = prediction.as_ref().is_some_and(|p| p.lifecycle_evidence == target.lifecycle_evidence);
        let joint = action && goal_ref && evidence;
        for (key, hit) in [("valid_json", prediction.is_some()), ("action_correct", action), ("goal_ref_correct", goal_ref), ("evidence_correct", evidence), ("joint_correct", joint)] {
            *counts.entry(key).or_default() += usize::from(hit);
        }
        let bucket = by_action.entry(target.action.clone()).or_default();
        *bucket.entry("rows").or_default() += 1;
        *bucket.entry("action_correct").or_default() += usize::from(action);
        *bucket.entry("joint_correct").or_default() += usize::from(joint);
        if let Some(predicted) = &prediction {
            if predicted.lifecycle_evidence == "explicit_completion" {
                *counts.entry("predicted_finish").or_default() += 1;
                *counts.entry("correct_finish").or_default() += usize::from(predicted.action == target.action);
            }
        }
        let record = json!({
            "id": row.id, "target": target, "input": input_envelope(row)?, "prediction": prediction, "raw": raw,
            "valid": prediction.is_some(), "action_correct": action, "goal_ref_correct": goal_ref,
            "evidence_correct": evidence, "joint_correct": joint, "metadata": row.metadata,
        });
        lines.push_str(&record.to_string());
        lines.push('\n');
    }
    fs::write(&run.predictions, lines)?;
    let total = rows.len();
    let count = |key: &str| counts.get(key).copied().unwrap_or(0);
    let report = json!({
        "rows": total,
        "model": run.model.display().to_string(),
        "valid_json": rate(count("valid_json"), total),
        "action_accuracy": rate(count("action_correct"), total),
        "goal_ref_accuracy": rate(count("goal_ref_correct"), total),
        "evidence_accuracy": rate(count("evidence_correct"), total),
        "joint_accuracy": rate(count("joint_correct"), total),
        "finish_precision": rate(count("correct_finish"), count("predicted_finish")),
        "by_action": by_action,
    });
    fs::write(&run.metrics, serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(report)
}
