use super::*;

/// Audits every prediction with as many concurrent Brama calls as Brama
/// measured `model`'s route to carry, and passes when no decision is
/// semantically wrong, unjudgeable or a dangerous finish and no audit call
/// failed. No tolerated share of wrong decisions was ever stated, so none is
/// tolerated. The report is written whole.
pub fn audit_predictions(input: &Path, output: &Path, model: &str) -> Result<Value> {
    let predictions = read_predictions(input)?;
    if predictions.is_empty() {
        return Err(Error("lifecycle predictions input is empty".to_string()));
    }
    let client = BramaClient::from_env()?;
    let workers = client.allowance(model)?.get().min(predictions.len());
    let queue = Mutex::new(predictions.iter().enumerate());
    let aborted = AtomicBool::new(false);
    let results: Mutex<Vec<Option<Result<AuditDecision>>>> =
        Mutex::new(predictions.iter().map(|_| None).collect());
    thread::scope(|scope| -> Result<()> {
        let (client, queue, aborted, results) = (&client, &queue, &aborted, &results);
        let handles: Vec<_> = std::iter::repeat_with(|| {
            scope.spawn(move || loop {
                if aborted.load(Ordering::Relaxed) {
                    break;
                }
                let Some((index, prediction)) =
                    queue.lock().expect("lifecycle audit queue lock").next()
                else {
                    break;
                };
                let result = audit_one(client, model, prediction);
                if result.is_err() {
                    aborted.store(true, Ordering::Relaxed);
                }
                results.lock().expect("lifecycle audit result lock")[index] = Some(result);
            })
        })
        .take(workers)
        .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| Error("lifecycle audit worker panicked".to_string()))?;
        }
        Ok(())
    })?;

    let mut decisions = Vec::with_capacity(predictions.len());
    let mut records = Vec::with_capacity(predictions.len());
    let mut failures = Vec::new();
    let results = results.into_inner().expect("lifecycle audit result lock");
    for (prediction, result) in predictions.iter().zip(results) {
        let error = match result {
            Some(Ok(decision)) => {
                records.push(serde_json::json!({ "id": prediction.id, "decision": decision }));
                decisions.push(decision);
                continue;
            }
            Some(Err(error)) => error.to_string(),
            None => "audit skipped after an earlier audit failure".to_string(),
        };
        records.push(serde_json::json!({ "id": prediction.id, "error": error }));
        failures.push(error);
    }
    let verdicts = |verdict: &str| {
        decisions
            .iter()
            .filter(|decision| decision.verdict == verdict)
            .count()
    };
    let sensible = verdicts("student-sensible");
    let wrong = verdicts("student-wrong");
    let unjudgeable = decisions.len() - sensible - wrong;
    let dangerous: Vec<&AuditDecision> = decisions
        .iter()
        .filter(|decision| decision.dangerous_finish)
        .collect();
    let passed = sensible == decisions.len() && dangerous.is_empty() && failures.is_empty();
    let report = serde_json::json!({
        "created_at": now_iso(),
        "review_model": model,
        "input": input,
        "counts": {
            "total": predictions.len(),
            "student_sensible": sensible,
            "student_wrong": wrong,
            "unjudgeable": unjudgeable,
            "dangerous_finish": dangerous.len(),
            "audit_errors": failures.len(),
        },
        "passed": passed,
        "failures": failures,
        "records": records,
    });
    crate::util::write_json_pretty(output, &report)?;
    Ok(report)
}
