use super::*;

/// Audits every prediction with as many concurrent Brama calls as Brama
/// measured `model`'s route to carry, and passes when at most
/// `max_wrong_share` of them are semantically wrong, none is unjudgeable or a
/// dangerous finish, and no audit call failed. The tolerated share is the
/// operator's.
pub fn audit_predictions(
    input: &Path,
    output: &Path,
    model: &str,
    max_wrong_share: f64,
) -> Result<Value> {
    // A share lies between none and all; the sign of a positive share is one.
    let share = max_wrong_share.is_finite()
        && !max_wrong_share.is_sign_negative()
        && max_wrong_share <= max_wrong_share.signum();
    if !share {
        return Err(Error(format!(
            "--max-wrong-share must be a share from 0 to 1, not {max_wrong_share}"
        )));
    }
    let predictions = read_predictions(input)?;
    if predictions.is_empty() {
        return Err(Error("lifecycle predictions input is empty".to_string()));
    }
    let client = BramaClient::from_env()?;
    let workers = client.allowance(model)?;
    let predictions = Arc::new(predictions);
    let next = Arc::new(AtomicUsize::new(0));
    let aborted = Arc::new(AtomicBool::new(false));
    let results: Arc<Mutex<Vec<Option<Result<AuditDecision>>>>> =
        Arc::new(Mutex::new((0..predictions.len()).map(|_| None).collect()));
    let workers = workers.get().min(predictions.len());
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let client = client.clone();
        let predictions = Arc::clone(&predictions);
        let next = Arc::clone(&next);
        let aborted = Arc::clone(&aborted);
        let results = Arc::clone(&results);
        let model = model.to_string();
        handles.push(thread::spawn(move || loop {
            if aborted.load(Ordering::Relaxed) {
                break;
            }
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= predictions.len() {
                break;
            }
            let result = audit_one(&client, &model, &predictions[index]);
            let failed = result.is_err();
            results.lock().expect("lifecycle audit result lock")[index] = Some(result);
            if failed {
                aborted.store(true, Ordering::Relaxed);
            }
        }));
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| Error("lifecycle audit worker panicked".to_string()))?;
    }

    let mut records = Vec::with_capacity(predictions.len());
    let mut sensible = 0usize;
    let mut wrong = 0usize;
    let mut unjudgeable = 0usize;
    let mut dangerous_finish = 0usize;
    let mut failures = Vec::new();
    for (index, result) in results
        .lock()
        .expect("lifecycle audit result lock")
        .iter_mut()
        .enumerate()
    {
        match result.take() {
            Some(Ok(decision)) => {
                match decision.verdict.as_str() {
                    "student-sensible" => sensible += 1,
                    "student-wrong" => wrong += 1,
                    _ => unjudgeable += 1,
                }
                dangerous_finish += usize::from(decision.dangerous_finish);
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "decision": decision,
                }));
            }
            Some(Err(error)) => {
                failures.push(error.to_string());
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "error": error.to_string(),
                }));
            }
            None => {
                let error = "audit skipped after an earlier audit failure".to_string();
                failures.push(error.clone());
                records.push(serde_json::json!({
                    "id": predictions[index].id,
                    "error": error,
                }));
            }
        }
    }
    let maximum_wrong = (max_wrong_share * predictions.len() as f64).floor() as usize;
    let passed =
        wrong <= maximum_wrong && unjudgeable == 0 && dangerous_finish == 0 && failures.is_empty();
    let report = serde_json::json!({
        "created_at": now_iso(),
        "review_model": model,
        "input": input,
        "counts": {
            "total": predictions.len(),
            "student_sensible": sensible,
            "student_wrong": wrong,
            "unjudgeable": unjudgeable,
            "dangerous_finish": dangerous_finish,
            "audit_errors": failures.len(),
        },
        "thresholds": {
            "max_wrong_share": max_wrong_share,
            "maximum_student_wrong": maximum_wrong,
            "maximum_unjudgeable": 0,
            "maximum_dangerous_finish": 0,
            "maximum_audit_errors": 0,
        },
        "passed": passed,
        "failures": failures,
        "records": records,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("json.{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(&report)?)?;
    fs::rename(temporary, output)?;
    Ok(report)
}
