use super::*;

pub fn audit_predictions(input: &Path, output: &Path, model: &str) -> Result<Value> {
    let predictions = read_predictions(input)?;
    if predictions.is_empty() {
        return Err(Error("lifecycle predictions input is empty".to_string()));
    }
    let client = BramaClient::from_env()?;
    let predictions = Arc::new(predictions);
    let next = Arc::new(AtomicUsize::new(0));
    let aborted = Arc::new(AtomicBool::new(false));
    let results: Arc<Mutex<Vec<Option<Result<AuditDecision>>>>> =
        Arc::new(Mutex::new((0..predictions.len()).map(|_| None).collect()));
    // The local Brama route accounts each request as two concurrency units and the
    // operator plan currently exposes four. More workers make a healthy route
    // deterministically return 429 and abort the entire final audit.
    let workers = usize::from(2_u8).min(predictions.len());
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
    let maximum_wrong = predictions.len() / 50;
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
