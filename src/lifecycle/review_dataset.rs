use super::*;

pub fn review_dataset(
    input: &Path,
    output: &Path,
    split: &str,
    model: &str,
    limit: Option<usize>,
) -> Result<Value> {
    if !["train", "eval"].contains(&split) {
        return Err(Error("--split must be train or eval".to_string()));
    }
    // A reviewed label is ground truth for every model trained from it, so the
    // reviewer must use an operator-approved judgment route, not a
    // product-serving route.
    if model.starts_with("wisent-backend/") {
        return Err(Error(format!(
            "{model} is a product route, not a label reviewer: pass an operator-approved \
             Brama route with --brama-model, or sign the subscription in so the default \
             route answers"
        )));
    }
    let mut rows = read_rows(input)?;
    if let Some(limit) = limit {
        rows.truncate(limit);
    }
    if rows.is_empty() {
        return Err(Error("lifecycle dataset input is empty".to_string()));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let existing_ids = reviewed_ids(output)?;
    let existing = existing_ids.len();
    rows.retain(|row| !existing_ids.contains(&row.id));
    if rows.is_empty() {
        return Ok(serde_json::json!({
            "input": input,
            "output": output,
            "split": split,
            "review_model": model,
            "rows": existing,
            "resumed": existing,
            "contract": "oko-goal-lifecycle-v1"
        }));
    }
    let client = BramaClient::from_env()?;
    let rows = Arc::new(rows);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<Option<Result<Value>>>>> =
        Arc::new(Mutex::new((0..rows.len()).map(|_| None).collect()));
    let workers = env::var("LIFECYCLE_REVIEW_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(WORKERS)
        .min(rows.len());
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let client = client.clone();
        let rows = Arc::clone(&rows);
        let next = Arc::clone(&next);
        let results = Arc::clone(&results);
        let model = model.to_string();
        let split = split.to_string();
        handles.push(thread::spawn(move || loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= rows.len() {
                break;
            }
            let row = rows[index].clone();
            let result = classify(&client, &model, &row)
                .and_then(|decision| reviewed_row(row, decision, &model, &split));
            results.lock().expect("lifecycle result lock")[index] = Some(result);
        }));
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| Error("lifecycle review worker panicked".to_string()))?;
    }
    let file = OpenOptions::new().create(true).append(true).open(output)?;
    let mut writer = BufWriter::new(file);
    let mut counts = serde_json::Map::new();
    let mut written = 0usize;
    let mut failures = Vec::new();
    for result in results.lock().expect("lifecycle result lock").iter_mut() {
        let Some(result) = result.take() else {
            failures.push("lifecycle review produced a missing row".to_string());
            continue;
        };
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                failures.push(error.to_string());
                continue;
            }
        };
        let action = value
            .pointer("/messages/2/content")
            .and_then(Value::as_str)
            .and_then(|content| serde_json::from_str::<Value>(content).ok())
            .and_then(|decision| {
                decision
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "invalid".to_string());
        let count = counts.entry(action).or_insert(Value::from(0));
        *count = Value::from(count.as_u64().unwrap_or(0) + 1);
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
        written += 1;
    }
    writer.flush()?;
    if !failures.is_empty() {
        return Err(Error(format!(
            "lifecycle review failed for {} row(s); preserved {} newly reviewed row(s) in {}: {}",
            failures.len(),
            written,
            output.display(),
            failures[0]
        )));
    }
    Ok(serde_json::json!({
        "input": input,
        "output": output,
        "split": split,
        "review_model": model,
        "rows": existing + written,
        "resumed": existing,
        "actions": counts,
        "contract": "oko-goal-lifecycle-v1"
    }))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct AuditPrediction {
    pub(crate) id: String,
    pub(crate) input: Value,
    pub(crate) target: Decision,
    pub(crate) prediction: Option<Decision>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct AuditDecision {
    pub(crate) verdict: String,
    pub(crate) dangerous_finish: bool,
}

pub(crate) fn read_predictions(path: &Path) -> Result<Vec<AuditPrediction>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .filter(|line| line.as_ref().map_or(true, |value| !value.trim().is_empty()))
        .map(|line| {
            let line = line?;
            serde_json::from_str(&line).map_err(Error::from)
        })
        .collect()
}

pub(crate) fn audit_one(
    client: &BramaClient,
    model: &str,
    prediction: &AuditPrediction,
) -> Result<AuditDecision> {
    let request = [
        Message {
            role: "system".to_string(),
            content: format!(
                "You are the independent final judge for Oko's goal-lifecycle model. \
                 Read the contract and the masked input. Decide whether the student decision \
                 is semantically safe and defensible. A different candidate reference is wrong \
                 when it points at a different goal. Any inferred completion without explicit \
                 successful-completion evidence is dangerous. Start-goal title wording is not \
                 lifecycle correctness: Oko replaces it with its separate title model, so ignore \
                 title differences when the student title satisfies the output contract. The \
                 reference decision is evidence, not authority. Return only strict JSON: \
                 {{\"verdict\":\"student-sensible|student-wrong|unjudgeable\",\
                 \"dangerous_finish\":false}}.\n\n{SYSTEM_PROMPT}"
            ),
        },
        Message {
            role: "user".to_string(),
            content: serde_json::to_string(&serde_json::json!({
                "input": prediction.input,
                "reference": prediction.target,
                "student": prediction.prediction,
            }))?,
        },
    ];
    let decision = client
        .chat(model, &request)
        .and_then(|answer| parse_json_object(&answer))
        .and_then(|value| serde_json::from_value::<AuditDecision>(value).map_err(Error::from))
        .map_err(|error| Error(format!("{}: {error}", prediction.id)))?;
    match decision.verdict.as_str() {
        "student-sensible" | "student-wrong" | "unjudgeable" => Ok(decision),
        other => Err(Error(format!("{} has an unknown audit verdict {other:?}", prediction.id))),
    }
}
