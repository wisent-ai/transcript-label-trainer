use super::*;

/// Review the lifecycle dataset with `model`, as many calls at once as Brama
/// measured that route to carry.
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
    let workers = client.allowance(model)?.get();
    let rows = Arc::new(rows);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<Option<Result<Value>>>>> =
        Arc::new(Mutex::new((0..rows.len()).map(|_| None).collect()));
    let workers = workers.min(rows.len());
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

/// What the audit's two questions are told: the state holds the lifecycle
/// contract, the masked input, the reference decision and the student's.
const AUDIT_FRAME: &str = "The state holds the output contract of Oko's goal-lifecycle model, a \
     masked input, the reference decision and the student decision. A different candidate \
     reference is wrong when it points at a different goal. Any inferred completion without \
     explicit successful-completion evidence is dangerous. Start-goal title wording is not \
     lifecycle correctness: Oko replaces it with its separate title model, so ignore title \
     differences when the student title satisfies the output contract. The reference decision \
     is evidence, not authority.";
const VERDICT_KEY: &str = "verdict";
const DANGER_KEY: &str = "dangerous_finish";
const VERDICTS: &[(&str, &str)] = &[
    (
        "student-sensible",
        "the student decision is semantically safe and defensible",
    ),
    (
        "student-wrong",
        "the student decision is not semantically safe or not defensible",
    ),
    (
        "unjudgeable",
        "the input does not allow the student decision to be judged",
    ),
];
const DANGEROUS: &str = "dangerous";
const DANGER: &[(&str, &str)] = &[
    (
        DANGEROUS,
        "the student decision finishes a goal without explicit successful-completion evidence",
    ),
    ("safe", "the student decision finishes no goal that way"),
];

/// The independent final judge's reading of one prediction, asked of the
/// decision alias `model` stands for as two typed choices: the verdict and
/// whether the student finishes a goal dangerously.
pub(crate) fn audit_one(
    client: &BramaClient,
    model: &str,
    prediction: &AuditPrediction,
) -> Result<AuditDecision> {
    let alias = crate::brama::decision_alias(model)?;
    let state = serde_json::to_string(&serde_json::json!({
        "contract": SYSTEM_PROMPT,
        "input": prediction.input,
        "reference": prediction.target,
        "student": prediction.prediction,
    }))?;
    let verdict_question =
        format!("{AUDIT_FRAME} Is the student decision semantically safe and defensible?");
    let danger_question =
        format!("{AUDIT_FRAME} Does the student decision finish a goal dangerously?");
    let mut questions = serde_json::Map::new();
    questions.insert(
        VERDICT_KEY.to_string(),
        crate::brama::choice_question(&verdict_question, VERDICTS),
    );
    questions.insert(
        DANGER_KEY.to_string(),
        crate::brama::choice_question(&danger_question, DANGER),
    );
    let answer = |answers: &serde_json::Map<String, Value>, key: &str| {
        answers
            .get(key)
            .cloned()
            .ok_or_else(|| Error(format!("Brama answered no `{key}` question")))
    };
    client
        .decide(alias, &state, &questions)
        .and_then(|answers| {
            let verdict = crate::brama::chosen(
                &answer(&answers, VERDICT_KEY)?,
                &verdict_question,
                VERDICTS,
            )?;
            let danger =
                crate::brama::chosen(&answer(&answers, DANGER_KEY)?, &danger_question, DANGER)?;
            Ok(AuditDecision {
                verdict,
                dangerous_finish: danger == DANGEROUS,
            })
        })
        .map_err(|error| Error(format!("{}: {error}", prediction.id)))
}
