use super::*;

use std::collections::{BTreeMap, VecDeque};

/// The immutable dataset split a review records.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Split {
    Train,
    Eval,
}

/// What a panicked worker said, in its own words when it gave any.
fn panic_words(panic: &(dyn std::any::Any + Send)) -> String {
    match (panic.downcast_ref::<&str>(), panic.downcast_ref::<String>()) {
        (Some(words), _) => (*words).to_string(),
        (None, Some(words)) => words.clone(),
        (None, None) => "a panic that carried no message".to_string(),
    }
}

/// Review the lifecycle dataset with `model`, as many calls at once as Brama
/// measured that route to carry.
pub fn review_dataset(
    input: &Path,
    output: &Path,
    split: &str,
    model: &str,
    limit: Option<usize>,
) -> Result<Value> {
    serde_json::from_value::<Split>(Value::String(split.to_string()))
        .map_err(|error| Error(format!("--split must be train or eval: {error}")))?;
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
            "contract": LIFECYCLE_CONTRACT
        }));
    }
    let client = BramaClient::from_env()?;
    let allowance = client.allowance(model)?.get();
    let total = rows.len();
    let queue = Arc::new(Mutex::new(
        rows.into_iter().enumerate().collect::<VecDeque<_>>(),
    ));
    let results: Arc<Mutex<BTreeMap<usize, Result<(Value, String)>>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
    let handles: Vec<_> = std::iter::repeat_with(|| {
        let client = client.clone();
        let queue = Arc::clone(&queue);
        let results = Arc::clone(&results);
        let model = model.to_string();
        let split = split.to_string();
        thread::spawn(move || loop {
            let Some((index, row)) = queue
                .lock()
                .expect("lifecycle review queue lock")
                .pop_front()
            else {
                break;
            };
            let result = classify(&client, &model, &row).and_then(|decision| {
                let action = decision.action.clone();
                reviewed_row(row, decision, &model, &split).map(|value| (value, action))
            });
            results
                .lock()
                .expect("lifecycle result lock")
                .insert(index, result);
        })
    })
    .take(allowance.min(total))
    .collect();
    for handle in handles {
        handle.join().map_err(|panic| {
            Error(format!(
                "a lifecycle review worker panicked: {}",
                panic_words(&*panic)
            ))
        })?;
    }
    let results = std::mem::take(&mut *results.lock().expect("lifecycle result lock"));
    let file = OpenOptions::new().create(true).append(true).open(output)?;
    let mut writer = BufWriter::new(file);
    let mut by_action: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut failures = Vec::new();
    if results.len() != total {
        failures.push(format!(
            "lifecycle review answered {} of {total} rows",
            results.len()
        ));
    }
    for result in results.into_values() {
        let (value, action) = match result {
            Ok(reviewed) => reviewed,
            Err(error) => {
                failures.push(error.to_string());
                continue;
            }
        };
        let id = value["id"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| Error("a reviewed lifecycle row has no id".to_string()))?;
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
        by_action.entry(action).or_default().push(id);
    }
    writer.flush()?;
    let written: usize = by_action.values().map(Vec::len).sum();
    if !failures.is_empty() {
        return Err(Error(format!(
            "lifecycle review failed for {} row(s); preserved {} newly reviewed row(s) in {}: {}",
            failures.len(),
            written,
            output.display(),
            failures.join("; ")
        )));
    }
    let counts: serde_json::Map<String, Value> = by_action
        .into_iter()
        .map(|(action, ids)| (action, Value::from(ids.len())))
        .collect();
    Ok(serde_json::json!({
        "input": input,
        "output": output,
        "split": split,
        "review_model": model,
        "rows": existing + written,
        "resumed": existing,
        "actions": counts,
        "contract": LIFECYCLE_CONTRACT
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
        .filter(|line| !line.as_ref().is_ok_and(|value| value.trim().is_empty()))
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
