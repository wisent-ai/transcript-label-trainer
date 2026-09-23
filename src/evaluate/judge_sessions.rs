use super::*;

/// One verdict per holdout session; a Brama error fails only its session.
pub(crate) fn judge_sessions(
    client: &brama::BramaClient,
    judge_model: &str,
    aspect: &str,
    task: Option<&str>,
    sessions: &[Value],
    texts: &HashMap<String, lake::SessionText>,
) -> (Vec<Value>, Vec<Value>) {
    let mut records: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    for session in sessions {
        let session_id = session
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let gold = session
            .get("gold")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let prediction = session
            .get("prediction")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = texts
            .get(session_id)
            .map(|entry| entry.text.as_str())
            .unwrap_or_default();
        let prompt = judge_prompt(aspect, task, gold, prediction, text);
        match client.chat(judge_model, &prompt) {
            Ok(answer) => match brama::parse_answer(&answer, &JUDGE_VALUES[..]) {
                Some((verdict, _exact)) => {
                    let mut record = session.clone();
                    if let Some(object) = record.as_object_mut() {
                        object.insert("verdict".to_string(), Value::String(verdict));
                    }
                    records.push(record);
                }
                None => failures.push(failure(
                    session_id,
                    &format!(
                        "unparseable judge answer: {}",
                        jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                    ),
                )),
            },
            Err(error) => failures.push(failure(session_id, &error.0)),
        }
    }
    (records, failures)
}

pub(crate) fn best_review_prompt(
    aspect: &str,
    task: Option<&str>,
    gold: &str,
    prediction: &str,
    verdict: &str,
    text: &str,
) -> Vec<brama::Message> {
    let purpose = match task {
        Some(task) if !task.is_empty() => format!("\nWhat the classifier is for: {task}"),
        _ => String::new(),
    };
    vec![
        brama::Message {
            role: "system".to_string(),
            content: format!(
                "You are the final semantic auditor for transcript labels and \
                 another judge's opinion. Answer with exactly one of: {}.",
                BEST_REVIEW_VALUES.join(", ")
            ),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}{purpose}\n\
                 Recorded ground-truth label: {gold}\n\
                 Classifier prediction: {prediction}\n\
                 Earlier judge verdict on that prediction: {verdict}\n\n\
                 Decide independently whether (1) the recorded label is a \
                 sensible reading of the transcript for this aspect, and \
                 (2) the earlier judge verdict is sensible given the prediction \
                 and transcript. Answer both-sensible when both are sound, \
                 label-nonsensical when only the label is unsound, \
                 judge-nonsensical when only the earlier verdict is unsound, or \
                 both-nonsensical when neither is sound.\n\n\
                 Transcript:\n{text}"
            ),
        },
    ]
}

pub(crate) fn best_review_sessions(
    client: &brama::BramaClient,
    aspect: &str,
    task: Option<&str>,
    sessions: &mut [Value],
    texts: &HashMap<String, lake::SessionText>,
) -> Vec<Value> {
    let mut failures = Vec::new();
    for session in sessions {
        let session_id = session
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let gold = session
            .get("gold")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let prediction = session
            .get("prediction")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let verdict = session
            .get("verdict")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let text = texts
            .get(&session_id)
            .map(|entry| entry.text.as_str())
            .unwrap_or_default();
        let prompt = best_review_prompt(aspect, task, &gold, &prediction, &verdict, text);
        match client.chat(brama::BEST_MODEL, &prompt) {
            Ok(answer) => match brama::parse_answer(&answer, &BEST_REVIEW_VALUES[..]) {
                Some((review, _exact)) => {
                    if let Some(object) = session.as_object_mut() {
                        object.insert("best_review".to_string(), Value::String(review));
                    }
                }
                None => failures.push(failure(
                    &session_id,
                    &format!(
                        "unparseable final review answer: {}",
                        jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                    ),
                )),
            },
            Err(error) => failures.push(failure(&session_id, &error.0)),
        }
    }
    failures
}

/// Adds the final reviewer's pass to a judged evaluation: every judge record is
/// reviewed by the best model and what it found is counted into `result`.
pub(crate) fn record_best_review(
    client: &brama::BramaClient,
    aspect: &str,
    task: Option<&str>,
    records: &mut [Value],
    texts: &HashMap<String, lake::SessionText>,
    failures: &[Value],
    result: &mut Map<String, Value>,
) -> Result<()> {
    let review_failures = best_review_sessions(&client, &aspect, task, records, &texts);
    let reviewed = records
        .iter()
        .filter(|record| record.get("best_review").and_then(Value::as_str).is_some())
        .count();
    if reviewed == 0 {
        let first = review_failures
            .first()
            .and_then(|entry| entry.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("(no sessions to review)");
        return Err(Error(format!(
            "the final Brama reviewer ({}) could not review any of the {} \
             judge record(s); first error: {first}",
            brama::BEST_MODEL,
            records.len()
        )));
    }
    let both_sensible = records
        .iter()
        .filter(|record| {
            record.get("best_review").and_then(Value::as_str) == Some(BEST_REVIEW_VALUES[0])
        })
        .count();
    let label_nonsensical = records
        .iter()
        .filter(|record| {
            matches!(
                record.get("best_review").and_then(Value::as_str),
                Some("label-nonsensical" | "both-nonsensical")
            )
        })
        .count();
    let judge_nonsensical = records
        .iter()
        .filter(|record| {
            matches!(
                record.get("best_review").and_then(Value::as_str),
                Some("judge-nonsensical" | "both-nonsensical")
            )
        })
        .count();
    let sensible =
        failures.is_empty() && review_failures.is_empty() && both_sensible == records.len();
    result.insert(
        "best_review".to_string(),
        serde_json::json!({
            "enabled": true,
            "model": brama::BEST_MODEL,
            "reviewed": reviewed,
            "both_sensible": both_sensible,
            "label_nonsensical": label_nonsensical,
            "judge_nonsensical": judge_nonsensical,
            "failed": review_failures.len(),
            "sensible": sensible,
        }),
    );
    result.insert(
        "best_review_failures".to_string(),
        Value::Array(review_failures),
    );
    Ok(())
}
