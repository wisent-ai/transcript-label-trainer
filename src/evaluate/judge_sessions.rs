use super::*;

/// What each judge verdict means, offered with the question so the answer is
/// one of them by construction.
fn judge_labels() -> Vec<(&'static str, &'static str)> {
    vec![
        (ACCEPTABLE, "the predicted label is a defensible reading of the transcript on the aspect, including when it differs from the recorded label but the session genuinely supports it"),
        (UNACCEPTABLE, "the predicted label is not a defensible reading of the transcript on the aspect"),
    ]
}

/// What each final-review outcome means.
fn best_review_labels() -> Vec<(&'static str, &'static str)> {
    vec![
        (BOTH_SENSIBLE, "the recorded label is a sensible reading of the transcript and the earlier verdict is sound"),
        (LABEL_NONSENSICAL, "only the recorded label is unsound"),
        (JUDGE_NONSENSICAL, "only the earlier judge verdict is unsound"),
        (BOTH_NONSENSICAL, "neither the recorded label nor the earlier verdict is sound"),
    ]
}

/// A text field every judged session record carries, or the record refused.
fn session_field<'a>(session: &'a Value, field: &str) -> std::result::Result<&'a str, String> {
    session
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("a judged session record has no {field}: {session}"))
}

/// The aspect, its purpose and the session's labels, as the judge reads them
/// beside the transcript.
fn judged_state(aspect: &str, task: Option<&str>, labels: &str, text: &str) -> String {
    let purpose = match task {
        Some(task) if !task.trim().is_empty() => format!("\nWhat the classifier is for: {task}"),
        _ => String::new(),
    };
    format!("Aspect: {aspect}{purpose}\n{labels}\n\nTranscript:\n{text}")
}

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
    let judge = match brama::decision_alias(judge_model) {
        Ok(judge) => judge,
        Err(error) => {
            failures.push(failure("", &error.0));
            return (records, failures);
        }
    };
    let labels = judge_labels();
    for session in sessions {
        let read = (|| -> std::result::Result<(String, String), String> {
            let session_id = session_field(session, "session_id")?;
            let gold = session_field(session, "gold")?;
            let prediction = session_field(session, "prediction")?;
            let text = texts
                .get(session_id)
                .map(|entry| entry.text.as_str())
                .ok_or_else(|| {
                    format!("session {session_id} has no transcript text in the lake")
                })?;
            let state = judged_state(
                aspect,
                task,
                &format!(
                    "Ground-truth label recorded by the evaluator: {gold}\nLabel predicted by the classifier: {prediction}"
                ),
                text,
            );
            Ok((session_id.to_string(), state))
        })();
        let (session_id, state) = match read {
            Ok(read) => read,
            Err(why) => {
                failures.push(failure("", &why));
                continue;
            }
        };
        match client.choose(
            judge,
            &state,
            "Is the predicted label a defensible reading of this transcript on this aspect?",
            &labels,
        ) {
            Ok(verdict) => {
                let mut record = session.clone();
                if let Some(object) = record.as_object_mut() {
                    object.insert("verdict".to_string(), Value::String(verdict));
                }
                records.push(record);
            }
            Err(error) => failures.push(failure(&session_id, &error.0)),
        }
    }
    (records, failures)
}

pub(crate) fn best_review_sessions(
    client: &brama::BramaClient,
    aspect: &str,
    task: Option<&str>,
    sessions: &mut [Value],
    texts: &HashMap<String, lake::SessionText>,
) -> Vec<Value> {
    let mut failures = Vec::new();
    let labels = best_review_labels();
    for session in sessions {
        let read = (|| -> std::result::Result<(String, String), String> {
            let session_id = session_field(session, "session_id")?;
            let gold = session_field(session, "gold")?;
            let prediction = session_field(session, "prediction")?;
            let verdict = session_field(session, "verdict")?;
            let text = texts
                .get(session_id)
                .map(|entry| entry.text.as_str())
                .ok_or_else(|| {
                    format!("session {session_id} has no transcript text in the lake")
                })?;
            let state = judged_state(
                aspect,
                task,
                &format!(
                    "Recorded ground-truth label: {gold}\nClassifier prediction: {prediction}\nEarlier judge verdict on that prediction: {verdict}"
                ),
                text,
            );
            Ok((session_id.to_string(), state))
        })();
        let (session_id, state) = match read {
            Ok(read) => read,
            Err(why) => {
                failures.push(failure("", &why));
                continue;
            }
        };
        match client.choose(
            brama::BEST_DECISION_MODEL,
            &state,
            "Decide independently whether the recorded label is a sensible reading of the transcript for this aspect, and whether the earlier judge verdict is sensible given the prediction and transcript.",
            &labels,
        ) {
            Ok(review) => {
                if let Some(object) = session.as_object_mut() {
                    object.insert("best_review".to_string(), Value::String(review));
                }
            }
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
    let review_failures = best_review_sessions(client, aspect, task, records, texts);
    let reviews: Vec<String> = records
        .iter()
        .filter_map(|record| record.get("best_review").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    if reviews.is_empty() {
        let cause = match review_failures.first().and_then(|entry| entry.get("error")) {
            Some(error) => format!("first error: {error}"),
            None => "there were no judge records to review".to_string(),
        };
        return Err(Error(format!(
            "the final Brama reviewer ({}) could not review any of the {} judge record(s); {cause}",
            brama::BEST_DECISION_MODEL,
            records.len()
        )));
    }
    let count = |wanted: &[&str]| {
        reviews
            .iter()
            .filter(|review| wanted.contains(&review.as_str()))
            .count()
    };
    let both_sensible_count = count(&[BOTH_SENSIBLE]);
    let sensible =
        failures.is_empty() && review_failures.is_empty() && both_sensible_count == records.len();
    result.insert(
        "best_review".to_string(),
        serde_json::json!({
            "enabled": true,
            "model": brama::BEST_DECISION_MODEL,
            "reviewed": reviews.len(),
            "both_sensible": both_sensible_count,
            "label_nonsensical": count(&[LABEL_NONSENSICAL, BOTH_NONSENSICAL]),
            "judge_nonsensical": count(&[JUDGE_NONSENSICAL, BOTH_NONSENSICAL]),
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
