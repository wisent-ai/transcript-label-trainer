use super::*;

/// Score the trained model on its frozen holdout and have Brama judge it.
///
/// `name` is a job name or a bare aspect — both are directory names under
/// `$TLT_HOME/models/`. Fails when there is no frozen holdout, when nothing is
/// trained, and when the gateway could not judge a single session.
pub fn evaluate(
    name: &str,
    judge: Option<bool>,
    judge_model: Option<&str>,
    best_review: bool,
) -> Result<Value> {
    let artifact = model::active_artifact(name)?;
    let metrics = &artifact.metrics;
    let aspect = metrics
        .get("aspect")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let empty = Value::Object(Map::new());
    let job_meta = match metrics.get("job") {
        Some(value) if json_truthy(value) => value.clone(),
        _ => empty.clone(),
    };
    let spec_judge = match job_meta.get("judge") {
        Some(value) if json_truthy(value) => value.clone(),
        _ => serde_json::to_value(jobs::default_judge())?,
    };
    let judge_enabled = judge.unwrap_or_else(|| {
        spec_judge
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    });
    let model_id = judge_model
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            spec_judge
                .get("model")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| brama::DEFAULT_MODEL.to_string());
    if best_review && !judge_enabled {
        return Err(Error(
            "--best requires the Brama judge to be enabled".to_string(),
        ));
    }

    let frozen = read_split(name)?;
    let labels = model::labels_for_artifact(metrics)?;
    let by_id: HashMap<&str, &lake::SessionLabel> = labels
        .iter()
        .map(|label| (label.session_id.as_str(), label))
        .collect();
    let holdout_ids: Vec<String> = frozen
        .session_ids
        .iter()
        .filter(|session_id| by_id.contains_key(session_id.as_str()))
        .cloned()
        .collect();
    let missing = frozen.session_ids.len() - holdout_ids.len();
    if holdout_ids.is_empty() {
        return Err(Error(format!(
            "none of the {} frozen holdout session(s) in {} still carry a \
             ground-truth label for aspect '{aspect}'; there is nothing to evaluate",
            frozen.session_ids.len(),
            split_path(name)?.display()
        )));
    }

    let texts = lake::session_texts(&holdout_ids)?;
    let usable: Vec<String> = holdout_ids
        .iter()
        .filter(|session_id| {
            texts
                .get(session_id.as_str())
                .is_some_and(|entry| !entry.text.trim().is_empty())
        })
        .cloned()
        .collect();
    let no_text = holdout_ids.len() - usable.len();
    if no_text > 0 {
        lake::warn(&format!(
            "{no_text} frozen holdout session(s) had no text in the lake"
        ));
    }
    if usable.is_empty() {
        return Err(Error(format!(
            "none of the {} frozen holdout session(s) have text in the lake; \
             there is nothing to evaluate",
            holdout_ids.len()
        )));
    }

    let gold: Vec<String> = usable
        .iter()
        .map(|session_id| by_id[session_id.as_str()].value.clone())
        .collect();
    let holdout_texts: Vec<String> = usable
        .iter()
        .map(|session_id| texts[session_id.as_str()].text.clone())
        .collect();
    let predictions = model::predict(&artifact, &holdout_texts)?;
    let report = holdout_report(&gold, &predictions);

    let sessions: Vec<Value> = usable
        .iter()
        .zip(&gold)
        .zip(&predictions)
        .map(|((session_id, actual), (predicted, confidence))| {
            let mut record = Map::new();
            record.insert("session_id".to_string(), Value::String(session_id.clone()));
            record.insert(
                "runtime".to_string(),
                texts[session_id.as_str()]
                    .runtime
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            record.insert("gold".to_string(), Value::String(actual.clone()));
            record.insert("prediction".to_string(), Value::String(predicted.clone()));
            record.insert("confidence".to_string(), number(round4(*confidence)));
            record.insert("correct".to_string(), Value::Bool(predicted == actual));
            Value::Object(record)
        })
        .collect();

    let mut eval_split = Map::new();
    eval_split.insert(
        "path".to_string(),
        Value::String(split_path(name)?.to_string_lossy().into_owned()),
    );
    eval_split.insert(
        "fraction".to_string(),
        frozen.fraction.map(number).unwrap_or(Value::Null),
    );
    eval_split.insert(
        "seed".to_string(),
        frozen
            .seed
            .map(|seed| Value::Number(seed.into()))
            .unwrap_or(Value::Null),
    );
    eval_split.insert(
        "created_at".to_string(),
        frozen
            .created_at
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    eval_split.insert(
        "frozen_sessions".to_string(),
        Value::Number(frozen.session_ids.len().into()),
    );
    eval_split.insert(
        "missing_ground_truth".to_string(),
        Value::Number(missing.into()),
    );
    eval_split.insert("skipped_no_text".to_string(), Value::Number(no_text.into()));

    let mut result = Map::new();
    result.insert("name".to_string(), Value::String(name.to_string()));
    result.insert("aspect".to_string(), Value::String(aspect.clone()));
    result.insert(
        "backend".to_string(),
        Value::String(artifact.backend.clone()),
    );
    result.insert(
        "model_path".to_string(),
        metrics.get("model_path").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "trained_at".to_string(),
        metrics.get("trained_at").cloned().unwrap_or(Value::Null),
    );
    result.insert("evaluated_at".to_string(), Value::String(now_iso()));
    result.insert("eval_split".to_string(), Value::Object(eval_split));
    result.insert("holdout_evaluation".to_string(), report);

    if !judge_enabled {
        let mut block = Map::new();
        block.insert("enabled".to_string(), Value::Bool(false));
        result.insert("judge".to_string(), Value::Object(block));
        result.insert("sessions".to_string(), Value::Array(sessions));
        return Ok(Value::Object(result));
    }

    let client = brama::BramaClient::from_env()?;
    let task = job_meta.get("task").and_then(Value::as_str);
    let (mut records, failures) =
        judge_sessions(&client, &model_id, &aspect, task, &sessions, &texts);
    if records.is_empty() {
        // No usable provider route: surface the gateway's own words and write
        // nothing. A verdict nobody produced is not a verdict.
        let first = failures
            .first()
            .and_then(|entry| entry.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("(no sessions to judge)")
            .to_string();
        return Err(Error(format!(
            "the Brama judge ({model_id}) could not judge any of the {} holdout \
             session(s); first error: {first}",
            sessions.len()
        )));
    }

    let acceptable = records
        .iter()
        .filter(|record| record.get("verdict").and_then(Value::as_str) == Some(JUDGE_VALUES[0]))
        .count();
    let mut block = Map::new();
    block.insert("enabled".to_string(), Value::Bool(true));
    block.insert("model".to_string(), Value::String(model_id.clone()));
    block.insert("judged".to_string(), Value::Number(records.len().into()));
    block.insert("acceptable".to_string(), Value::Number(acceptable.into()));
    block.insert(
        "unacceptable".to_string(),
        Value::Number((records.len() - acceptable).into()),
    );
    block.insert("failed".to_string(), Value::Number(failures.len().into()));
    block.insert(
        "agreement_rate".to_string(),
        number(round4(acceptable as f64 / records.len() as f64)),
    );
    result.insert("judge".to_string(), Value::Object(block));
    if best_review {
        record_best_review(&client, &aspect, task, &mut records, &texts, &failures, &mut result)?;
    }
    result.insert("sessions".to_string(), Value::Array(records));
    result.insert("failures".to_string(), Value::Array(failures));

    let path = judge_path(name)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&Value::Object(result.clone()))?;
    std::fs::write(&path, body + "\n")?;
    result.insert(
        "judge_path".to_string(),
        Value::String(path.to_string_lossy().into_owned()),
    );
    Ok(Value::Object(result))
}
