use super::*;

pub fn infer(aspect: &str, session: Option<&str>, limit: Option<i64>) -> Result<Value> {
    let artifact = active_artifact(aspect)?;

    // (session id, runtime as the session listing knows it)
    let targets: Vec<(String, Option<String>)> = match session {
        Some(session) => vec![(session.to_string(), None)],
        None => {
            let labeled: HashSet<String> = lake::load_labels(aspect)?
                .into_iter()
                .map(|label| label.session_id)
                .collect();
            let mut unlabeled: Vec<(String, Option<String>)> = lake::all_sessions()?
                .into_iter()
                .filter(|row| !labeled.contains(&row.session_id))
                .map(|row| (row.session_id, row.runtime))
                .collect();
            if let Some(limit) = limit {
                unlabeled.truncate(limit.max(0) as usize);
            }
            unlabeled
        }
    };

    let ids: Vec<String> = targets.iter().map(|(id, _)| id.clone()).collect();
    let texts_by_id = lake::session_texts(&ids)?;

    let mut usable: Vec<(String, Option<String>, String)> = Vec::new();
    for (session_id, runtime) in targets {
        let Some(entry) = texts_by_id.get(&session_id) else {
            continue;
        };
        if entry.text.trim().is_empty() {
            continue;
        }
        let runtime = match entry.runtime.as_deref() {
            Some(value) if !value.is_empty() => Some(value.to_string()),
            _ => runtime,
        };
        usable.push((session_id, runtime, entry.text.clone()));
    }
    if usable.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    let texts: Vec<String> = usable.iter().map(|(_, _, text)| text.clone()).collect();
    let predictions = predict(&artifact, &texts)?;

    let mut suggestions = Vec::with_capacity(usable.len());
    for ((session_id, runtime, _), (value, confidence)) in usable.into_iter().zip(predictions) {
        let mut suggestion = Map::new();
        suggestion.insert("ts".to_string(), json!(crate::util::now_iso()));
        suggestion.insert("session_id".to_string(), json!(session_id));
        suggestion.insert("runtime".to_string(), json!(runtime));
        suggestion.insert("aspect".to_string(), json!(aspect));
        suggestion.insert("value".to_string(), json!(value));
        suggestion.insert(
            "note".to_string(),
            json!(format!("confidence={confidence:.2}")),
        );
        suggestion.insert("source".to_string(), json!("model"));
        suggestions.push(Value::Object(suggestion));
    }
    // The first real result this product produces for an operator: a trained
    // classifier's labels over real session text. Recorded here, where the
    // suggestions exist, and nowhere else.
    crate::onboarding::record_first_success(aspect, suggestions.len());
    Ok(Value::Array(suggestions))
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

/// One entry per trained aspect: every artifact, newest marked active.
pub fn info() -> Result<Vec<Value>> {
    let mut entries = Vec::new();
    let root = models_dir();
    if !root.is_dir() {
        return Ok(entries);
    }
    // Sorted by file name, because `info` prints these in order.
    let mut names: Vec<String> = std::fs::read_dir(&root)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
        .collect();
    names.sort_unstable();

    for name in names {
        // Nothing this trainer wrote can be named otherwise, and the name is
        // about to be joined onto a path.
        if !valid_name(&name) {
            continue;
        }
        let mut entry = Map::new();
        entry.insert("aspect".to_string(), json!(name));
        entry.insert(
            "dir".to_string(),
            json!(root.join(&name).display().to_string()),
        );
        let found = artifacts(&name)?;
        match found.last() {
            None => {
                entry.insert("artifacts".to_string(), Value::Array(Vec::new()));
            }
            Some(newest) => {
                entry.insert("active".to_string(), json!(newest.backend));
                entry.insert(
                    "artifacts".to_string(),
                    Value::Array(
                        found
                            .iter()
                            .map(|artifact| {
                                let mut object = Map::new();
                                object.insert("backend".to_string(), json!(artifact.backend));
                                object.insert(
                                    "dir".to_string(),
                                    json!(artifact.dir.display().to_string()),
                                );
                                object.insert("metrics".to_string(), artifact.metrics.clone());
                                Value::Object(object)
                            })
                            .collect(),
                    ),
                );
            }
        }
        entries.push(Value::Object(entry));
    }
    Ok(entries)
}
