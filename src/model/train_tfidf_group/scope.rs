use super::*;

// ---------------------------------------------------------------------------
// Declarative training jobs (run command): which labels a job's scope selects
// ---------------------------------------------------------------------------

/// A label record's timestamp, or a `since` bound, as an instant: RFC 3339
/// with a zone, a date-time without one (read as UTC), or a date (its
/// midnight in UTC). Any other text is refused naming it, since a time that
/// cannot be read cannot be placed inside or outside a window.
pub(crate) fn label_instant(ts: &str) -> Result<chrono::DateTime<chrono::Utc>> {
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(ts) {
        return Ok(parsed.with_timezone(&chrono::Utc));
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(ts, format) {
            return Ok(parsed.and_utc());
        }
    }
    if let Ok(parsed) = chrono::NaiveDate::parse_from_str(ts, "%Y-%m-%d") {
        return Ok(parsed.and_time(chrono::NaiveTime::MIN).and_utc());
    }
    Err(Error(format!(
        "label time {ts:?} is not an RFC 3339 time, a date-time or a date"
    )))
}

/// Label records matching an evaluator and scope filters.
pub fn select_labels(evaluator: &str, scope: &jobs::Scope) -> Result<Vec<lake::SessionLabel>> {
    let labels = lake::load_labels(&scope.aspect)?;
    let since = scope.since.as_deref().map(label_instant).transpose()?;
    let mut selected = Vec::new();
    for record in labels {
        if record.source.trim() != evaluator {
            continue;
        }
        if let Some(allowed) = &scope.runtimes {
            let runs_there = record
                .runtime
                .as_deref()
                .is_some_and(|runtime| allowed.iter().any(|value| value == runtime));
            if !runs_there {
                continue;
            }
        }
        if let Some(since) = since {
            if label_instant(&record.ts)? < since {
                continue;
            }
        }
        if let Some(allowed) = &scope.values {
            if !allowed.iter().any(|value| value == &record.value) {
                continue;
            }
        }
        selected.push(record);
    }
    Ok(selected)
}

pub(crate) fn job_scope_json(scope: &jobs::Scope) -> Value {
    let mut object = Map::new();
    object.insert("aspect".to_string(), json!(scope.aspect));
    object.insert("runtimes".to_string(), json!(scope.runtimes));
    object.insert("since".to_string(), json!(scope.since));
    object.insert("values".to_string(), json!(scope.values));
    object.insert("min_text_chars".to_string(), json!(scope.min_text_chars));
    Value::Object(object)
}

/// The inverse of [`job_scope_json`], for the scope an artifact recorded.
/// `jobs::Scope` is deliberately not a serde type — it exists only as the
/// output of the spec validator — so the artifact's copy is read back field
/// by field here. A field the artifact does not carry, or carries as null
/// or an empty list, means "unset"; a field of another shape is refused.
pub(crate) fn scope_from_json(scope: &Value) -> Result<jobs::Scope> {
    let string_list = |key: &str| -> Result<Option<Vec<String>>> {
        let items = match scope.get(key) {
            None | Some(Value::Null) => return Ok(None),
            Some(Value::Array(items)) => items,
            Some(other) => {
                return Err(Error(format!(
                    "the artifact's scope {key} is not a list: {other}"
                )))
            }
        };
        let texts = items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    Error(format!("the artifact's scope {key} holds {item}, not text"))
                })
            })
            .collect::<Result<Vec<String>>>()?;
        Ok((!texts.is_empty()).then_some(texts))
    };
    let text = |key: &str| -> Result<Option<String>> {
        match scope.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(other) => Err(Error(format!(
                "the artifact's scope {key} is not text: {other}"
            ))),
        }
    };
    let aspect = text("aspect")?
        .ok_or_else(|| Error("artifact metrics carry a 'scope' without an 'aspect'".to_string()))?;
    let min_text_chars = match scope.get("min_text_chars") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_u64().ok_or_else(|| {
            Error(format!(
                "the artifact's scope min_text_chars is not a count: {value}"
            ))
        })?),
    };
    Ok(jobs::Scope {
        aspect,
        runtimes: string_list("runtimes")?,
        since: text("since")?,
        values: string_list("values")?,
        min_text_chars,
    })
}
