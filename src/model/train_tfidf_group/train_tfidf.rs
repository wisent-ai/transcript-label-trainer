use super::*;

// ---------------------------------------------------------------------------
// tfidf-logreg backend
// ---------------------------------------------------------------------------

pub(crate) fn train_tfidf(plan: &Plan, settings: &jobs::TfidfTraining) -> Result<Value, TrainFailure> {
    let (texts, values) = side(plan, &plan.split.train_index);
    let counts = class_counts(values.iter().map(String::as_str));

    // Stratified folds need every class in every fold, so the smallest class
    // decides how many folds there can be; a class of one member cannot be
    // both trained on and tested, and then no cross-validated accuracy exists.
    let folds = counts.values().copied().min().filter(|members| *members > members.signum());
    let cv_accuracy = folds.map(|folds| round4(cross_val_accuracy(&texts, &values, folds, settings)));
    let cv_folds = folds.unwrap_or_default();

    let model = TfidfModel::fit(&texts, &values, settings);

    let out_dir = aspect_dir(&plan.out_name)?;
    std::fs::create_dir_all(&out_dir).map_err(Error::from)?;
    let model_path = out_dir.join(MODEL_FILE);

    let mut metrics = base_metrics(
        &plan.aspect,
        TFIDF_BACKEND,
        &tfidf_model_desc(settings),
        texts.len(),
        &counts,
    );
    metrics.insert(
        "cv_accuracy".to_string(),
        match cv_accuracy {
            Some(value) => json!(value),
            None => Value::Null,
        },
    );
    metrics.insert("cv_folds".to_string(), json!(cv_folds));
    metrics.insert("hyperparameters".to_string(), json!(settings));
    metrics.insert("eval_split".to_string(), plan.split.frozen.clone());
    metrics.insert(
        "model_path".to_string(),
        json!(model_path.display().to_string()),
    );

    let (holdout_texts, holdout_values) = side(plan, &plan.split.holdout_index);
    if !holdout_texts.is_empty() {
        let predictions = model.predict(&holdout_texts);
        metrics.insert(
            "holdout_evaluation".to_string(),
            evaluate::holdout_report(&holdout_values, &predictions),
        );
    }
    if let Some(job) = &plan.job {
        metrics.insert("job".to_string(), job.clone());
    }

    // The artifact first, then the metrics that describe it — the order the
    // Python build wrote them in, so an interrupted run leaves a model without
    // metrics rather than metrics pointing at a model that was never written.
    let serialized = serde_json::to_string(&model.to_file(settings)).map_err(Error::from)?;
    std::fs::write(&model_path, serialized + "\n").map_err(Error::from)?;
    let metrics = Value::Object(metrics);
    write_pretty(&out_dir.join(METRICS_FILE), &metrics)?;
    Ok(metrics)
}

// ---------------------------------------------------------------------------
// HuggingFace backend (optional 'hf' feature)
// ---------------------------------------------------------------------------

/// The Python build raised this when the optional `hf` extra was not
/// installed. The condition survives the rewrite; the remedy changes, because
/// it is a compile-time feature now rather than a pip extra.
#[cfg(not(feature = "hf"))]
pub(crate) const HF_FEATURE_MISSING: &str = "fine-tuning with --model requires the optional 'hf' feature; \
     rebuild with: cargo build --release --features hf";

#[cfg(feature = "hf")]
pub(crate) fn train_hf(
    plan: &Plan,
    model_id: &str,
    training: &jobs::HfTraining,
) -> Result<Value, TrainFailure> {
    let (texts, values) = side(plan, &plan.split.train_index);
    let counts = class_counts(values.iter().map(String::as_str));

    let out_dir = aspect_dir(&plan.out_name)?;
    std::fs::create_dir_all(&out_dir).map_err(Error::from)?;

    let max_length = training.max_length;
    let config = crate::hf::TrainConfig {
        aspect: &plan.aspect,
        model_id,
        epochs: training.epochs,
        batch_size: training.batch_size,
        lr: training.learning_rate,
        max_length,
        seed: training.seed,
        weight_decay: training.weight_decay,
        max_grad_norm: training.max_grad_norm,
        in_training_eval_share: training.in_training_eval_share,
    };
    let trained = crate::hf::train(&out_dir, &texts, &values, &config)?;

    let mut metrics = base_metrics(
        &plan.aspect,
        "hf",
        &format!("fine-tuned {model_id} (sequence classification)"),
        texts.len(),
        &counts,
    );
    // The backend's own fragment, with `eval_split` slotted in ahead of
    // `model_path` so the key order matches what the Python build wrote.
    let mut backend_model_path = None;
    for (key, value) in trained.metrics {
        if key == "model_path" {
            backend_model_path = Some(value);
        } else {
            metrics.insert(key, value);
        }
    }
    metrics.insert("eval_split".to_string(), plan.split.frozen.clone());
    metrics.insert(
        "model_path".to_string(),
        backend_model_path.unwrap_or_else(|| json!(trained.dir.display().to_string())),
    );

    let (holdout_texts, holdout_values) = side(plan, &plan.split.holdout_index);
    if !holdout_texts.is_empty() {
        let predictions =
            crate::hf::predict(&trained.dir, &holdout_texts, max_length, training.batch_size, training.seed)?;
        metrics.insert(
            "holdout_evaluation".to_string(),
            evaluate::holdout_report(&holdout_values, &predictions),
        );
    }
    if let Some(job) = &plan.job {
        metrics.insert("job".to_string(), job.clone());
    }

    let metrics = Value::Object(metrics);
    write_pretty(&trained.dir.join(METRICS_FILE), &metrics)?;
    Ok(metrics)
}

#[cfg(not(feature = "hf"))]
pub(crate) fn train_hf(
    _plan: &Plan,
    _model_id: &str,
    _training: &jobs::HfTraining,
) -> Result<Value, TrainFailure> {
    Err(TrainFailure::Failed(Error(HF_FEATURE_MISSING.to_string())))
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

/// Train one aspect from the command line with the backend settings the
/// caller stated (`model_id` names the HuggingFace model a fine-tune starts
/// from); `min_sessions` is the caller's floor of labeled sessions on the
/// training side.
pub fn train(
    aspect: &str,
    model_id: Option<&str>,
    training: &jobs::Training,
    eval_split: &Value,
    min_sessions: usize,
) -> Result<Value, TrainFailure> {
    let eval_split: jobs::EvalSplit =
        serde_json::from_value(eval_split.clone()).map_err(Error::from)?;
    let labels = lake::load_labels(aspect)?;
    let plan = build_plan(
        aspect,
        &labels,
        &format!("aspect '{aspect}'"),
        aspect,
        eval_split,
        None,
        min_sessions,
        None,
    )?;
    match (model_id, training) {
        (None, jobs::Training::Tfidf(settings)) => train_tfidf(&plan, settings),
        (Some(model_id), jobs::Training::Hf(settings)) => train_hf(&plan, model_id, settings),
        (None, jobs::Training::Hf(_)) => Err("fine-tuning settings need --model".into()),
        (Some(model_id), jobs::Training::Tfidf(_)) => {
            Err(format!("{model_id} is fine-tuned with HuggingFace settings, not tfidf-logreg settings").into())
        }
    }
}

// ---------------------------------------------------------------------------
// Declarative training jobs (run command)
// ---------------------------------------------------------------------------

/// A label record's timestamp as a comparable instant. An empty or
/// unparseable `ts` sorts before every real one, which is what excludes it
/// from a `since` window — the Python original did the same with
/// `datetime.min`.
pub(crate) fn label_instant(ts: &str) -> (i64, u32) {
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(ts) {
        return (parsed.timestamp(), parsed.timestamp_subsec_nanos());
    }
    // A timestamp without a zone is read as UTC rather than crashing the run.
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(ts, format) {
            let at = parsed.and_utc();
            return (at.timestamp(), at.timestamp_subsec_nanos());
        }
    }
    if let Ok(parsed) = chrono::NaiveDate::parse_from_str(ts, "%Y-%m-%d") {
        if let Some(at) = parsed.and_hms_opt(0, 0, 0) {
            let at = at.and_utc();
            return (at.timestamp(), at.timestamp_subsec_nanos());
        }
    }
    (i64::MIN, 0)
}

/// Label records matching an evaluator and scope filters.
pub fn select_labels(evaluator: &str, scope: &jobs::Scope) -> Result<Vec<lake::SessionLabel>> {
    let labels = lake::load_labels(&scope.aspect)?;
    let since = scope.since.as_deref().map(label_instant);
    Ok(labels
        .into_iter()
        .filter(|record| record.source.trim() == evaluator)
        .filter(|record| match &scope.runtimes {
            Some(allowed) => record
                .runtime
                .as_deref()
                .is_some_and(|runtime| allowed.iter().any(|value| value == runtime)),
            None => true,
        })
        .filter(|record| match since {
            Some(since) => label_instant(&record.ts) >= since,
            None => true,
        })
        .filter(|record| match &scope.values {
            Some(allowed) => allowed.iter().any(|value| value == &record.value),
            None => true,
        })
        .collect())
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
/// by field here, and a field the artifact does not carry means "unset",
/// exactly as an absent key did in the Python spec.
pub(crate) fn scope_from_json(scope: &Value) -> Result<jobs::Scope> {
    let string_list = |key: &str| -> Option<Vec<String>> {
        let items: Vec<String> = scope
            .get(key)?
            .as_array()?
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect();
        (!items.is_empty()).then_some(items)
    };
    let aspect = scope
        .get("aspect")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("artifact metrics carry a 'scope' without an 'aspect'".to_string()))?;
    Ok(jobs::Scope {
        aspect: aspect.to_string(),
        runtimes: string_list("runtimes"),
        since: scope
            .get("since")
            .and_then(Value::as_str)
            .map(str::to_string),
        values: string_list("values"),
        min_text_chars: scope.get("min_text_chars").and_then(Value::as_u64),
    })
}
