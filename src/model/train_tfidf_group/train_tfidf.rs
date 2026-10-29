use super::*;

// ---------------------------------------------------------------------------
// tfidf-logreg backend
// ---------------------------------------------------------------------------

pub(crate) fn train_tfidf(
    plan: &Plan,
    settings: &jobs::TfidfTraining,
) -> Result<Value, TrainFailure> {
    let (texts, values) = side(plan, &plan.split.train_index);
    let counts = class_counts(values.iter().map(String::as_str));

    // Stratified folds need every class in every fold, so the smallest class
    // decides how many folds there can be; a class of one member cannot be
    // both trained on and tested, and then no cross-validated accuracy exists.
    let folds = counts
        .values()
        .copied()
        .min()
        .filter(|members| *members >= 2); // stratified k-fold needs at least two splits: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    let cv_accuracy = folds
        .and_then(|folds| cross_val_accuracy(&texts, &values, folds, settings))
        .map(evaluate::round4);

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
    // An accuracy that does not exist is null with the reason beside it, never
    // a number: a class too small to fold, or folds none of which could be
    // scored (every training side held one class).
    match (cv_accuracy, folds) {
        (Some(value), _) => {
            metrics.insert("cv_accuracy".to_string(), json!(value));
        }
        (None, None) => {
            metrics.insert("cv_accuracy".to_string(), Value::Null);
            metrics.insert(
                "cv_unavailable".to_string(),
                json!("a class has fewer than two training sessions"),
            );
        }
        (None, Some(folds)) => {
            metrics.insert("cv_accuracy".to_string(), Value::Null);
            metrics.insert(
                "cv_unavailable".to_string(),
                json!(format!(
                    "none of the {folds} folds could be scored: each training side held a single class or its test side was empty"
                )),
            );
        }
    }
    metrics.insert(
        "cv_folds".to_string(),
        match folds {
            Some(folds) => json!(folds),
            None => Value::Null,
        },
    );
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
    crate::util::write_whole(&model_path, (serialized + "\n").as_bytes())?;
    let metrics = Value::Object(metrics);
    crate::util::write_json_pretty(&out_dir.join(METRICS_FILE), &metrics)?;
    Ok(metrics)
}

// ---------------------------------------------------------------------------
// HuggingFace backend (optional 'hf' feature)
// ---------------------------------------------------------------------------

/// The Python build raised this when the optional `hf` extra was not
/// installed. The condition survives the rewrite; the remedy changes, because
/// it is a compile-time feature now rather than a pip extra.
#[cfg(not(feature = "hf"))]
pub(crate) const HF_FEATURE_MISSING: &str =
    "fine-tuning with --model requires the optional 'hf' feature; \
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
    // `model_path`; the path is the directory the backend trained into.
    for (key, value) in trained.metrics {
        if key != "model_path" {
            metrics.insert(key, value);
        }
    }
    metrics.insert("eval_split".to_string(), plan.split.frozen.clone());
    metrics.insert(
        "model_path".to_string(),
        json!(trained.dir.display().to_string()),
    );

    let (holdout_texts, holdout_values) = side(plan, &plan.split.holdout_index);
    if !holdout_texts.is_empty() {
        let predictions = crate::hf::predict(
            &trained.dir,
            &holdout_texts,
            max_length,
            training.batch_size,
            training.seed,
        )?;
        metrics.insert(
            "holdout_evaluation".to_string(),
            evaluate::holdout_report(&holdout_values, &predictions),
        );
    }
    if let Some(job) = &plan.job {
        metrics.insert("job".to_string(), job.clone());
    }

    let metrics = Value::Object(metrics);
    crate::util::write_json_pretty(&trained.dir.join(METRICS_FILE), &metrics)?;
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
/// from).
pub fn train(
    aspect: &str,
    model_id: Option<&str>,
    training: &jobs::Training,
    eval_split: &Value,
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
        None,
    )?;
    match (model_id, training) {
        (None, jobs::Training::Tfidf(settings)) => train_tfidf(&plan, settings),
        (Some(model_id), jobs::Training::Hf(settings)) => train_hf(&plan, model_id, settings),
        (None, jobs::Training::Hf(_)) => Err("fine-tuning settings need --model".into()),
        (Some(model_id), jobs::Training::Tfidf(_)) => Err(format!(
            "{model_id} is fine-tuned with HuggingFace settings, not tfidf-logreg settings"
        )
        .into()),
    }
}
