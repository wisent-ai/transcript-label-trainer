use super::*;

/// Resolve a validated job spec to its training data, without training.
pub fn resolve_job(job: &jobs::Job) -> Result<Resolved> {
    let labels = select_labels(job)?;
    let counts = class_counts(labels.iter().map(|label| label.value.as_str()));
    Ok(Resolved { labels, counts })
}

/// The resolved-summary printed before training.
pub fn job_summary(job: &jobs::Job, resolved: &Resolved) -> Value {
    let mut summary = Map::new();
    summary.insert("name".to_string(), json!(job.name));
    summary.insert("task".to_string(), json!(job.task));
    summary.insert("evaluator".to_string(), json!(job.evaluator));
    summary.insert("model".to_string(), json!(job.model));
    summary.insert("scope".to_string(), job_scope_json(&job.scope));
    summary.insert("sessions_found".to_string(), json!(resolved.labels.len()));
    summary.insert("counts".to_string(), counts_json(&resolved.counts));
    Value::Object(summary)
}

/// Everything a job needs to train, including its frozen evaluation split.
///
/// Separate from [`run_job`] so `run` can print the resolved split — which
/// sessions train and which are held out — before training starts.
pub fn prepare_job(job: &jobs::Job, resolved: &Resolved) -> Result<Plan, TrainFailure> {
    let subject = format!(
        "job '{}' (evaluator '{}', aspect '{}')",
        job.name, job.evaluator, job.scope.aspect
    );
    let mut job_meta = Map::new();
    job_meta.insert("name".to_string(), json!(job.name));
    job_meta.insert("task".to_string(), json!(job.task));
    job_meta.insert("evaluator".to_string(), json!(job.evaluator));
    job_meta.insert("scope".to_string(), job_scope_json(&job.scope));
    job_meta.insert(
        "eval_split".to_string(),
        serde_json::to_value(&job.eval_split).map_err(Error::from)?,
    );
    job_meta.insert(
        "judge".to_string(),
        serde_json::to_value(&job.judge).map_err(Error::from)?,
    );

    build_plan(
        &job.scope.aspect,
        &resolved.labels,
        &subject,
        &job.name,
        job.eval_split.clone(),
        job.scope.min_text_chars,
        Some(Value::Object(job_meta)),
    )
}

/// Train from a prepared job and persist spec copy + job metadata.
pub fn run_job(job: &jobs::Job, plan: &Plan) -> Result<Value, TrainFailure> {
    let metrics = if job.model == jobs::SKLEARN_MODEL {
        train_tfidf(plan)?
    } else {
        train_hf(plan, &job.model, 3.0, 8, 2e-5, 512)?
    };
    let mut spec_copy = match plan.job.clone() {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    spec_copy.insert("model".to_string(), json!(job.model));
    let yaml = serde_yaml::to_string(&Value::Object(spec_copy)).map_err(Error::from)?;
    std::fs::write(aspect_dir(&job.name)?.join("job.yaml"), yaml).map_err(Error::from)?;
    Ok(metrics)
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// One trained artifact: which backend produced it, where it lives, and the
/// metrics it was written with.
pub struct Artifact {
    pub backend: String,
    pub dir: PathBuf,
    pub metrics: Value,
}

/// All trained artifacts for one aspect, oldest first.
pub(crate) fn artifacts(name: &str) -> Result<Vec<Artifact>> {
    let base = aspect_dir(name)?;
    let mut found: Vec<Artifact> = Vec::new();
    let metrics_path = base.join(METRICS_FILE);
    // A Python-trained artifact carries model.joblib instead of model.json. It
    // is still a real training run with real metrics, so `info` keeps
    // reporting it; only loading it for inference fails, and it says why.
    let has_model = base.join(MODEL_FILE).is_file() || base.join(LEGACY_MODEL_FILE).is_file();
    if has_model && metrics_path.is_file() {
        found.push(Artifact {
            backend: TFIDF_BACKEND.to_string(),
            dir: base.clone(),
            metrics: read_metrics(&metrics_path)?,
        });
    }
    if base.is_dir() {
        // Sorted by file name: std::fs::read_dir yields raw directory order,
        // where Python's glob() was sorted, and this order is observable in
        // `info`.
        let mut subdirectories: Vec<(String, PathBuf)> = std::fs::read_dir(&base)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name().to_str()?.to_string();
                let path = entry.path();
                (name.starts_with("hf-") && path.is_dir()).then_some((name, path))
            })
            .collect();
        subdirectories.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (_, sub) in subdirectories {
            let metrics_path = sub.join(METRICS_FILE);
            if metrics_path.is_file() {
                found.push(Artifact {
                    backend: "hf".to_string(),
                    dir: sub,
                    metrics: read_metrics(&metrics_path)?,
                });
            }
        }
    }
    found.sort_by(|a, b| trained_at(a).cmp(trained_at(b)));
    Ok(found)
}

pub(crate) fn trained_at(artifact: &Artifact) -> &str {
    artifact
        .metrics
        .get("trained_at")
        .and_then(Value::as_str)
        .unwrap_or("")
}

pub(crate) fn read_metrics(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|error| Error(format!("{}: {error}", path.display())))
}

/// The newest artifact trained under one aspect or job name.
pub fn active_artifact(name: &str) -> Result<Artifact> {
    let mut found = artifacts(name)?;
    match found.pop() {
        Some(artifact) => Ok(artifact),
        None => Err(Error(format!(
            "no trained model under {}; train it first with \
             'transcript-label-trainer train --aspect {name}' or \
             'transcript-label-trainer run <job.yaml>'",
            aspect_dir(name)?.display()
        ))),
    }
}

/// The ground-truth label records an artifact was trained against.
///
/// A job artifact carries its evaluator and scope, so the same selection is
/// reproduced; a bare `train` artifact was fitted on every label for its
/// aspect.
pub fn labels_for_artifact(metrics: &Value) -> Result<Vec<lake::SessionLabel>> {
    let Some(job_meta) = metrics.get("job").filter(|value| !value.is_null()) else {
        let aspect = metrics
            .get("aspect")
            .and_then(Value::as_str)
            .ok_or_else(|| Error("artifact metrics carry no 'aspect'".to_string()))?;
        return lake::load_labels(aspect);
    };
    let scope_value = job_meta
        .get("scope")
        .ok_or_else(|| Error("artifact metrics carry a 'job' without a 'scope'".to_string()))?;
    let scope = scope_from_json(scope_value)?;
    let mut job = synthetic_job(
        job_meta
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("artifact"),
        &scope.aspect,
        jobs::EvalSplit {
            enabled: false,
            fraction: None,
            seed: None,
        },
    );
    job.evaluator = job_meta
        .get("evaluator")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    job.scope = scope;
    select_labels(&job)
}

pub(crate) fn infer_tfidf(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    let path = artifact.dir.join(MODEL_FILE);
    if !path.is_file() {
        if artifact.dir.join(LEGACY_MODEL_FILE).is_file() {
            let name = artifact
                .dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<name>");
            crate::bail!(
                "the artifact in {} holds {LEGACY_MODEL_FILE}, a pickle written by the \
                 Python build that this binary cannot read; retrain it with \
                 'transcript-label-trainer train --aspect {name}' or \
                 'transcript-label-trainer run <job.yaml>' to produce {MODEL_FILE}",
                artifact.dir.display()
            );
        }
        crate::bail!("no {MODEL_FILE} in {}", artifact.dir.display());
    }
    let file: ModelFile = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .map_err(|error| Error(format!("{}: {error}", path.display())))?;
    Ok(TfidfModel::from_file(file)?.predict(texts))
}

#[cfg(feature = "hf")]
pub(crate) fn infer_hf(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    let max_length = artifact
        .metrics
        .get("hyperparameters")
        .and_then(|hyperparameters| hyperparameters.get("max_length"))
        .and_then(Value::as_u64)
        .unwrap_or(512) as usize;
    crate::hf::predict(&artifact.dir, texts, max_length)
}

#[cfg(not(feature = "hf"))]
pub(crate) fn infer_hf(artifact: &Artifact, _texts: &[String]) -> Result<Vec<(String, f64)>> {
    Err(Error(format!(
        "the artifact in {} is a fine-tuned HuggingFace model, which this build cannot \
         load; rebuild with: cargo build --release --features hf",
        artifact.dir.display()
    )))
}

/// Whether this build can load an artifact of that backend at all: the
/// HuggingFace backend is only compiled in behind the optional `hf` feature.
pub fn loadable(backend: &str) -> bool {
    backend == TFIDF_BACKEND || cfg!(feature = "hf")
}

/// (value, confidence) per text, from whichever backend the artifact is.
pub fn predict(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    if artifact.backend == TFIDF_BACKEND {
        infer_tfidf(artifact, texts)
    } else {
        infer_hf(artifact, texts)
    }
}
