use super::*;

pub(crate) fn count(value: Option<i64>, fallback: usize) -> usize {
    match value {
        Some(value) if value >= 0 => value as usize,
        Some(_) => 0,
        None => fallback,
    }
}

// ---------------------------------------------------------------- printers

pub(crate) static NULL: Value = Value::Null;

pub(crate) fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&NULL)
}

pub(crate) fn text(value: &Value) -> String {
    json_text(value)
}

pub(crate) fn print_info(entries: &[Value]) {
    if entries.is_empty() {
        outln!("no trained aspects under {}", model::models_dir().display());
        return;
    }
    for entry in entries {
        let artifacts: &[Value] = match field(entry, "artifacts") {
            Value::Array(artifacts) => artifacts.as_slice(),
            _ => &[],
        };
        if artifacts.is_empty() {
            outln!(
                "{}: no trained artifacts in {}",
                text(field(entry, "aspect")),
                text(field(entry, "dir"))
            );
            continue;
        }
        let active = text(field(entry, "active"));
        outln!(
            "{} (active backend: {active}):",
            text(field(entry, "aspect"))
        );
        for artifact in artifacts {
            let metrics = field(artifact, "metrics");
            let backend = text(field(artifact, "backend"));
            let marker = if backend == active { "*" } else { " " };
            let job = field(metrics, "job");
            if json_truthy(job) {
                outln!(
                    "    job:     {} — {} (evaluator: {})",
                    text(field(job, "name")),
                    text(field(job, "task")),
                    text(field(job, "evaluator"))
                );
            }
            let quality = if backend == "sklearn" {
                let cv = field(metrics, "cv_accuracy");
                if cv.is_null() {
                    "cv_accuracy=n/a".to_string()
                } else {
                    format!(
                        "cv_accuracy={} ({}-fold)",
                        text(cv),
                        text(field(metrics, "cv_folds"))
                    )
                }
            } else {
                let hyperparameters = field(metrics, "hyperparameters");
                let accuracy = field(field(metrics, "in_training_eval"), "accuracy");
                let mut quality = if accuracy.is_null() {
                    "in_training_accuracy=n/a".to_string()
                } else {
                    format!("in_training_accuracy={}", text(accuracy))
                };
                let _ = write!(
                    quality,
                    " ({}, epochs={}, lr={}, device={})",
                    text(field(metrics, "base_model")),
                    text(field(hyperparameters, "epochs")),
                    text(field(hyperparameters, "lr")),
                    text(field(metrics, "device"))
                );
                quality
            };
            outln!(
                " {marker} {backend}: {} sessions trained on, classes={}, {quality}\n\
             \x20   model:   {}\n\
             \x20   trained: {}",
                text(field(metrics, "n_sessions")),
                repr(field(metrics, "classes")),
                text(field(metrics, "model_path")),
                text(field(metrics, "trained_at"))
            );
            let holdout = field(metrics, "holdout_evaluation");
            let split = field(metrics, "eval_split");
            if json_truthy(holdout) {
                outln!(
                    "    frozen holdout: accuracy={} on {} session(s), seed={}, fraction={}",
                    text(field(holdout, "accuracy")),
                    text(field(holdout, "n_sessions")),
                    text(field(split, "seed")),
                    text(field(split, "fraction"))
                );
            } else if json_truthy(split) && !split.get("enabled").map(json_truthy).unwrap_or(true) {
                outln!("    frozen holdout: disabled by eval_split: false");
            }
        }
    }
}

/// Where Stado puts this run — and, when it could not, why it is local.
pub(crate) fn print_placement() {
    let resolved = placement::resolve_placement();
    outln!("placement:");
    outln!("    source:        {}", resolved.source);
    outln!(
        "    training host: {}",
        resolved
            .training_host
            .as_deref()
            .filter(|host| !host.is_empty())
            .unwrap_or("undeclared")
    );
    outln!("    training root: {}", resolved.training_root.display());
    outln!("    storage root:  {}", resolved.storage_root.display());
    if resolved.source == "local-fallback" {
        outln!("    fallback:      {}", resolved.detail);
    }
    outln!();
}

/// The evaluate report: frozen-holdout scores, then the teacher's verdict.
pub(crate) fn print_verdict(verdict: &Value) {
    let split = field(verdict, "eval_split");
    let holdout = field(verdict, "holdout_evaluation");
    outln!(
        "{} (aspect: {}, backend: {}):",
        text(field(verdict, "name")),
        text(field(verdict, "aspect")),
        text(field(verdict, "backend"))
    );
    outln!(
        "    frozen split:  {} session(s), fraction={}, seed={}, created {}\n\
     \x20   split file:    {}",
        text(field(split, "frozen_sessions")),
        text(field(split, "fraction")),
        text(field(split, "seed")),
        text(field(split, "created_at")),
        text(field(split, "path"))
    );
    if json_truthy(field(split, "missing_ground_truth")) {
        outln!(
            "    unlabeled now: {} (excluded)",
            text(field(split, "missing_ground_truth"))
        );
    }
    if json_truthy(field(split, "skipped_no_text")) {
        outln!(
            "    without text:  {} (excluded)",
            text(field(split, "skipped_no_text"))
        );
    }
    outln!(
        "    holdout:       accuracy={} on {} session(s)",
        text(field(holdout, "accuracy")),
        text(field(holdout, "n_sessions"))
    );
    let correct = field(holdout, "correct");
    if let Value::Object(counts) = field(holdout, "counts") {
        let mut values: Vec<&String> = counts.keys().collect();
        values.sort_unstable();
        for value in values {
            let scored = correct.get(value.as_str()).unwrap_or(&Value::Null);
            let scored = if scored.is_null() {
                "0".to_string()
            } else {
                text(scored)
            };
            outln!(
                "        {value}: {scored}/{} correct",
                text(counts.get(value).unwrap_or(&NULL))
            );
        }
    }
    if let Value::Array(pairs) = field(holdout, "confusion") {
        for pair in pairs {
            outln!(
                "        confused {} -> {} ({}x)",
                text(field(pair, "gold")),
                text(field(pair, "predicted")),
                text(field(pair, "n"))
            );
        }
    }
    let judge = field(verdict, "judge");
    if !json_truthy(field(judge, "enabled")) {
        outln!("    judge:         skipped (--no-judge)");
        return;
    }
    outln!(
        "    judge:         {} calls {}/{} prediction(s) acceptable \
     (agreement_rate={}, failed={})",
        text(field(judge, "model")),
        text(field(judge, "acceptable")),
        text(field(judge, "judged")),
        text(field(judge, "agreement_rate")),
        text(field(judge, "failed"))
    );
    let best = field(verdict, "best_review");
    if json_truthy(field(best, "enabled")) {
        outln!(
            "    best review:   {} reviewed={}, labels nonsensical={}, \
             judge opinions nonsensical={}, failed={}, sensible={}",
            text(field(best, "model")),
            text(field(best, "reviewed")),
            text(field(best, "label_nonsensical")),
            text(field(best, "judge_nonsensical")),
            text(field(best, "failed")),
            text(field(best, "sensible"))
        );
    }
    if let Value::Array(records) = field(verdict, "sessions") {
        for record in records {
            let mark = if text(field(record, "verdict")) == evaluate::JUDGE_VALUES[0] {
                "ok "
            } else {
                "bad"
            };
            outln!(
                "        {mark} {}: gold={} predicted={} ({})",
                text(field(record, "session_id")),
                text(field(record, "gold")),
                text(field(record, "prediction")),
                text(field(record, "confidence"))
            );
            if json_truthy(field(record, "best_review")) {
                outln!(
                    "             final review={}",
                    text(field(record, "best_review"))
                );
            }
        }
    }
    if let Value::Array(failures) = field(verdict, "failures") {
        for failure in failures {
            outln!(
                "        err {}: {}",
                text(field(failure, "session_id")),
                text(field(failure, "error"))
            );
        }
    }
    outln!("    verdict file:  {}", text(field(verdict, "judge_path")));
}

// -------------------------------------------------------------- json output

/// `json.dumps(value, indent=2)`, including `ensure_ascii`.
///
/// Rust would print `ą` and `2e-5` where Python printed `\u0105` and `2e-05`,
/// and these bytes land in files the lake and the operator already have.
pub(crate) fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value, 0);
    out
}
