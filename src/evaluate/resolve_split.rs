use super::*;

/// Load or create the frozen holdout for this artifact directory.
///
/// Returns which rows of `sessions` train, which are held out, and the
/// provenance of the file. The file is written only on the first run, and only
/// after the training side is known to clear the minimums, so a run that fails
/// cannot leave a split behind that a later run would inherit.
pub fn resolve_split(
    job: &jobs::Job,
    sessions: &[lake::SessionLabel],
    subject: &str,
) -> Result<Split, TrainFailure> {
    let total = sessions.len();
    let session_ids: Vec<String> = sessions
        .iter()
        .map(|session| session.session_id.clone())
        .collect();
    let values: Vec<String> = sessions
        .iter()
        .map(|session| session.value.clone())
        .collect();

    if !job.eval_split.enabled {
        let mut split = Split {
            enabled: false,
            fraction: None,
            seed: None,
            created_at: None,
            path: None,
            holdout_index: Vec::new(),
            train_index: (0..total).collect(),
            missing_from_selection: 0,
            reused: false,
            frozen: Value::Null,
        };
        split.frozen = split_json(&split);
        return Ok(split);
    }

    let path = split_path(&job.name)?;
    let (holdout_ids, fraction, seed, created_at, reused) = if path.is_file() {
        let frozen = read_split(&job.name)?;
        if frozen.fraction != job.eval_split.fraction || frozen.seed != job.eval_split.seed {
            lake::warn(&format!(
                "eval_split in the spec (fraction={}, seed={}) differs from the \
                 frozen split in {} (fraction={}, seed={}); the frozen file wins \
                 — that is what frozen means",
                opt_float(job.eval_split.fraction),
                opt_int(job.eval_split.seed),
                path.display(),
                opt_float(frozen.fraction),
                opt_int(frozen.seed),
            ));
        }
        (
            frozen.session_ids,
            frozen.fraction,
            frozen.seed,
            frozen.created_at,
            true,
        )
    } else {
        let fraction = job
            .eval_split
            .fraction
            .unwrap_or(jobs::DEFAULT_EVAL_FRACTION);
        let seed = job.eval_split.seed.unwrap_or(jobs::DEFAULT_EVAL_SEED);
        let chosen = choose_holdout(&session_ids, &values, fraction, seed);
        (chosen, Some(fraction), Some(seed), Some(now_iso()), false)
    };

    let holdout: BTreeSet<&str> = holdout_ids.iter().map(String::as_str).collect();
    let mut holdout_index: Vec<usize> = Vec::new();
    let mut train_index: Vec<usize> = Vec::new();
    for (index, session_id) in session_ids.iter().enumerate() {
        if holdout.contains(session_id.as_str()) {
            holdout_index.push(index);
        } else {
            train_index.push(index);
        }
    }

    let train_values: BTreeSet<&str> = train_index
        .iter()
        .map(|index| values[*index].as_str())
        .collect();
    if train_index.len() < model::MIN_LABELED_SESSIONS || train_values.len() < 2 {
        return Err(TrainFailure::NotEnoughData(format!(
            "{subject} has {total} usable labeled session(s), of which {} are held \
             out by the frozen evaluation split (fraction={}, seed={}), leaving {} \
             session(s) across {} distinct value(s) to train on; at least {} \
             sessions and 2 distinct values are required. Add labels with \
             'transcript-lake label add', or set 'eval_split: false' in the job \
             spec to train on every labeled session.",
            holdout_index.len(),
            opt_float(fraction),
            opt_int(seed),
            train_index.len(),
            train_values.len(),
            model::MIN_LABELED_SESSIONS,
        )));
    }

    if !reused {
        let directory = model::aspect_dir(&job.name)?;
        std::fs::create_dir_all(&directory).map_err(Error::from)?;
        let mut record = Map::new();
        record.insert(
            "fraction".to_string(),
            fraction.map(number).unwrap_or(Value::Null),
        );
        record.insert(
            "seed".to_string(),
            seed.map(|seed| Value::Number(seed.into()))
                .unwrap_or(Value::Null),
        );
        record.insert(
            "created_at".to_string(),
            created_at.clone().map(Value::String).unwrap_or(Value::Null),
        );
        record.insert(
            "session_ids".to_string(),
            Value::Array(holdout_ids.iter().cloned().map(Value::String).collect()),
        );
        let body = serde_json::to_string_pretty(&Value::Object(record)).map_err(Error::from)?;
        std::fs::write(&path, body + "\n").map_err(Error::from)?;
    }

    let mut split = Split {
        enabled: true,
        fraction,
        seed,
        created_at,
        path: Some(path.to_string_lossy().into_owned()),
        missing_from_selection: holdout_ids.len() - holdout_index.len(),
        holdout_index,
        train_index,
        reused,
        frozen: Value::Null,
    };
    split.frozen = split_json(&split);
    Ok(split)
}

/// Accuracy, per-class counts and confusion pairs on the frozen holdout.
///
/// This is not the HF backend's `in_training_eval`: that one is a stratified
/// slice of the training side, resplit on every run. This one is the frozen
/// set, identical across runs and across backends.
pub fn holdout_report(gold: &[String], predictions: &[(String, f64)]) -> Value {
    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    let mut correct_per_class: BTreeMap<&str, u64> = BTreeMap::new();
    let mut confusion: BTreeMap<(&str, &str), u64> = BTreeMap::new();
    let mut correct = 0u64;
    for (actual, (predicted, _confidence)) in gold.iter().zip(predictions) {
        *counts.entry(actual.as_str()).or_insert(0) += 1;
        correct_per_class.entry(actual.as_str()).or_insert(0);
        if predicted == actual {
            *correct_per_class.entry(actual.as_str()).or_insert(0) += 1;
            correct += 1;
        } else {
            *confusion
                .entry((actual.as_str(), predicted.as_str()))
                .or_insert(0) += 1;
        }
    }
    let total = gold.len();

    let mut ordered: Vec<((&str, &str), u64)> =
        confusion.into_iter().map(|(pair, n)| (pair, n)).collect();
    ordered.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let pairs: Vec<Value> = ordered
        .into_iter()
        .map(|((actual, predicted), n)| {
            let mut pair = Map::new();
            pair.insert("gold".to_string(), Value::String(actual.to_string()));
            pair.insert(
                "predicted".to_string(),
                Value::String(predicted.to_string()),
            );
            pair.insert("n".to_string(), Value::Number(n.into()));
            Value::Object(pair)
        })
        .collect();

    let as_object = |source: BTreeMap<&str, u64>| {
        let mut object = Map::new();
        for (key, value) in source {
            object.insert(key.to_string(), Value::Number(value.into()));
        }
        Value::Object(object)
    };

    let mut report = Map::new();
    report.insert("n_sessions".to_string(), Value::Number(total.into()));
    report.insert(
        "accuracy".to_string(),
        if total > 0 {
            number(round4(correct as f64 / total as f64))
        } else {
            Value::Null
        },
    );
    report.insert("counts".to_string(), as_object(counts));
    report.insert("correct".to_string(), as_object(correct_per_class));
    report.insert("confusion".to_string(), Value::Array(pairs));
    Value::Object(report)
}

// ---------------------------------------------------------------------------
// The Brama teacher's verdict
// ---------------------------------------------------------------------------

/// Ask the teacher whether one prediction is defensible for one session.
pub fn judge_prompt(
    aspect: &str,
    task: Option<&str>,
    gold: &str,
    prediction: &str,
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
                "You audit a small classifier that assigns aspect labels to \
                 coding-agent session transcripts. Answer with exactly one \
                 word, {} or {}, and nothing else.",
                JUDGE_VALUES[0], JUDGE_VALUES[1]
            ),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}{purpose}\n\
                 Ground-truth label recorded by the evaluator: {gold}\n\
                 Label predicted by the classifier: {prediction}\n\n\
                 Is the predicted label a defensible reading of this transcript \
                 on this aspect? Answer {} if it is (including when it differs \
                 from the ground truth but the session genuinely supports it), \
                 {} if it is not.\n\n\
                 Transcript:\n{text}",
                JUDGE_VALUES[0], JUDGE_VALUES[1]
            ),
        },
    ]
}

pub(crate) fn failure(session_id: &str, error: &str) -> Value {
    let mut record = Map::new();
    record.insert(
        "session_id".to_string(),
        Value::String(session_id.to_string()),
    );
    record.insert("error".to_string(), Value::String(error.to_string()));
    Value::Object(record)
}
