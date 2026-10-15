use super::*;

/// What a split is resolved for: the artifact directory's name and the
/// stated holdout.
pub struct SplitRequest<'a> {
    pub name: &'a str,
    pub eval_split: &'a jobs::EvalSplit,
}

/// How many frozen holdout sessions are not among the sessions selected now.
fn missing_from(holdout_ids: &[String], session_ids: &[String]) -> usize {
    holdout_ids
        .iter()
        .filter(|session_id| !session_ids.contains(session_id))
        .count()
}

/// Load or create the frozen holdout for this artifact directory.
///
/// Returns which rows of `sessions` train, which are held out, and the
/// provenance of the file. The file is written only on the first run, and only
/// after the training side is known to clear the minimums, so a run that fails
/// cannot leave a split behind that a later run would inherit.
pub fn resolve_split(
    job: &SplitRequest<'_>,
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
            train_index: session_ids
                .iter()
                .enumerate()
                .map(|(index, _)| index)
                .collect(),
            missing_from_selection: missing_from(&[], &session_ids),
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
        // An enabled split always carries the job's stated fraction and seed
        // (jobs::eval_split refuses a spec without them).
        let (Some(fraction), Some(seed)) = (job.eval_split.fraction, job.eval_split.seed) else {
            return Err("eval_split is enabled but states no fraction and seed".into());
        };
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
    if !model::at_least_two(train_values.iter()) {
        return Err(TrainFailure::NotEnoughData(format!(
            "{subject} has {total} usable labeled session(s), of which {} are held \
             out by the frozen evaluation split (fraction={}, seed={}), leaving {} \
             session(s) across {} distinct value(s) to train on; two distinct \
             values are required. Add labels with 'transcript-lake label add', or \
             set 'eval_split: false' in the job spec to train on every labeled \
             session.",
            holdout_index.len(),
            opt_float(fraction),
            opt_int(seed),
            train_index.len(),
            train_values.len(),
        )));
    }

    if !reused {
        let directory = model::aspect_dir(&job.name)?;
        std::fs::create_dir_all(&directory).map_err(Error::from)?;
        let mut record = Map::new();
        record.insert(
            "fraction".to_string(),
            match fraction {
                Some(fraction) => number(fraction),
                None => Value::Null,
            },
        );
        record.insert("seed".to_string(), serde_json::json!(seed));
        record.insert("created_at".to_string(), serde_json::json!(created_at));
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
        missing_from_selection: missing_from(&holdout_ids, &session_ids),
        holdout_index,
        train_index,
        reused,
        frozen: Value::Null,
    };
    split.frozen = split_json(&split);
    Ok(split)
}

/// How many judged (gold, predicted) pairs `keep` accepts.
fn tally(judged: &[(&str, &str)], keep: impl Fn(&str, &str) -> bool) -> usize {
    judged
        .iter()
        .filter(|(actual, predicted)| keep(*actual, *predicted))
        .count()
}

/// Accuracy, per-class counts and confusion pairs on the frozen holdout.
///
/// This is not the HF backend's `in_training_eval`: that one is a stratified
/// slice of the training side, resplit on every run. This one is the frozen
/// set, identical across runs and across backends.
pub fn holdout_report(gold: &[String], predictions: &[(String, f64)]) -> Value {
    let judged: Vec<(&str, &str)> = gold
        .iter()
        .zip(predictions)
        .map(|(actual, (predicted, _confidence))| (actual.as_str(), predicted.as_str()))
        .collect();
    let classes: BTreeSet<&str> = judged.iter().map(|(actual, _)| *actual).collect();
    let mut counts = Map::new();
    let mut correct_per_class = Map::new();
    for class in &classes {
        counts.insert(
            class.to_string(),
            serde_json::json!(tally(&judged, |actual, _| actual == *class)),
        );
        correct_per_class.insert(
            class.to_string(),
            serde_json::json!(tally(&judged, |actual, predicted| actual == *class
                && predicted == *class)),
        );
    }
    let correct = tally(&judged, |actual, predicted| actual == predicted);
    let total = gold.len();

    let confused: BTreeSet<(&str, &str)> = judged
        .iter()
        .copied()
        .filter(|(actual, predicted)| actual != predicted)
        .collect();
    let mut ordered: Vec<((&str, &str), usize)> = confused
        .into_iter()
        .map(|pair| {
            (
                pair,
                tally(&judged, |actual, predicted| (actual, predicted) == pair),
            )
        })
        .collect();
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
            pair.insert("n".to_string(), serde_json::json!(n));
            Value::Object(pair)
        })
        .collect();

    let mut report = Map::new();
    report.insert("n_sessions".to_string(), serde_json::json!(total));
    report.insert(
        "accuracy".to_string(),
        if judged.is_empty() {
            Value::Null
        } else {
            number(round4(correct as f64 / total as f64))
        },
    );
    report.insert("counts".to_string(), Value::Object(counts));
    report.insert("correct".to_string(), Value::Object(correct_per_class));
    report.insert("confusion".to_string(), Value::Array(pairs));
    Value::Object(report)
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
