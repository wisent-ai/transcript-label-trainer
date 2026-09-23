use super::*;

/// Below this many labeled sessions a classifier is not meaningful, so train
/// refuses with an explicit message instead of fitting noise. This is a
/// product floor, not a library requirement.
pub const MIN_LABELED_SESSIONS: usize = 8;

/// Cross-validated accuracy is reported once every class can spare members
/// for stratified folds; below that the metric would be noise, so it is
/// omitted.
pub(crate) const MIN_SESSIONS_FOR_CV: usize = 10;

/// The fitted tfidf-logreg artifact. Replaces the Python `model.joblib`.
pub(crate) const MODEL_FILE: &str = "model.json";

/// What the Python build wrote instead. Artifacts trained by it are still
/// listed by `info` — they are real training runs with real metrics — but
/// they cannot be loaded for inference, and say so.
pub(crate) const LEGACY_MODEL_FILE: &str = "model.joblib";

pub(crate) const METRICS_FILE: &str = "metrics.json";

/// The `backend` discriminator persisted in every `metrics.json` on disk and
/// printed by `info` and `evaluate`. It stays exactly as the Python build
/// wrote it: renaming it would strand the existing artifacts and change the
/// operator-facing output, neither of which the rewrite is allowed to do.
pub(crate) const TFIDF_BACKEND: &str = "sklearn";

pub(crate) const TFIDF_MODEL_DESC: &str = "tfidf(1-2gram, sublinear) + logistic-regression";

pub fn models_dir() -> PathBuf {
    placement::resolve_placement().training_root.join("models")
}

/// `^[a-z0-9][a-z0-9_-]*$` — an aspect or job name becomes a directory name,
/// so it is checked before it is joined onto a path.
pub(crate) fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

pub fn aspect_dir(aspect: &str) -> Result<PathBuf> {
    if !valid_name(aspect) {
        crate::bail!("invalid aspect name '{aspect}': use lowercase letters, digits, '-' and '_'");
    }
    Ok(models_dir().join(aspect))
}

pub(crate) fn not_enough(message: String) -> TrainFailure {
    TrainFailure::NotEnoughData(message)
}

// ---------------------------------------------------------------------------
// The frame: labels joined with their lake text
// ---------------------------------------------------------------------------

/// Everything a backend needs to train: the frame plus the frozen split.
pub struct Plan {
    pub aspect: String,
    pub out_name: String,
    pub texts: Vec<String>,
    pub values: Vec<String>,
    pub split: evaluate::Split,
    pub job: Option<Value>,
}

/// A job resolved to its training data, without training.
pub struct Resolved {
    pub labels: Vec<lake::SessionLabel>,
    pub counts: BTreeMap<String, usize>,
}

/// Preselected label records joined with their lake text.
///
/// `subject` names the selection in error messages ("aspect 'topic'" for
/// train, "job 'topic-v1' (…)" for run). Returns the rows that survived, in
/// selection order, plus the per-value counts. Fails with the exact numbers
/// when the selection cannot be trained.
pub(crate) fn frame_from_labels(
    labels: &[lake::SessionLabel],
    subject: &str,
    min_text_chars: Option<u64>,
) -> Result<(Vec<lake::SessionLabel>, BTreeMap<String, usize>), TrainFailure> {
    let n_labeled = labels.len();
    if n_labeled < MIN_LABELED_SESSIONS {
        return Err(not_enough(format!(
            "{subject} has {n_labeled} labeled session(s); at least \
             {MIN_LABELED_SESSIONS} are required to train. Add labels with \
             'transcript-lake label add' and retry."
        )));
    }
    let ids: Vec<String> = labels.iter().map(|l| l.session_id.clone()).collect();
    let texts_by_id = lake::session_texts(&ids)?;

    let mut rows: Vec<lake::SessionLabel> = Vec::with_capacity(labels.len());
    let mut skipped = 0usize;
    let mut skipped_short = 0usize;
    for label in labels {
        let text = match texts_by_id.get(&label.session_id) {
            Some(entry) if !entry.text.trim().is_empty() => &entry.text,
            _ => {
                skipped += 1;
                continue;
            }
        };
        if let Some(min_chars) = min_text_chars {
            if (text.chars().count() as u64) < min_chars {
                skipped_short += 1;
                continue;
            }
        }
        let mut row = label.clone();
        row.text = text.clone();
        rows.push(row);
    }
    if skipped > 0 {
        lake::warn(&format!(
            "{skipped} labeled session(s) had no text in the lake and were skipped"
        ));
    }
    if skipped_short > 0 {
        let min_chars = min_text_chars.unwrap_or(0);
        lake::warn(&format!(
            "{skipped_short} labeled session(s) were shorter than \
             scope.min_text_chars={min_chars} and were skipped"
        ));
    }

    let counts = class_counts(rows.iter().map(|r| r.value.as_str()));
    if rows.len() < MIN_LABELED_SESSIONS || counts.len() < 2 {
        return Err(not_enough(format!(
            "{subject} has {} usable labeled session(s) across {} distinct \
             value(s); at least {MIN_LABELED_SESSIONS} sessions and 2 distinct \
             values are required to train. Add labels with 'transcript-lake \
             label add' and retry.",
            rows.len(),
            counts.len()
        )));
    }
    Ok((rows, counts))
}

pub(crate) fn class_counts<'a>(values: impl Iterator<Item = &'a str>) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for value in values {
        *counts.entry(value.to_string()).or_insert(0) += 1;
    }
    counts
}

/// A job spec good enough to resolve a split with. `train --aspect` has no
/// job file, and `labels_for_artifact` reconstructs only the selection half
/// of one, so both build their `Job` here rather than inventing a second
/// shape for the same thing.
pub(crate) fn synthetic_job(name: &str, aspect: &str, eval_split: jobs::EvalSplit) -> jobs::Job {
    jobs::Job {
        name: name.to_string(),
        task: String::new(),
        evaluator: String::new(),
        model: jobs::SKLEARN_MODEL.to_string(),
        scope: jobs::Scope {
            aspect: aspect.to_string(),
            runtimes: None,
            since: None,
            values: None,
            min_text_chars: None,
        },
        eval_split,
        judge: jobs::Judge {
            enabled: false,
            model: None,
        },
    }
}

/// Resolving the split here — before any backend runs — is what lets `run`
/// print the train/holdout counts ahead of training, and what keeps both
/// backends of one job scored on the same untouched sessions.
pub(crate) fn build_plan(
    aspect: &str,
    labels: &[lake::SessionLabel],
    subject: &str,
    out_name: &str,
    eval_split: jobs::EvalSplit,
    min_text_chars: Option<u64>,
    job_meta: Option<Value>,
) -> Result<Plan, TrainFailure> {
    let (rows, _counts) = frame_from_labels(labels, subject, min_text_chars)?;
    let job = synthetic_job(out_name, aspect, eval_split);
    let split = evaluate::resolve_split(&job, &rows, subject)?;
    Ok(Plan {
        aspect: aspect.to_string(),
        out_name: out_name.to_string(),
        texts: rows.iter().map(|r| r.text.clone()).collect(),
        values: rows.iter().map(|r| r.value.clone()).collect(),
        split,
        job: job_meta,
    })
}

/// The (texts, values) of one side of a plan.
pub(crate) fn side(plan: &Plan, index: &[usize]) -> (Vec<String>, Vec<String>) {
    let texts = index.iter().map(|&i| plan.texts[i].clone()).collect();
    let values = index.iter().map(|&i| plan.values[i].clone()).collect();
    (texts, values)
}

/// The split line printed before training starts.
pub fn split_summary(plan: &Plan) -> Value {
    plan.split.frozen.clone()
}

// ---------------------------------------------------------------------------
// TF-IDF vectorizer
//
// sklearn's TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True) with every
// other setting left at its default, reimplemented so the crate carries no ML
// dependency. The defaults that matter, and that the artifact records:
//   lowercase=True, analyzer='word', token_pattern=r'(?u)\b\w\w+\b',
//   min_df=1, max_df=1.0 (neither prunes anything), smooth_idf=True,
//   use_idf=True, norm='l2', binary=False.
// ---------------------------------------------------------------------------

pub(crate) const TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";

pub(crate) const NGRAM_MAX: usize = 2;

pub(crate) const SUBLINEAR_TF: bool = true;

pub(crate) const SMOOTH_IDF: bool = true;

/// Terms in fewer than this many documents are dropped. 1 drops nothing.
pub(crate) const MIN_DF: f64 = 1.0;

/// Terms in more than this share of documents are dropped. 1.0 drops nothing.
pub(crate) const MAX_DF: f64 = 1.0;

/// One document as (column, weight) pairs, ascending by column. Sorted so
/// that every floating-point accumulation over a row happens in one fixed
/// order — hash iteration order must never reach the arithmetic.
pub(crate) type SparseRow = Vec<(u32, f64)>;

/// `(?u)\b\w\w+\b`: maximal runs of word characters, two or more long.
/// Written out by hand because the crate carries no regex engine and this is
/// the whole of the pattern.
pub(crate) fn tokenize(prepared: &str, out: &mut Vec<String>) {
    let mut start: Option<usize> = None;
    let mut len = 0usize;
    for (offset, ch) in prepared.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(offset);
                len = 0;
            }
            len += 1;
        } else if let Some(begin) = start.take() {
            if len >= 2 {
                out.push(prepared[begin..offset].to_string());
            }
        }
    }
    if let Some(begin) = start {
        if len >= 2 {
            out.push(prepared[begin..].to_string());
        }
    }
}
