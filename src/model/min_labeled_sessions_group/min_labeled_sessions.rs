use super::*;

/// Whether `items` yields at least two entries: a classifier needs two
/// distinct values to separate, a library requirement and not a choice.
pub(crate) fn at_least_two<T>(mut items: impl Iterator<Item = T>) -> bool {
    items.next().is_some() && items.next().is_some()
}

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

/// How `info` names a tfidf-logreg model trained with `settings`.
pub(crate) fn tfidf_model_desc(settings: &jobs::TfidfTraining) -> String {
    let tf = if settings.sublinear_tf { "sublinear" } else { "raw" };
    format!("tfidf(word n-grams up to {}, {tf} tf) + logistic-regression", settings.ngram_max)
}

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
/// train, "job 'topic-v1' (…)" for run), and `min_sessions` is the caller's
/// stated floor of labeled sessions. Returns the rows that survived, in
/// selection order, plus the per-value counts. Fails with the exact numbers
/// when the selection cannot be trained.
pub(crate) fn frame_from_labels(
    labels: &[lake::SessionLabel],
    subject: &str,
    min_text_chars: Option<u64>,
    min_sessions: usize,
) -> Result<(Vec<lake::SessionLabel>, BTreeMap<String, usize>), TrainFailure> {
    let n_labeled = labels.len();
    if n_labeled < min_sessions {
        return Err(not_enough(format!(
            "{subject} has {n_labeled} labeled session(s); at least \
             {min_sessions} are required to train. Add labels with \
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
    if rows.len() < min_sessions || !at_least_two(counts.keys()) {
        return Err(not_enough(format!(
            "{subject} has {} usable labeled session(s) across {} distinct \
             value(s); at least {min_sessions} sessions and two distinct \
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
    min_sessions: usize,
    job_meta: Option<Value>,
) -> Result<Plan, TrainFailure> {
    let (rows, _counts) = frame_from_labels(labels, subject, min_text_chars, min_sessions)?;
    let request = evaluate::SplitRequest {
        name: out_name,
        eval_split: &eval_split,
        min_labeled_sessions: min_sessions,
    };
    let split = evaluate::resolve_split(&request, &rows, subject)?;
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
// sklearn's TfidfVectorizer reimplemented so the crate carries no ML
// dependency: word analyzer, the token pattern below, L2 row norm, idf on.
// The n-gram length, lowercasing, sublinear tf, idf smoothing and document
// frequency cuts are the run's stated settings, recorded in the artifact.
// ---------------------------------------------------------------------------

pub(crate) const TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";

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
