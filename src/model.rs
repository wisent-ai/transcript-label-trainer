//! Training, inference, and artifact inspection for aspect-label classifiers.
//!
//! Two backends share one data path (lake label store + lake CLI session text):
//!
//! - tfidf-logreg (default): TF-IDF + multinomial logistic regression,
//!   artifacts directly in `<training root>/models/<aspect>/` (`model.json` +
//!   `metrics.json`);
//! - hf (`--model <hf-model-id>`, optional `hf` feature): fine-tuned
//!   sequence classifier in `<training root>/models/<aspect>/hf-<id>/`.
//!
//! The training root is resolved by [`crate::placement`], never read straight
//! out of the environment here.
//!
//! Neither backend ever trains on the frozen evaluation split:
//! [`crate::evaluate::resolve_split`] resolves it before training starts, both
//! backends fit on the training side only, and both report the holdout under
//! `holdout_evaluation`.
//!
//! When both backends exist for an aspect, inference uses the newest artifact
//! by `trained_at`.
//!
//! The Python original delegated the vectorizer and the classifier to sklearn
//! and froze the fitted pipeline into `model.joblib`. A pickle of Python
//! objects is not portable to this binary, so the artifact is `model.json`:
//! the vocabulary, the idf vector, the class list and the weight matrix, in
//! the one format anything can read. Everything around it — `metrics.json`,
//! `eval-split.json`, `job.yaml`, the metric keys, the guard rails — is
//! unchanged, because `info`, `evaluate` and the docs read those.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::util::{Error, Result, TrainFailure};
use crate::{evaluate, jobs, lake, placement};

/// Below this many labeled sessions a classifier is not meaningful, so train
/// refuses with an explicit message instead of fitting noise. This is a
/// product floor, not a library requirement.
pub const MIN_LABELED_SESSIONS: usize = 8;

/// Cross-validated accuracy is reported once every class can spare members
/// for stratified folds; below that the metric would be noise, so it is
/// omitted.
const MIN_SESSIONS_FOR_CV: usize = 10;

/// The fitted tfidf-logreg artifact. Replaces the Python `model.joblib`.
const MODEL_FILE: &str = "model.json";

/// What the Python build wrote instead. Artifacts trained by it are still
/// listed by `info` — they are real training runs with real metrics — but
/// they cannot be loaded for inference, and say so.
const LEGACY_MODEL_FILE: &str = "model.joblib";

const METRICS_FILE: &str = "metrics.json";

/// The `backend` discriminator persisted in every `metrics.json` on disk and
/// printed by `info` and `evaluate`. It stays exactly as the Python build
/// wrote it: renaming it would strand the existing artifacts and change the
/// operator-facing output, neither of which the rewrite is allowed to do.
const TFIDF_BACKEND: &str = "sklearn";

const TFIDF_MODEL_DESC: &str = "tfidf(1-2gram, sublinear) + logistic-regression";

pub fn models_dir() -> PathBuf {
    placement::resolve_placement().training_root.join("models")
}

/// `^[a-z0-9][a-z0-9_-]*$` — an aspect or job name becomes a directory name,
/// so it is checked before it is joined onto a path.
fn valid_name(name: &str) -> bool {
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

fn not_enough(message: String) -> TrainFailure {
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
fn frame_from_labels(
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

fn class_counts<'a>(values: impl Iterator<Item = &'a str>) -> BTreeMap<String, usize> {
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
fn synthetic_job(name: &str, aspect: &str, eval_split: jobs::EvalSplit) -> jobs::Job {
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
        judge: jobs::Judge { enabled: false, model: None },
    }
}

/// Resolving the split here — before any backend runs — is what lets `run`
/// print the train/holdout counts ahead of training, and what keeps both
/// backends of one job scored on the same untouched sessions.
fn build_plan(
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
fn side(plan: &Plan, index: &[usize]) -> (Vec<String>, Vec<String>) {
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

const TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";
const NGRAM_MAX: usize = 2;
const SUBLINEAR_TF: bool = true;
const SMOOTH_IDF: bool = true;
/// Terms in fewer than this many documents are dropped. 1 drops nothing.
const MIN_DF: f64 = 1.0;
/// Terms in more than this share of documents are dropped. 1.0 drops nothing.
const MAX_DF: f64 = 1.0;

/// One document as (column, weight) pairs, ascending by column. Sorted so
/// that every floating-point accumulation over a row happens in one fixed
/// order — hash iteration order must never reach the arithmetic.
type SparseRow = Vec<(u32, f64)>;

/// `(?u)\b\w\w+\b`: maximal runs of word characters, two or more long.
/// Written out by hand because the crate carries no regex engine and this is
/// the whole of the pattern.
fn tokenize(prepared: &str, out: &mut Vec<String>) {
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

/// Term counts for one document: unigrams plus, when `ngram_max` is 2, the
/// space-joined adjacent pairs sklearn's `_word_ngrams` produces.
fn count_document(doc: &str, lowercase: bool, ngram_max: usize) -> HashMap<String, usize> {
    let prepared = if lowercase { doc.to_lowercase() } else { doc.to_string() };
    let mut unigrams: Vec<String> = Vec::new();
    tokenize(&prepared, &mut unigrams);
    let mut counts: HashMap<String, usize> =
        HashMap::with_capacity(unigrams.len() * ngram_max.max(1));
    for n in 1..=ngram_max {
        if unigrams.len() < n {
            break;
        }
        for window in unigrams.windows(n) {
            *counts.entry(window.join(" ")).or_insert(0) += 1;
        }
    }
    counts
}

struct Vectorizer {
    lowercase: bool,
    ngram_max: usize,
    sublinear_tf: bool,
    vocabulary: Vec<String>,
    index: HashMap<String, u32>,
    idf: Vec<f64>,
}

impl Vectorizer {
    /// Fit on a corpus and return the fitted rows in the same pass, the way
    /// `Pipeline.fit` does — the documents are counted exactly once.
    fn fit_transform(docs: &[String]) -> (Self, Vec<SparseRow>) {
        let counts: Vec<HashMap<String, usize>> = docs
            .iter()
            .map(|doc| count_document(doc, true, NGRAM_MAX))
            .collect();

        let n_docs = docs.len() as f64;
        let mut df: HashMap<&str, usize> = HashMap::new();
        for doc in &counts {
            for term in doc.keys() {
                *df.entry(term.as_str()).or_insert(0) += 1;
            }
        }

        let high = MAX_DF * n_docs;
        let mut vocabulary: Vec<String> = df
            .iter()
            .filter(|(_, &count)| count as f64 >= MIN_DF && count as f64 <= high)
            .map(|(term, _)| (*term).to_string())
            .collect();
        vocabulary.sort_unstable();

        let mut index = HashMap::with_capacity(vocabulary.len());
        let mut idf = Vec::with_capacity(vocabulary.len());
        for (column, term) in vocabulary.iter().enumerate() {
            index.insert(term.clone(), column as u32);
            let document_frequency = *df.get(term.as_str()).unwrap_or(&0) as f64;
            idf.push(if SMOOTH_IDF {
                ((1.0 + n_docs) / (1.0 + document_frequency)).ln() + 1.0
            } else {
                (n_docs / document_frequency).ln() + 1.0
            });
        }

        let vectorizer = Vectorizer {
            lowercase: true,
            ngram_max: NGRAM_MAX,
            sublinear_tf: SUBLINEAR_TF,
            vocabulary,
            index,
            idf,
        };
        let rows = counts.iter().map(|doc| vectorizer.row(doc)).collect();
        (vectorizer, rows)
    }

    fn transform(&self, docs: &[String]) -> Vec<SparseRow> {
        docs.iter()
            .map(|doc| self.row(&count_document(doc, self.lowercase, self.ngram_max)))
            .collect()
    }

    /// tf (sublinear) x idf, then L2 normalisation, exactly the order
    /// `TfidfTransformer` applies them in. Terms outside the vocabulary are
    /// dropped, which is what makes inference on unseen text well-defined.
    fn row(&self, counts: &HashMap<String, usize>) -> SparseRow {
        let mut row: SparseRow = counts
            .iter()
            .filter_map(|(term, &count)| {
                self.index.get(term.as_str()).map(|&column| {
                    let tf = if self.sublinear_tf {
                        1.0 + (count as f64).ln()
                    } else {
                        count as f64
                    };
                    (column, tf * self.idf[column as usize])
                })
            })
            .collect();
        row.sort_unstable_by_key(|&(column, _)| column);
        let norm = row.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for entry in row.iter_mut() {
                entry.1 /= norm;
            }
        }
        row
    }

    fn n_features(&self) -> usize {
        self.vocabulary.len()
    }
}

// ---------------------------------------------------------------------------
// Multinomial logistic regression
//
// The objective sklearn's LogisticRegression(max_iter=1000) minimises with its
// defaults: softmax cross-entropy plus an L2 penalty of 1/C = 1 on the
// coefficients (never on the intercepts), solved with L-BFGS. There is no
// randomness anywhere in the fit — the starting point is zero, the sample
// order is the plan's order, and every sum runs over sorted columns — so two
// runs on identical input produce bit-identical weights. That is a product
// property, not an implementation detail: the frozen eval split exists to
// compare models over time, and it can only do that if a model is a function
// of its inputs alone.
// ---------------------------------------------------------------------------

/// 1 / C for sklearn's default C = 1.0.
const L2_ALPHA: f64 = 1.0;
const MAX_ITER: usize = 1000;
/// Stop when the largest gradient component falls below this — the same
/// criterion, and the same value, as the lbfgs solver's default `tol`.
const TOL: f64 = 1e-4;
const LBFGS_MEMORY: usize = 8;
/// Armijo sufficient-decrease constant, the backtracking factor, and the cap
/// on halvings before the step is abandoned.
const ARMIJO_C1: f64 = 1e-4;
const BACKTRACK: f64 = 0.5;
const MAX_BACKTRACKS: usize = 40;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Parameters are one flat vector: `k * f` coefficients, class-major, then
/// `k` intercepts.
struct Problem<'a> {
    rows: &'a [SparseRow],
    y: &'a [usize],
    n_classes: usize,
    n_features: usize,
}

impl Problem<'_> {
    fn dim(&self) -> usize {
        self.n_classes * (self.n_features + 1)
    }

    fn loss_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let (k, f) = (self.n_classes, self.n_features);
        let mut grad = vec![0.0f64; self.dim()];
        let mut loss = 0.0f64;
        let mut z = vec![0.0f64; k];
        for (i, row) in self.rows.iter().enumerate() {
            for c in 0..k {
                let base = c * f;
                let mut score = x[k * f + c];
                for &(column, value) in row {
                    score += x[base + column as usize] * value;
                }
                z[c] = score;
            }
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            for &score in z.iter() {
                sum_exp += (score - max).exp();
            }
            let log_sum_exp = max + sum_exp.ln();
            loss += log_sum_exp - z[self.y[i]];
            for c in 0..k {
                let delta = (z[c] - log_sum_exp).exp() - if c == self.y[i] { 1.0 } else { 0.0 };
                grad[k * f + c] += delta;
                let base = c * f;
                for &(column, value) in row {
                    grad[base + column as usize] += delta * value;
                }
            }
        }
        let mut squared = 0.0f64;
        for c in 0..k {
            for j in 0..f {
                let at = c * f + j;
                squared += x[at] * x[at];
                grad[at] += L2_ALPHA * x[at];
            }
        }
        (loss + 0.5 * L2_ALPHA * squared, grad)
    }
}

struct Fit {
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    iterations: usize,
    converged: bool,
}

/// L-BFGS with a two-loop recursion, `LBFGS_MEMORY` correction pairs and an
/// Armijo backtracking line search. Curvature pairs whose `s·y` is not
/// positive are skipped rather than stored, which keeps the implicit inverse
/// Hessian positive definite without needing a full Wolfe search.
fn fit_lbfgs(problem: &Problem<'_>) -> (Vec<f64>, usize, bool) {
    let dim = problem.dim();
    let mut x = vec![0.0f64; dim];
    let (mut fx, mut g) = problem.loss_grad(&x);
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    let mut iterations = 0usize;

    while iterations < MAX_ITER {
        let gradient_max = g.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if gradient_max <= TOL {
            return (x, iterations, true);
        }

        let mut q = g.clone();
        let mut alpha = vec![0.0f64; s_history.len()];
        for i in (0..s_history.len()).rev() {
            let a = rho[i] * dot(&s_history[i], &q);
            alpha[i] = a;
            for (slot, &yi) in q.iter_mut().zip(y_history[i].iter()) {
                *slot -= a * yi;
            }
        }
        if let Some(last) = s_history.len().checked_sub(1) {
            let yy = dot(&y_history[last], &y_history[last]);
            if yy > 0.0 {
                let gamma = dot(&s_history[last], &y_history[last]) / yy;
                for slot in q.iter_mut() {
                    *slot *= gamma;
                }
            }
        }
        for i in 0..s_history.len() {
            let beta = rho[i] * dot(&y_history[i], &q);
            let scale = alpha[i] - beta;
            for (slot, &si) in q.iter_mut().zip(s_history[i].iter()) {
                *slot += scale * si;
            }
        }

        let mut direction: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&direction, &g);
        if !(slope < 0.0) {
            // Not a descent direction, or not finite: fall back to steepest
            // descent, which always is one.
            direction = g.iter().map(|v| -v).collect();
            slope = dot(&direction, &g);
        }

        let mut step = 1.0f64;
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACKS {
            let candidate: Vec<f64> = (0..dim).map(|i| x[i] + step * direction[i]).collect();
            let (candidate_f, candidate_g) = problem.loss_grad(&candidate);
            if candidate_f.is_finite() && candidate_f <= fx + ARMIJO_C1 * step * slope {
                accepted = Some((candidate, candidate_f, candidate_g));
                break;
            }
            step *= BACKTRACK;
        }
        let Some((new_x, new_f, new_g)) = accepted else {
            // No step along a descent direction lowers the objective: the
            // iterate is at the precision floor, which is as converged as
            // this arithmetic gets.
            return (x, iterations, true);
        };

        let s: Vec<f64> = (0..dim).map(|i| new_x[i] - x[i]).collect();
        let y: Vec<f64> = (0..dim).map(|i| new_g[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_history.len() == LBFGS_MEMORY {
                s_history.remove(0);
                y_history.remove(0);
                rho.remove(0);
            }
            rho.push(1.0 / sy);
            s_history.push(s);
            y_history.push(y);
        }
        x = new_x;
        fx = new_f;
        g = new_g;
        iterations += 1;
    }
    (x, iterations, false)
}

fn fit_logistic(rows: &[SparseRow], y: &[usize], n_classes: usize, n_features: usize) -> Fit {
    let problem = Problem { rows, y, n_classes, n_features };
    let (x, iterations, converged) = fit_lbfgs(&problem);
    let coef = (0..n_classes)
        .map(|c| x[c * n_features..(c + 1) * n_features].to_vec())
        .collect();
    let intercept = x[n_classes * n_features..].to_vec();
    Fit { coef, intercept, iterations, converged }
}

/// (class index, probability) for one row, first maximum winning — the same
/// tie-break `numpy.argmax` gives, over the same sorted class order.
fn predict_row(coef: &[Vec<f64>], intercept: &[f64], row: &SparseRow) -> (usize, f64) {
    let k = intercept.len();
    let mut z = Vec::with_capacity(k);
    for c in 0..k {
        let weights = &coef[c];
        let mut score = intercept[c];
        for &(column, value) in row {
            score += weights[column as usize] * value;
        }
        z.push(score);
    }
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = z.iter().map(|s| (s - max).exp()).sum();
    let mut best = 0usize;
    for c in 1..k {
        if z[c] > z[best] {
            best = c;
        }
    }
    (best, (z[best] - max).exp() / sum_exp)
}

// ---------------------------------------------------------------------------
// The tfidf-logreg artifact
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct VectorizerFile {
    analyzer: String,
    lowercase: bool,
    token_pattern: String,
    ngram_range: [usize; 2],
    sublinear_tf: bool,
    smooth_idf: bool,
    norm: String,
    min_df: f64,
    max_df: f64,
    vocabulary: Vec<String>,
    idf: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct ClassifierFile {
    kind: String,
    multi_class: String,
    #[serde(rename = "C")]
    c: f64,
    max_iter: usize,
    tol: f64,
    iterations: usize,
    converged: bool,
    classes: Vec<String>,
    intercept: Vec<f64>,
    coef: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct ModelFile {
    backend: String,
    format: u32,
    vectorizer: VectorizerFile,
    classifier: ClassifierFile,
}

struct TfidfModel {
    vectorizer: Vectorizer,
    classes: Vec<String>,
    coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    iterations: usize,
    converged: bool,
}

impl TfidfModel {
    fn fit(texts: &[String], values: &[String]) -> TfidfModel {
        let classes: Vec<String> = {
            let mut distinct: Vec<String> = values.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            distinct
        };
        let class_index: HashMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, value)| (value.as_str(), i))
            .collect();
        let y: Vec<usize> = values.iter().map(|v| class_index[v.as_str()]).collect();

        let (vectorizer, rows) = Vectorizer::fit_transform(texts);
        let fit = fit_logistic(&rows, &y, classes.len(), vectorizer.n_features());
        TfidfModel {
            vectorizer,
            classes,
            coef: fit.coef,
            intercept: fit.intercept,
            iterations: fit.iterations,
            converged: fit.converged,
        }
    }

    /// (value, confidence) per text.
    fn predict(&self, texts: &[String]) -> Vec<(String, f64)> {
        self.vectorizer
            .transform(texts)
            .iter()
            .map(|row| {
                let (best, probability) = predict_row(&self.coef, &self.intercept, row);
                (self.classes[best].clone(), probability)
            })
            .collect()
    }

    fn to_file(&self) -> ModelFile {
        ModelFile {
            backend: TFIDF_BACKEND.to_string(),
            format: 1,
            vectorizer: VectorizerFile {
                analyzer: "word".to_string(),
                lowercase: self.vectorizer.lowercase,
                token_pattern: TOKEN_PATTERN.to_string(),
                ngram_range: [1, self.vectorizer.ngram_max],
                sublinear_tf: self.vectorizer.sublinear_tf,
                smooth_idf: SMOOTH_IDF,
                norm: "l2".to_string(),
                min_df: MIN_DF,
                max_df: MAX_DF,
                vocabulary: self.vectorizer.vocabulary.clone(),
                idf: self.vectorizer.idf.clone(),
            },
            classifier: ClassifierFile {
                kind: "logistic-regression".to_string(),
                multi_class: "multinomial".to_string(),
                c: 1.0 / L2_ALPHA,
                max_iter: MAX_ITER,
                tol: TOL,
                iterations: self.iterations,
                converged: self.converged,
                classes: self.classes.clone(),
                intercept: self.intercept.clone(),
                coef: self.coef.clone(),
            },
        }
    }

    fn from_file(file: ModelFile) -> Result<TfidfModel> {
        let VectorizerFile { lowercase, ngram_range, sublinear_tf, vocabulary, idf, .. } =
            file.vectorizer;
        if vocabulary.len() != idf.len() {
            crate::bail!(
                "model file is inconsistent: {} vocabulary term(s) but {} idf weight(s)",
                vocabulary.len(),
                idf.len()
            );
        }
        let classifier = file.classifier;
        if classifier.classes.len() != classifier.coef.len()
            || classifier.classes.len() != classifier.intercept.len()
        {
            crate::bail!(
                "model file is inconsistent: {} class(es) but {} weight row(s) and {} intercept(s)",
                classifier.classes.len(),
                classifier.coef.len(),
                classifier.intercept.len()
            );
        }
        let index = vocabulary
            .iter()
            .enumerate()
            .map(|(column, term)| (term.clone(), column as u32))
            .collect();
        Ok(TfidfModel {
            vectorizer: Vectorizer {
                lowercase,
                ngram_max: ngram_range[1],
                sublinear_tf,
                vocabulary,
                index,
                idf,
            },
            classes: classifier.classes,
            coef: classifier.coef,
            intercept: classifier.intercept,
            iterations: classifier.iterations,
            converged: classifier.converged,
        })
    }
}

// ---------------------------------------------------------------------------
// Cross-validated accuracy
// ---------------------------------------------------------------------------

/// splitmix64: enough of a generator to shuffle one class's members, small
/// enough to be obviously reproducible. It stands in for the numpy
/// `RandomState` behind `StratifiedKFold(shuffle=True, random_state=0)`,
/// which cannot be reproduced outside numpy and whose exact permutation was
/// never part of the product — the reported metric is.
struct Splitmix(u64);

impl Splitmix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

const CV_SEED: u64 = 0;

/// Mean accuracy over stratified folds, each fold refitting the whole
/// pipeline — vectorizer included — on the other folds, the way
/// `cross_val_score` over a `Pipeline` does. Shuffling is keyed per class so
/// a class that gains members does not reshuffle the others.
fn cross_val_accuracy(texts: &[String], values: &[String], folds: usize) -> f64 {
    let mut by_class: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, value) in values.iter().enumerate() {
        by_class.entry(value.as_str()).or_default().push(i);
    }
    let mut assignment = vec![0usize; texts.len()];
    for (class_number, (_, members)) in by_class.iter_mut().enumerate() {
        let mut rng = Splitmix(CV_SEED.wrapping_add(class_number as u64));
        rng.shuffle(members);
        for (position, &row) in members.iter().enumerate() {
            assignment[row] = position % folds;
        }
    }

    let mut scores = Vec::with_capacity(folds);
    for fold in 0..folds {
        let mut train_texts = Vec::new();
        let mut train_values = Vec::new();
        let mut test_texts = Vec::new();
        let mut test_values = Vec::new();
        for i in 0..texts.len() {
            if assignment[i] == fold {
                test_texts.push(texts[i].clone());
                test_values.push(values[i].clone());
            } else {
                train_texts.push(texts[i].clone());
                train_values.push(values[i].clone());
            }
        }
        let distinct: HashSet<&String> = train_values.iter().collect();
        if test_texts.is_empty() || distinct.len() < 2 {
            continue;
        }
        let model = TfidfModel::fit(&train_texts, &train_values);
        let correct = model
            .predict(&test_texts)
            .iter()
            .zip(&test_values)
            .filter(|((predicted, _), actual)| predicted == *actual)
            .count();
        scores.push(correct as f64 / test_texts.len() as f64);
    }
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn base_metrics(
    aspect: &str,
    backend: &str,
    model_desc: &str,
    n_sessions: usize,
    counts: &BTreeMap<String, usize>,
) -> Map<String, Value> {
    let mut metrics = Map::new();
    metrics.insert("aspect".to_string(), json!(aspect));
    metrics.insert("backend".to_string(), json!(backend));
    metrics.insert("trained_at".to_string(), json!(crate::util::now_iso()));
    metrics.insert("trainer_version".to_string(), json!(env!("CARGO_PKG_VERSION")));
    metrics.insert("model".to_string(), json!(model_desc));
    metrics.insert("n_sessions".to_string(), json!(n_sessions));
    metrics.insert("classes".to_string(), json!(counts.keys().collect::<Vec<_>>()));
    metrics.insert("counts".to_string(), counts_json(counts));
    metrics
}

fn counts_json(counts: &BTreeMap<String, usize>) -> Value {
    let mut object = Map::new();
    for (value, count) in counts {
        object.insert(value.clone(), json!(count));
    }
    Value::Object(object)
}

fn write_pretty(path: &Path, value: &Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// tfidf-logreg backend
// ---------------------------------------------------------------------------

fn train_tfidf(plan: &Plan) -> Result<Value, TrainFailure> {
    let (texts, values) = side(plan, &plan.split.train_index);
    let counts = class_counts(values.iter().map(String::as_str));

    let min_class = counts.values().copied().min().unwrap_or(0);
    let mut cv_accuracy: Option<f64> = None;
    let mut cv_folds = 0usize;
    if texts.len() >= MIN_SESSIONS_FOR_CV && min_class >= 2 {
        cv_folds = std::cmp::min(5, min_class);
        cv_accuracy = Some(round4(cross_val_accuracy(&texts, &values, cv_folds)));
    }

    let model = TfidfModel::fit(&texts, &values);

    let out_dir = aspect_dir(&plan.out_name)?;
    std::fs::create_dir_all(&out_dir).map_err(Error::from)?;
    let model_path = out_dir.join(MODEL_FILE);

    let mut metrics =
        base_metrics(&plan.aspect, TFIDF_BACKEND, TFIDF_MODEL_DESC, texts.len(), &counts);
    metrics.insert(
        "cv_accuracy".to_string(),
        match cv_accuracy {
            Some(value) => json!(value),
            None => Value::Null,
        },
    );
    metrics.insert("cv_folds".to_string(), json!(cv_folds));
    metrics.insert("eval_split".to_string(), plan.split.frozen.clone());
    metrics.insert("model_path".to_string(), json!(model_path.display().to_string()));

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
    let serialized = serde_json::to_string(&model.to_file()).map_err(Error::from)?;
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
const HF_FEATURE_MISSING: &str = "fine-tuning with --model requires the optional 'hf' feature; \
     rebuild with: cargo build --release --features hf";

#[cfg(feature = "hf")]
fn train_hf(
    plan: &Plan,
    model_id: &str,
    epochs: f64,
    batch_size: usize,
    lr: f64,
    max_length: usize,
) -> Result<Value, TrainFailure> {
    let (texts, values) = side(plan, &plan.split.train_index);
    let counts = class_counts(values.iter().map(String::as_str));

    let out_dir = aspect_dir(&plan.out_name)?;
    std::fs::create_dir_all(&out_dir).map_err(Error::from)?;

    let config = crate::hf::TrainConfig {
        aspect: &plan.aspect,
        model_id,
        epochs,
        batch_size,
        lr,
        max_length,
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
        let predictions = crate::hf::predict(&trained.dir, &holdout_texts, max_length)?;
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
fn train_hf(
    _plan: &Plan,
    _model_id: &str,
    _epochs: f64,
    _batch_size: usize,
    _lr: f64,
    _max_length: usize,
) -> Result<Value, TrainFailure> {
    Err(TrainFailure::Failed(Error(HF_FEATURE_MISSING.to_string())))
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn train(
    aspect: &str,
    model_id: Option<&str>,
    epochs: f64,
    batch_size: usize,
    lr: f64,
    max_length: usize,
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
    match model_id {
        None => train_tfidf(&plan),
        Some(model_id) => train_hf(&plan, model_id, epochs, batch_size, lr, max_length),
    }
}

// ---------------------------------------------------------------------------
// Declarative training jobs (run command)
// ---------------------------------------------------------------------------

/// A label record's timestamp as a comparable instant. An empty or
/// unparseable `ts` sorts before every real one, which is what excludes it
/// from a `since` window — the Python original did the same with
/// `datetime.min`.
fn label_instant(ts: &str) -> (i64, u32) {
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

/// Label records matching a job's evaluator and scope filters.
pub fn select_labels(job: &jobs::Job) -> Result<Vec<lake::SessionLabel>> {
    let labels = lake::load_labels(&job.scope.aspect)?;
    let since = job.scope.since.as_deref().map(label_instant);
    Ok(labels
        .into_iter()
        .filter(|record| record.source.trim() == job.evaluator)
        .filter(|record| match &job.scope.runtimes {
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
        .filter(|record| match &job.scope.values {
            Some(allowed) => allowed.iter().any(|value| value == &record.value),
            None => true,
        })
        .collect())
}

fn job_scope_json(scope: &jobs::Scope) -> Value {
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
fn scope_from_json(scope: &Value) -> Result<jobs::Scope> {
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
        since: scope.get("since").and_then(Value::as_str).map(str::to_string),
        values: string_list("values"),
        min_text_chars: scope.get("min_text_chars").and_then(Value::as_u64),
    })
}

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
fn artifacts(name: &str) -> Result<Vec<Artifact>> {
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

fn trained_at(artifact: &Artifact) -> &str {
    artifact.metrics.get("trained_at").and_then(Value::as_str).unwrap_or("")
}

fn read_metrics(path: &Path) -> Result<Value> {
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
        job_meta.get("name").and_then(Value::as_str).unwrap_or("artifact"),
        &scope.aspect,
        jobs::EvalSplit { enabled: false, fraction: None, seed: None },
    );
    job.evaluator = job_meta
        .get("evaluator")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    job.scope = scope;
    select_labels(&job)
}

fn infer_tfidf(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    let path = artifact.dir.join(MODEL_FILE);
    if !path.is_file() {
        if artifact.dir.join(LEGACY_MODEL_FILE).is_file() {
            let name =
                artifact.dir.file_name().and_then(|name| name.to_str()).unwrap_or("<name>");
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
fn infer_hf(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    let max_length = artifact
        .metrics
        .get("hyperparameters")
        .and_then(|hyperparameters| hyperparameters.get("max_length"))
        .and_then(Value::as_u64)
        .unwrap_or(512) as usize;
    crate::hf::predict(&artifact.dir, texts, max_length)
}

#[cfg(not(feature = "hf"))]
fn infer_hf(artifact: &Artifact, _texts: &[String]) -> Result<Vec<(String, f64)>> {
    Err(Error(format!(
        "the artifact in {} is a fine-tuned HuggingFace model, which this build cannot \
         load; rebuild with: cargo build --release --features hf",
        artifact.dir.display()
    )))
}

/// (value, confidence) per text, from whichever backend the artifact is.
pub fn predict(artifact: &Artifact, texts: &[String]) -> Result<Vec<(String, f64)>> {
    if artifact.backend == TFIDF_BACKEND {
        infer_tfidf(artifact, texts)
    } else {
        infer_hf(artifact, texts)
    }
}

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
        let Some(entry) = texts_by_id.get(&session_id) else { continue };
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
        suggestion.insert("note".to_string(), json!(format!("confidence={confidence:.2}")));
        suggestion.insert("source".to_string(), json!("model"));
        suggestions.push(Value::Object(suggestion));
    }
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
        entry.insert("dir".to_string(), json!(root.join(&name).display().to_string()));
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
