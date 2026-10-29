use super::*;

/// The shortest n-gram counted: a single token.
// https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html : ngram_range starts at unigrams.
const UNIGRAM: usize = 1;

/// The one added to the document count and to every document frequency
/// when idf is smoothed, as if one document held every term, and to every
/// idf so a term in every document still weighs.
// https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting : idf(t) = ln((1 + n) / (1 + df(t))) + 1, smoothed; ln(n / df(t)) + 1 otherwise.
const IDF_ONE: f64 = 1.0;

/// The one sublinear tf adds to the log of a count.
// https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html : sublinear_tf replaces tf with 1 + log(tf).
const SUBLINEAR_ONE: f64 = 1.0;

/// The intercept every class carries beside its coefficients.
// https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression : the model is X w + c, one intercept c per class.
const INTERCEPTS_PER_CLASS: usize = 1;

/// The L2 penalty is half the squared norm, so its gradient is the weight
/// times the strength.
// https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression : the l2 penalty is (1/2) ||w||^2.
const L2_HALF: f64 = 0.5;

/// Term counts for one document: unigrams plus, when `ngram_max` is 2, the
/// space-joined adjacent pairs sklearn's `_word_ngrams` produces.
pub(crate) fn count_document(
    doc: &str,
    lowercase: bool,
    ngram_max: usize,
) -> HashMap<String, usize> {
    let prepared = if lowercase {
        doc.to_lowercase()
    } else {
        doc.to_string()
    };
    let mut unigrams: Vec<String> = Vec::new();
    tokenize(&prepared, &mut unigrams);
    let mut counts: HashMap<String, usize> =
        HashMap::with_capacity(unigrams.len() * ngram_max.max(UNIGRAM));
    for n in UNIGRAM..=ngram_max {
        if unigrams.len() < n {
            break;
        }
        for window in unigrams.windows(n) {
            *counts.entry(window.join(" ")).or_insert(0) += 1;
        }
    }
    counts
}

pub(crate) struct Vectorizer {
    pub(crate) lowercase: bool,
    pub(crate) ngram_max: usize,
    pub(crate) sublinear_tf: bool,
    pub(crate) vocabulary: Vec<String>,
    pub(crate) index: HashMap<String, u32>,
    pub(crate) idf: Vec<f64>,
}

impl Vectorizer {
    /// Fit on a corpus and return the fitted rows in the same pass, the way
    /// `Pipeline.fit` does — the documents are counted exactly once.
    pub(crate) fn fit_transform(
        docs: &[String],
        settings: &jobs::TfidfTraining,
    ) -> (Self, Vec<SparseRow>) {
        let counts: Vec<HashMap<String, usize>> = docs
            .iter()
            .map(|doc| count_document(doc, settings.lowercase, settings.ngram_max))
            .collect();

        let n_docs = docs.len() as f64;
        let mut df: HashMap<&str, usize> = HashMap::new();
        for doc in &counts {
            for term in doc.keys() {
                *df.entry(term.as_str()).or_insert(0) += 1;
            }
        }

        // Each kept term carries the document frequency it was kept for, so
        // its idf is computed from that count and never looked up again.
        let high = settings.max_df * n_docs;
        let mut kept: Vec<(String, usize)> = df
            .iter()
            .filter(|(_, &count)| count as f64 >= settings.min_df && count as f64 <= high)
            .map(|(term, &count)| ((*term).to_string(), count))
            .collect();
        kept.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

        let mut index = HashMap::with_capacity(kept.len());
        let mut idf = Vec::with_capacity(kept.len());
        let mut vocabulary = Vec::with_capacity(kept.len());
        for (column, (term, documents)) in kept.into_iter().enumerate() {
            let document_frequency = documents as f64;
            index.insert(term.clone(), column as u32);
            idf.push(if settings.smooth_idf {
                ((IDF_ONE + n_docs) / (IDF_ONE + document_frequency)).ln() + IDF_ONE
            } else {
                (n_docs / document_frequency).ln() + IDF_ONE
            });
            vocabulary.push(term);
        }

        let vectorizer = Vectorizer {
            lowercase: settings.lowercase,
            ngram_max: settings.ngram_max,
            sublinear_tf: settings.sublinear_tf,
            vocabulary,
            index,
            idf,
        };
        let rows = counts.iter().map(|doc| vectorizer.row(doc)).collect();
        (vectorizer, rows)
    }

    pub(crate) fn transform(&self, docs: &[String]) -> Vec<SparseRow> {
        docs.iter()
            .map(|doc| self.row(&count_document(doc, self.lowercase, self.ngram_max)))
            .collect()
    }

    /// tf (sublinear) x idf, then L2 normalisation, exactly the order
    /// `TfidfTransformer` applies them in. Terms outside the vocabulary are
    /// dropped, which is what makes inference on unseen text well-defined.
    /// Every kept weight is a count of at least one times an idf of at least
    /// one, so a row with any term has a positive norm and only an empty row
    /// is left unnormalised.
    pub(crate) fn row(&self, counts: &HashMap<String, usize>) -> SparseRow {
        let mut row: SparseRow = counts
            .iter()
            .filter_map(|(term, &count)| {
                self.index.get(term.as_str()).map(|&column| {
                    let tf = if self.sublinear_tf {
                        SUBLINEAR_ONE + (count as f64).ln()
                    } else {
                        count as f64
                    };
                    (column, tf * self.idf[column as usize])
                })
            })
            .collect();
        row.sort_unstable_by_key(|&(column, _)| column);
        if !row.is_empty() {
            let norm = row.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
            for entry in row.iter_mut() {
                entry.1 /= norm;
            }
        }
        row
    }

    pub(crate) fn n_features(&self) -> usize {
        self.vocabulary.len()
    }
}

// ---------------------------------------------------------------------------
// Multinomial logistic regression
//
// Softmax cross-entropy plus an L2 penalty of 1/c on the coefficients (never
// on the intercepts), solved with L-BFGS under the run's stated settings.
// There is no randomness anywhere in the fit — the starting point is zero,
// the sample order is the plan's order, and every sum runs over sorted
// columns — so two runs on identical input and settings produce
// bit-identical weights. That is a product property, not an implementation
// detail: the frozen eval split exists to compare models over time, and it
// can only do that if a model is a function of its inputs alone.
// ---------------------------------------------------------------------------

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Parameters are one flat vector: `k * f` coefficients, class-major, then
/// `k` intercepts. `l2` is the penalty strength, the inverse of the stated c.
pub(crate) struct Problem<'a> {
    pub(crate) rows: &'a [SparseRow],
    pub(crate) y: &'a [usize],
    pub(crate) n_classes: usize,
    pub(crate) n_features: usize,
    pub(crate) l2: f64,
}

impl Problem<'_> {
    pub(crate) fn dim(&self) -> usize {
        self.n_classes * (self.n_features + INTERCEPTS_PER_CLASS)
    }

    /// The penalised loss and its gradient. The cross-entropy gradient of a
    /// class score is the class's softmax probability minus whether it is
    /// the example's class:
    /// https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_loss_function_and_logistic_regression
    pub(crate) fn loss_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let (k, f) = (self.n_classes, self.n_features);
        let coefficients = k * f;
        let mut grad = vec![0.0; self.dim()];
        let mut loss = 0.0;
        let mut z = vec![0.0; k];
        for (row, &label) in self.rows.iter().zip(self.y) {
            for (c, score) in z.iter_mut().enumerate() {
                let base = c * f;
                // Accumulated from the intercept in column order, as every
                // earlier fit did, so the same input gives the same bits.
                *score = row
                    .iter()
                    .fold(x[coefficients + c], |score, &(column, value)| {
                        score + x[base + column as usize] * value
                    });
            }
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = z.iter().map(|&score| (score - max).exp()).sum();
            let log_sum_exp = max + sum_exp.ln();
            loss += log_sum_exp - z[label];
            for (c, &score) in z.iter().enumerate() {
                let is_label = if c == label { 1.0 } else { 0.0 };
                let delta = (score - log_sum_exp).exp() - is_label;
                grad[coefficients + c] += delta;
                let base = c * f;
                for &(column, value) in row {
                    grad[base + column as usize] += delta * value;
                }
            }
        }
        let weights = &x[..coefficients];
        let squared: f64 = weights.iter().map(|weight| weight * weight).sum();
        for (gradient, weight) in grad[..coefficients].iter_mut().zip(weights) {
            *gradient += self.l2 * weight;
        }
        (loss + self.l2 * squared * L2_HALF, grad)
    }
}

pub(crate) struct Fit {
    pub(crate) coef: Vec<Vec<f64>>,
    pub(crate) intercept: Vec<f64>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}
