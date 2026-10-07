use super::*;

/// Term counts for one document: unigrams plus, when `ngram_max` is 2, the
/// space-joined adjacent pairs sklearn's `_word_ngrams` produces.
pub(crate) fn count_document(doc: &str, lowercase: bool, ngram_max: usize) -> HashMap<String, usize> {
    let prepared = if lowercase {
        doc.to_lowercase()
    } else {
        doc.to_string()
    };
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
    pub(crate) fn fit_transform(docs: &[String], settings: &jobs::TfidfTraining) -> (Self, Vec<SparseRow>) {
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

        let high = settings.max_df * n_docs;
        let mut vocabulary: Vec<String> = df
            .iter()
            .filter(|(_, &count)| count as f64 >= settings.min_df && count as f64 <= high)
            .map(|(term, _)| (*term).to_string())
            .collect();
        vocabulary.sort_unstable();

        let mut index = HashMap::with_capacity(vocabulary.len());
        let mut idf = Vec::with_capacity(vocabulary.len());
        for (column, term) in vocabulary.iter().enumerate() {
            index.insert(term.clone(), column as u32);
            let document_frequency = *df.get(term.as_str()).unwrap_or(&0) as f64;
            idf.push(if settings.smooth_idf {
                ((1.0 + n_docs) / (1.0 + document_frequency)).ln() + 1.0
            } else {
                (n_docs / document_frequency).ln() + 1.0
            });
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
    pub(crate) fn row(&self, counts: &HashMap<String, usize>) -> SparseRow {
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
        self.n_classes * (self.n_features + 1)
    }

    pub(crate) fn loss_grad(&self, x: &[f64]) -> (f64, Vec<f64>) {
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
                grad[at] += self.l2 * x[at];
            }
        }
        (loss + self.l2 * squared / (self.l2.signum() + self.l2.signum()), grad)
    }
}

pub(crate) struct Fit {
    pub(crate) coef: Vec<Vec<f64>>,
    pub(crate) intercept: Vec<f64>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}
