use super::*;

impl TfidfModel {
    pub(crate) fn fit(texts: &[String], values: &[String]) -> TfidfModel {
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
    pub(crate) fn predict(&self, texts: &[String]) -> Vec<(String, f64)> {
        self.vectorizer
            .transform(texts)
            .iter()
            .map(|row| {
                let (best, probability) = predict_row(&self.coef, &self.intercept, row);
                (self.classes[best].clone(), probability)
            })
            .collect()
    }

    pub(crate) fn to_file(&self) -> ModelFile {
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

    pub(crate) fn from_file(file: ModelFile) -> Result<TfidfModel> {
        let VectorizerFile {
            lowercase,
            ngram_range,
            sublinear_tf,
            vocabulary,
            idf,
            ..
        } = file.vectorizer;
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
pub(crate) struct Splitmix(pub(crate) u64);

impl Splitmix {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

pub(crate) const CV_SEED: u64 = 0;

/// Mean accuracy over stratified folds, each fold refitting the whole
/// pipeline — vectorizer included — on the other folds, the way
/// `cross_val_score` over a `Pipeline` does. Shuffling is keyed per class so
/// a class that gains members does not reshuffle the others.
pub(crate) fn cross_val_accuracy(texts: &[String], values: &[String], folds: usize) -> f64 {
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

pub(crate) fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

pub(crate) fn base_metrics(
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
    metrics.insert(
        "trainer_version".to_string(),
        json!(env!("CARGO_PKG_VERSION")),
    );
    metrics.insert("model".to_string(), json!(model_desc));
    metrics.insert("n_sessions".to_string(), json!(n_sessions));
    metrics.insert(
        "classes".to_string(),
        json!(counts.keys().collect::<Vec<_>>()),
    );
    metrics.insert("counts".to_string(), counts_json(counts));
    metrics
}

pub(crate) fn counts_json(counts: &BTreeMap<String, usize>) -> Value {
    let mut object = Map::new();
    for (value, count) in counts {
        object.insert(value.clone(), json!(count));
    }
    Value::Object(object)
}

pub(crate) fn write_pretty(path: &Path, value: &Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}
