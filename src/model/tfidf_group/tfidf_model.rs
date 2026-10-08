use super::*;

impl TfidfModel {
    pub(crate) fn fit(texts: &[String], values: &[String], settings: &jobs::TfidfTraining) -> TfidfModel {
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

        let (vectorizer, rows) = Vectorizer::fit_transform(texts, settings);
        let fit = fit_logistic(&rows, &y, classes.len(), vectorizer.n_features(), settings);
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

    /// The artifact file, recording the settings this model was fitted with.
    pub(crate) fn to_file(&self, settings: &jobs::TfidfTraining) -> ModelFile {
        ModelFile {
            backend: TFIDF_BACKEND.to_string(),
            format: 1,
            vectorizer: VectorizerFile {
                analyzer: "word".to_string(),
                lowercase: self.vectorizer.lowercase,
                token_pattern: TOKEN_PATTERN.to_string(),
                ngram_range: [1, self.vectorizer.ngram_max],
                sublinear_tf: self.vectorizer.sublinear_tf,
                smooth_idf: settings.smooth_idf,
                norm: "l2".to_string(),
                min_df: settings.min_df,
                max_df: settings.max_df,
                vocabulary: self.vectorizer.vocabulary.clone(),
                idf: self.vectorizer.idf.clone(),
            },
            classifier: ClassifierFile {
                kind: "logistic-regression".to_string(),
                multi_class: "multinomial".to_string(),
                c: settings.c,
                max_iter: settings.max_iter,
                tol: settings.tol,
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

/// Mean accuracy over stratified folds, each fold refitting the whole
/// pipeline under the run's settings — vectorizer included — on the other
/// folds, the way `cross_val_score` over a `Pipeline` does. Folds follow each
/// class's own order without shuffling, as scikit-learn's StratifiedKFold
/// does by default (shuffle=False,
/// https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html),
/// so no seed is needed and the same labels always give the same folds.
pub(crate) fn cross_val_accuracy(
    texts: &[String],
    values: &[String],
    folds: usize,
    settings: &jobs::TfidfTraining,
) -> f64 {
    let mut by_class: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, value) in values.iter().enumerate() {
        by_class.entry(value.as_str()).or_default().push(i);
    }
    let mut assignment = vec![0usize; texts.len()];
    for members in by_class.values() {
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
        let model = TfidfModel::fit(&train_texts, &train_values, settings);
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
    // What the classifier was trained on, so inference feeds it the same.
    metrics.insert("session_text".to_string(), json!(SESSION_TEXT_WHOLE));
    metrics.insert("n_sessions".to_string(), json!(n_sessions));
    metrics.insert(
        "classes".to_string(),
        json!(counts.keys().collect::<Vec<_>>()),
    );
    metrics.insert("counts".to_string(), counts_json(counts));
    metrics
}

/// The `session_text` an artifact trained on whole session text records.
pub(crate) const SESSION_TEXT_WHOLE: &str = "whole";

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
