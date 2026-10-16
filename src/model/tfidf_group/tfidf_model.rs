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
            format: MODEL_FILE_FORMAT,
            vectorizer: VectorizerFile {
                analyzer: "word".to_string(),
                lowercase: self.vectorizer.lowercase,
                token_pattern: TOKEN_PATTERN.to_string(),
                ngram_range: [NGRAM_MIN, self.vectorizer.ngram_max],
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
        if file.format != MODEL_FILE_FORMAT {
            crate::bail!(
                "model file has layout {}, and this trainer reads only layout {MODEL_FILE_FORMAT} (docs/guide/models/artifact.md)",
                file.format
            );
        }
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
        let [_, ngram_max] = ngram_range;
        Ok(TfidfModel {
            vectorizer: Vectorizer {
                lowercase,
                ngram_max,
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

/// The `model.json` layout this binary writes and `from_file` reads.
const MODEL_FILE_FORMAT: u32 = 1; // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/guide/models/artifact.md field `format`

/// The shortest n-gram the vectorizer counts: single words, as
/// `count_document` counts every length from one word up to `ngram_max`
/// (scikit-learn's `ngram_range` lower bound).
const NGRAM_MIN: usize = 1; // https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html ngram_range=(1, n)

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
///
/// A fold is scored only when its test side holds a row and its training
/// side holds more than one class. When no fold can be scored there is no
/// accuracy, and the answer is `None`, not a measured zero.
pub(crate) fn cross_val_accuracy(
    texts: &[String],
    values: &[String],
    folds: usize,
    settings: &jobs::TfidfTraining,
) -> Option<f64> {
    let mut by_class: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, value) in values.iter().enumerate() {
        by_class.entry(value.as_str()).or_default().push(i);
    }
    let mut assignment: HashMap<usize, usize> = HashMap::with_capacity(texts.len());
    for members in by_class.values() {
        for (position, &row) in members.iter().enumerate() {
            assignment.insert(row, position % folds);
        }
    }
    let fold_ids: std::collections::BTreeSet<usize> = assignment.values().copied().collect();

    let mut scores = Vec::with_capacity(folds);
    for fold in fold_ids {
        let mut train_texts = Vec::new();
        let mut train_values = Vec::new();
        let mut test_texts = Vec::new();
        let mut test_values = Vec::new();
        for (i, (text, value)) in texts.iter().zip(values).enumerate() {
            if assignment.get(&i) == Some(&fold) {
                test_texts.push(text.clone());
                test_values.push(value.clone());
            } else {
                train_texts.push(text.clone());
                train_values.push(value.clone());
            }
        }
        let mut trained = train_values.iter();
        let one_class = trained
            .next()
            .is_none_or(|first| trained.all(|value| value == first));
        if test_texts.is_empty() || one_class {
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
        return None;
    }
    Some(scores.iter().sum::<f64>() / scores.len() as f64)
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
