//! Named training presets: a whole backend's settings taken from one vendor's
//! documentation, chosen by name (`training: scikit-learn` in a job,
//! `train --training scikit-learn`, or the Train panel's fill button) instead
//! of typed key by key. Every value cites the document it comes from; a preset
//! is a stated choice, never a fallback for a missing key.

use super::*;

/// The tfidf-logreg backend reproduces scikit-learn's TfidfVectorizer and
/// LogisticRegression(solver="lbfgs"); this preset is their documented
/// defaults, with the solver details SciPy's L-BFGS-B documents.
pub(crate) const SCIKIT_LEARN: &str = "scikit-learn";

/// Every preset name, for refusals and the GUI.
pub(crate) const PRESETS: &[&str] = &[SCIKIT_LEARN];

fn scikit_learn() -> TfidfTraining {
    TfidfTraining {
        // TfidfVectorizer ngram_range=(1, 1), lowercase=True, sublinear_tf=False,
        // smooth_idf=True, min_df=1, max_df=1.0:
        // https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        ngram_max: 1,
        lowercase: true,
        sublinear_tf: false,
        smooth_idf: true,
        // https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        min_df: 1.0,
        // https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        max_df: 1.0,
        // LogisticRegression C=1.0, max_iter=100, tol=1e-4:
        // https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        c: 1.0,
        // https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        max_iter: 100,
        // https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        tol: 1e-4,
        // SciPy L-BFGS-B maxcor=10, the correction pairs kept:
        // https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        lbfgs_memory: 10,
        // SciPy scalar_search_armijo c1=1e-4:
        // https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py
        armijo_c1: 1e-4,
        // LogisticRegression hands L-BFGS-B "maxls": 50 line-search steps:
        // https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py
        max_backtracks: 50,
    }
}

/// The settings preset `name` states for `model`, or a refusal naming the
/// presets that exist and the backend they apply to.
pub(crate) fn preset(name: &str, model: &str) -> Result<Training> {
    if model != SKLEARN_MODEL {
        bail!(
            "training preset {} applies to model {}, not {}; a HuggingFace model states {}",
            py_repr_str(name),
            py_repr_str(SKLEARN_MODEL),
            py_repr_str(model),
            HF_KEYS.join(", ")
        )
    }
    match name {
        SCIKIT_LEARN => Ok(Training::Tfidf(scikit_learn())),
        _ => bail!(
            "unknown training preset {}: the presets are {}",
            py_repr_str(name),
            PRESETS.join(", ")
        ),
    }
}

/// Each preset's settings per backend, as the GUI shows them for its fill
/// button: `{tfidf-logreg: {scikit-learn: {key: value}}}`.
pub(crate) fn presets_json() -> Result<serde_json::Value> {
    let mut by_name = serde_json::Map::new();
    for name in PRESETS {
        if let Training::Tfidf(settings) = preset(name, SKLEARN_MODEL)? {
            by_name.insert((*name).to_string(), serde_json::to_value(settings)?);
        }
    }
    Ok(serde_json::json!({ (SKLEARN_MODEL): by_name }))
}
