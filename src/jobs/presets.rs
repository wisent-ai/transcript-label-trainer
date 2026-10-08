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

/// The `ster tune sft` preset the model jobs accept as `--ster-options trl`:
/// the documented defaults of Hugging Face's own LoRA fine-tuning stack (PEFT
/// for the adapter, TRL's SFTConfig over Transformers' TrainingArguments for
/// the run), each cited where it is read.
pub(crate) const TRL: &str = "trl";

// PEFT LoraConfig r=8: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
const LORA_RANK: usize = 8;
// PEFT LoraConfig lora_alpha=8: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py
const LORA_ALPHA: usize = 8;
// Transformers TrainingArguments num_train_epochs=3.0: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
const EPOCHS: usize = 3;
// TRL SFTConfig learning_rate=2e-5: https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
const LEARNING_RATE: f64 = 2e-5;
// Transformers TrainingArguments gradient_accumulation_steps=1: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
const ACCUMULATION: usize = 1;
// TRL SFTConfig max_length=1024: https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
const MAX_SEQUENCE: usize = 1024;
// Transformers TrainingArguments per_device_train_batch_size=8: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
const BATCH_SIZE: usize = 8;
// Transformers TrainingArguments seed=42: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
const SEED: u64 = 42;

/// The `ster tune sft` options a preset name stands for, or None when the
/// text is not a preset name (the job then passes it to Ster as written).
pub(crate) fn ster_preset(name: &str) -> Option<String> {
    (name == TRL).then(|| {
        format!(
            "--rank {LORA_RANK} --alpha {LORA_ALPHA} --epochs {EPOCHS} --learning-rate {LEARNING_RATE} \
             --accumulation {ACCUMULATION} --max-sequence {MAX_SEQUENCE} --batch-size {BATCH_SIZE} --seed {SEED}"
        )
    })
}

/// The holdout preset (`eval_split: scikit-learn`, `train --eval-split
/// scikit-learn`): scikit-learn's documented test share, and a seed taken
/// from the split's own name so every run of the same job or aspect freezes
/// the same sessions without anyone choosing a number.
pub(crate) const EVAL_SPLIT_PRESET: &str = SCIKIT_LEARN;

// train_test_split test_size: "If train_size is also None, it will be set to 0.25",
// https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
const TEST_SIZE: f64 = 0.25;

/// The holdout `eval_split: scikit-learn` states for the split named `name`.
pub(crate) fn eval_split_preset(name: &str) -> EvalSplit {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(name.as_bytes());
    let leading = digest.first_chunk().map(|bytes| i64::from_be_bytes(*bytes));
    EvalSplit { enabled: true, fraction: Some(TEST_SIZE), seed: leading.map(i64::saturating_abs) }
}
