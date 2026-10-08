//! The settings a training run states, for either backend. tfidf-logreg
//! states how session text becomes TF-IDF rows, how hard the multinomial
//! logistic regression is regularised, how the L-BFGS solver searches and
//! stops, and how cross-validation shuffles; a HuggingFace fine-tune states
//! its optimisation. None has a default: a job's `training` section states
//! each key, and `train` takes the same keys as flags spelled with dashes,
//! validated by the same code.

use super::*;

/// The tfidf-logreg keys.
pub(crate) const TFIDF_KEYS: &[&str] = &[
    "ngram_max",
    "lowercase",
    "sublinear_tf",
    "smooth_idf",
    "min_df",
    "max_df",
    "c",
    "max_iter",
    "tol",
    "lbfgs_memory",
    "armijo_c1",
    "max_backtracks",
];

/// The HuggingFace fine-tune keys.
pub(crate) const HF_KEYS: &[&str] = &[
    "epochs",
    "batch_size",
    "learning_rate",
    "max_length",
    "seed",
    "weight_decay",
    "max_grad_norm",
    "in_training_eval_share",
];

/// One tfidf-logreg run's settings, recorded under `hyperparameters`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfidfTraining {
    /// Longest word n-gram counted.
    pub ngram_max: usize,
    pub lowercase: bool,
    /// Term frequency as one plus its logarithm instead of the raw count.
    pub sublinear_tf: bool,
    /// Inverse document frequency as if one more document held every term.
    pub smooth_idf: bool,
    /// Terms in fewer documents than this are dropped.
    pub min_df: f64,
    /// Terms in more than this share of documents are dropped.
    pub max_df: f64,
    /// Inverse strength of the L2 penalty on the coefficients.
    pub c: f64,
    pub max_iter: usize,
    /// The solver stops once no gradient component exceeds this.
    pub tol: f64,
    /// Correction pairs the L-BFGS solver keeps.
    pub lbfgs_memory: usize,
    /// Share of the predicted decrease a line-search step must achieve.
    pub armijo_c1: f64,
    /// Line-search candidates tried before a step is abandoned.
    pub max_backtracks: usize,
}

/// One HuggingFace fine-tune's settings, recorded under `hyperparameters`.
/// `seed` drives the head initialisation, the in-training slice and the
/// shuffle; `in_training_eval_share` is the share of the training side
/// sliced off to watch the loss, resplit every run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfTraining {
    pub epochs: f64,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_length: usize,
    pub seed: u64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub in_training_eval_share: f64,
}

/// The two backends' stated settings; every run states one.
#[derive(Debug, Clone)]
pub enum Training {
    Tfidf(TfidfTraining),
    Hf(HfTraining),
}

/// A share strictly between none and all: its sign is positive (zero has
/// none), and it lies below that sign, which is one.
pub(crate) fn strict_share(value: f64) -> bool {
    value.is_normal() && value.is_sign_positive() && value < value.signum()
}

/// The `train` flag that states `key`.
pub(crate) fn flag_name(key: &str) -> String {
    format!("--{}", key.replace('_', "-"))
}

fn backend_keys(model: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match model == SKLEARN_MODEL {
        true => (TFIDF_KEYS, HF_KEYS),
        false => (HF_KEYS, TFIDF_KEYS),
    }
}

/// Validate a job's `training` section for `model`: a mapping of every key,
/// or the name of a preset (`training: scikit-learn`).
pub(crate) fn training(raw: &serde_yaml::Mapping, model: &str) -> Result<Training> {
    match get(raw, "training") {
        Some(Yaml::Mapping(mapping)) => settings(mapping, model),
        Some(Yaml::String(name)) => super::presets::preset(name, model),
        _ => bail!(
            "'training' is required for model {}: a mapping with {}, or a preset ({})",
            py_repr_str(model),
            backend_keys(model).0.join(", "),
            super::PRESETS.join(", ")
        ),
    }
}

/// The settings `train` was given as flags: `--training PRESET`, or each key
/// of the model's backend read from its `--dashed-name` flag as a YAML scalar,
/// then validated exactly as a job's `training` section is. A flag of the
/// other backend is refused, and so is a key flag beside a preset: each value
/// has one source.
pub fn training_from_flags<'a>(model: &str, text: impl Fn(&str) -> Option<&'a str>) -> Result<Training> {
    let (keys, other) = backend_keys(model);
    if let Some(stray) = other.iter().map(|key| flag_name(key)).find(|flag| text(flag.as_str()).is_some()) {
        bail!("{stray} does not apply to model {}", py_repr_str(model))
    }
    if let Some(name) = text("--training") {
        if let Some(beside) = keys.iter().map(|key| flag_name(key)).find(|flag| text(flag.as_str()).is_some()) {
            bail!("{beside} cannot be stated beside --training {name}: the preset states every setting")
        }
        return super::presets::preset(name, model);
    }
    let mut mapping = serde_yaml::Mapping::new();
    for key in keys {
        let flag = flag_name(key);
        if let Some(value) = text(flag.as_str()) {
            let parsed: Yaml = serde_yaml::from_str(value)
                .map_err(|error| crate::util::Error(format!("{flag} {value:?} is not a value: {error}")))?;
            mapping.insert(Yaml::String((*key).to_string()), parsed);
        }
    }
    settings(&mapping, model)
}

/// Reads one section's keys, refusing a missing or invalid one by its key
/// and its `train` flag.
struct Section<'a> {
    mapping: &'a serde_yaml::Mapping,
    model: &'a str,
}

impl Section<'_> {
    fn value(&self, key: &str) -> Result<&Yaml> {
        get(self.mapping, key).ok_or_else(|| {
            crate::util::Error(format!(
                "training.{key} ({}) is required: {} assumes no value for it",
                flag_name(key),
                self.model
            ))
        })
    }

    fn refuse<T>(&self, key: &str, wanted: &str) -> Result<T> {
        let shown = self.value(key).map(py_repr).unwrap_or_default();
        bail!("training.{key} ({}) must be {wanted}, got {shown}", flag_name(key))
    }

    fn flag(&self, key: &str) -> Result<bool> {
        match self.value(key)? {
            Yaml::Bool(value) => Ok(*value),
            _ => self.refuse(key, "true or false"),
        }
    }

    fn seed(&self, key: &str) -> Result<u64> {
        match self.value(key)?.as_u64() {
            Some(value) => Ok(value),
            None => self.refuse(key, "a non-negative whole number"),
        }
    }

    fn whole(&self, key: &str) -> Result<usize> {
        let value = self.value(key)?.as_u64().and_then(|value| usize::try_from(value).ok());
        match value.and_then(std::num::NonZeroUsize::new) {
            Some(value) => Ok(value.get()),
            None => self.refuse(key, "a positive whole number"),
        }
    }

    fn number(&self, key: &str, accept: impl Fn(f64) -> bool, wanted: &str) -> Result<f64> {
        match self.value(key)?.as_f64() {
            Some(value) if accept(value) => Ok(value),
            _ => self.refuse(key, wanted),
        }
    }

    fn positive(&self, key: &str) -> Result<f64> {
        self.number(key, |value| value.is_normal() && value.is_sign_positive(), "a positive number")
    }

    fn share(&self, key: &str) -> Result<f64> {
        self.number(key, strict_share, "a share above none and below all")
    }
}

fn settings(mapping: &serde_yaml::Mapping, model: &str) -> Result<Training> {
    let unknown = unknown_keys(mapping, backend_keys(model).0);
    if !unknown.is_empty() {
        bail!("unknown training field(s) for model {}: {}", py_repr_str(model), unknown.join(", "))
    }
    let section = Section { mapping, model };
    if model == SKLEARN_MODEL {
        let all = |value: f64| value.is_normal() && value.is_sign_positive() && value <= value.signum();
        return Ok(Training::Tfidf(TfidfTraining {
            ngram_max: section.whole("ngram_max")?,
            lowercase: section.flag("lowercase")?,
            sublinear_tf: section.flag("sublinear_tf")?,
            smooth_idf: section.flag("smooth_idf")?,
            min_df: section.positive("min_df")?,
            max_df: section.number("max_df", all, "a share of documents above none, at most all")?,
            c: section.positive("c")?,
            max_iter: section.whole("max_iter")?,
            tol: section.positive("tol")?,
            lbfgs_memory: section.whole("lbfgs_memory")?,
            armijo_c1: section.share("armijo_c1")?,
            max_backtracks: section.whole("max_backtracks")?,
        }));
    }
    Ok(Training::Hf(HfTraining {
        epochs: section.positive("epochs")?,
        batch_size: section.whole("batch_size")?,
        learning_rate: section.positive("learning_rate")?,
        max_length: section.whole("max_length")?,
        seed: section.seed("seed")?,
        weight_decay: section.number(
            "weight_decay",
            |value| value.is_finite() && !value.is_sign_negative(),
            "zero or more",
        )?,
        max_grad_norm: section.positive("max_grad_norm")?,
        in_training_eval_share: section.share("in_training_eval_share")?,
    }))
}
