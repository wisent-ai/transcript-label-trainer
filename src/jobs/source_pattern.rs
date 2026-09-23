use super::*;

/// The lake labeler's source provenance grammar, as the operator sees it in
/// the error message. Matched by [`source_matches`], not by a regex engine.
pub const SOURCE_PATTERN: &str = "^(manual|human|model|brama)(:[A-Za-z0-9._/-]+)?$";

pub(crate) const SOURCE_PREFIXES: [&str; 4] = ["manual", "human", "model", "brama"];

/// Job names become artifact directory names under `<training root>/models/`.
pub const NAME_PATTERN: &str = "^[a-z0-9][a-z0-9_-]*$";

/// The one reserved model name: the existing sklearn backend. Anything else is
/// a HuggingFace model id and selects the HF backend.
pub const SKLEARN_MODEL: &str = "tfidf-logreg";

// The frozen evaluation split is on by default: model comparisons over time
// have to run on the same untouched sessions, so the holdout is decided once
// and persisted next to the artifacts. The seed is fixed so a first run on the
// same labels always picks the same sessions.
pub const DEFAULT_EVAL_FRACTION: f64 = 0.2;

pub const DEFAULT_EVAL_SEED: i64 = 20260808;

/// A fraction above this would starve training rather than measure it.
pub const MAX_EVAL_FRACTION: f64 = 0.5;

pub(crate) const TOP_LEVEL_KEYS: [&str; 7] = [
    "name",
    "task",
    "evaluator",
    "model",
    "scope",
    "eval_split",
    "judge",
];

pub(crate) const SCOPE_KEYS: [&str; 5] = ["aspect", "runtimes", "since", "values", "min_text_chars"];

pub(crate) const EVAL_SPLIT_KEYS: [&str; 2] = ["fraction", "seed"];

pub(crate) const JUDGE_KEYS: [&str; 1] = ["model"];

/// The frozen holdout section of a validated spec. Serialized verbatim into
/// `metrics.json["job"]["eval_split"]` and `job.yaml`, so the field order and
/// the nulls-when-disabled shape are part of the on-disk format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSplit {
    pub enabled: bool,
    pub fraction: Option<f64>,
    pub seed: Option<i64>,
}

/// The Brama teacher section of a validated spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judge {
    pub enabled: bool,
    pub model: Option<String>,
}

/// The training-data selection. `runtimes`/`values` are `Option` rather than a
/// possibly-empty `Vec` because the artifacts record `null` for "unset" and an
/// empty list is rejected by validation, so the two can never be confused.
#[derive(Debug, Clone)]
pub struct Scope {
    pub aspect: String,
    pub runtimes: Option<Vec<String>>,
    /// Normalized ISO-8601 with offset, e.g. `2026-07-01T00:00:00+00:00`.
    pub since: Option<String>,
    pub values: Option<Vec<String>>,
    pub min_text_chars: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Job {
    pub name: String,
    pub task: String,
    pub evaluator: String,
    pub model: String,
    pub scope: Scope,
    pub eval_split: EvalSplit,
    pub judge: Judge,
}

/// The frozen split every run gets unless the spec says `false`.
pub fn default_eval_split() -> EvalSplit {
    EvalSplit {
        enabled: true,
        fraction: Some(DEFAULT_EVAL_FRACTION),
        seed: Some(DEFAULT_EVAL_SEED),
    }
}

/// The Brama teacher verdict every run gets unless the spec says `false`.
pub fn default_judge() -> Judge {
    Judge {
        enabled: true,
        model: Some(brama::DEFAULT_MODEL.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Python-shaped rendering, so the error sentences are the ones the docs quote
// ---------------------------------------------------------------------------

/// `repr()` of a Python string: single quotes unless that would need escaping
/// and double quotes would not.
pub(crate) fn py_repr_str(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr()` of a value that came out of the YAML loader.
pub(crate) fn py_repr(value: &Yaml) -> String {
    match value {
        Yaml::Null => "None".to_string(),
        Yaml::Bool(true) => "True".to_string(),
        Yaml::Bool(false) => "False".to_string(),
        Yaml::Number(number) => {
            if let Some(signed) = number.as_i64() {
                signed.to_string()
            } else if let Some(unsigned) = number.as_u64() {
                unsigned.to_string()
            } else {
                float_repr(number.as_f64().unwrap_or(f64::NAN))
            }
        }
        Yaml::String(text) => py_repr_str(text),
        Yaml::Sequence(items) => {
            let rendered: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        Yaml::Mapping(mapping) => {
            let rendered: Vec<String> = mapping
                .iter()
                .map(|(key, item)| format!("{}: {}", py_repr(key), py_repr(item)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        Yaml::Tagged(tagged) => py_repr(&tagged.value),
    }
}

// ---------------------------------------------------------------------------
// Grammar checks that stand in for the two compiled patterns
// ---------------------------------------------------------------------------

/// `^[a-z0-9][a-z0-9_-]*$`
pub fn name_matches(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    characters.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// `^(manual|human|model|brama)(:[A-Za-z0-9._/-]+)?$`
pub fn source_matches(value: &str) -> bool {
    for prefix in SOURCE_PREFIXES {
        let Some(rest) = value.strip_prefix(prefix) else {
            continue;
        };
        if rest.is_empty() {
            return true;
        }
        if let Some(detail) = rest.strip_prefix(':') {
            if !detail.is_empty()
                && detail
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
            {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Field validation
// ---------------------------------------------------------------------------

/// Lookup by string key without relying on `Mapping`'s indexing trait, which
/// changed shape across serde_yaml 0.9 point releases.
pub(crate) fn get_raw<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a Yaml> {
    mapping
        .iter()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

/// As [`get_raw`], but an explicit `null` reads as absent — Python's
/// `mapping.get(key)` cannot tell those apart either.
pub(crate) fn get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a Yaml> {
    get_raw(mapping, key).filter(|value| !value.is_null())
}

pub(crate) fn key_name(key: &Yaml) -> String {
    match key {
        Yaml::String(text) => text.clone(),
        other => py_repr(other),
    }
}

pub(crate) fn unknown_keys(mapping: &serde_yaml::Mapping, allowed: &[&str]) -> Vec<String> {
    let mut unknown: Vec<String> = mapping
        .keys()
        .map(key_name)
        .filter(|key| !allowed.contains(&key.as_str()))
        .collect();
    unknown.sort();
    unknown
}

pub(crate) fn require_string(mapping: &serde_yaml::Mapping, key: &str) -> Result<String> {
    match get(mapping, key).and_then(Yaml::as_str) {
        Some(text) if !text.trim().is_empty() => Ok(text.trim().to_string()),
        _ => bail!("'{key}' must be a non-empty string"),
    }
}

pub(crate) fn string_list(scope: &serde_yaml::Mapping, key: &str) -> Result<Option<Vec<String>>> {
    let Some(value) = get(scope, key) else {
        return Ok(None);
    };
    let Yaml::Sequence(items) = value else {
        bail!("scope.{key} must be a non-empty list of strings")
    };
    if items.is_empty() {
        bail!("scope.{key} must be a non-empty list of strings")
    }
    let mut collected = Vec::with_capacity(items.len());
    for item in items {
        match item.as_str() {
            Some(text) if !text.trim().is_empty() => collected.push(text.trim().to_string()),
            _ => bail!("scope.{key} must be a non-empty list of strings"),
        }
    }
    Ok(Some(collected))
}
