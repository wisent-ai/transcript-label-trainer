//! Declarative training jobs: load and validate a YAML job spec.
//!
//! A job spec captures the four things an operator declares per training run:
//! WHO evaluated the transcripts (evaluator — the exact label-store source that
//! counts as ground truth), WHICH model to train (model), the SCOPE of training
//! data (scope), and the TASK (free text stored with the artifacts).
//!
//! Two more sections govern how the run is judged, and both are ON unless the
//! spec turns them off: `eval_split` freezes a holdout of labeled sessions that
//! training never sees, and `judge` has a Brama-routed teacher rule on whether
//! the trained model's holdout predictions are acceptable.
//!
//! Every field is validated here; invalid specs fail with clear errors and no
//! silent defaults.
use std::path::Path;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, SecondsFormat};
use serde::{Deserialize, Serialize};
use serde_yaml::Value as Yaml;

use crate::bail;
use crate::brama;
use crate::util::{float_repr, Result};

/// The lake labeler's source provenance grammar, as the operator sees it in
/// the error message. Matched by [`source_matches`], not by a regex engine.
pub const SOURCE_PATTERN: &str = "^(manual|human|model|brama)(:[A-Za-z0-9._/-]+)?$";
const SOURCE_PREFIXES: [&str; 4] = ["manual", "human", "model", "brama"];

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

const TOP_LEVEL_KEYS: [&str; 7] = [
    "name",
    "task",
    "evaluator",
    "model",
    "scope",
    "eval_split",
    "judge",
];
const SCOPE_KEYS: [&str; 5] = ["aspect", "runtimes", "since", "values", "min_text_chars"];
const EVAL_SPLIT_KEYS: [&str; 2] = ["fraction", "seed"];
const JUDGE_KEYS: [&str; 1] = ["model"];

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
fn py_repr(value: &Yaml) -> String {
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
fn get_raw<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a Yaml> {
    mapping
        .iter()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

/// As [`get_raw`], but an explicit `null` reads as absent — Python's
/// `mapping.get(key)` cannot tell those apart either.
fn get<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a Yaml> {
    get_raw(mapping, key).filter(|value| !value.is_null())
}

fn key_name(key: &Yaml) -> String {
    match key {
        Yaml::String(text) => text.clone(),
        other => py_repr(other),
    }
}

fn unknown_keys(mapping: &serde_yaml::Mapping, allowed: &[&str]) -> Vec<String> {
    let mut unknown: Vec<String> = mapping
        .keys()
        .map(key_name)
        .filter(|key| !allowed.contains(&key.as_str()))
        .collect();
    unknown.sort();
    unknown
}

fn require_string(mapping: &serde_yaml::Mapping, key: &str) -> Result<String> {
    match get(mapping, key).and_then(Yaml::as_str) {
        Some(text) if !text.trim().is_empty() => Ok(text.trim().to_string()),
        _ => bail!("'{key}' must be a non-empty string"),
    }
}

fn string_list(scope: &serde_yaml::Mapping, key: &str) -> Result<Option<Vec<String>>> {
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

/// Validate the eval_split section. Absent means the default, on.
fn eval_split(raw: &serde_yaml::Mapping) -> Result<EvalSplit> {
    let value = get_raw(raw, "eval_split");
    let value = match value {
        None | Some(Yaml::Null) => return Ok(default_eval_split()),
        Some(Yaml::Bool(true)) => return Ok(default_eval_split()),
        Some(Yaml::Bool(false)) => {
            return Ok(EvalSplit {
                enabled: false,
                fraction: None,
                seed: None,
            })
        }
        Some(other) => other,
    };
    let Yaml::Mapping(mapping) = value else {
        bail!(
            "'eval_split' must be a mapping with 'fraction' and/or 'seed', \
             true for the defaults, or false to train on every labeled session"
        )
    };
    let unknown = unknown_keys(mapping, &EVAL_SPLIT_KEYS);
    if !unknown.is_empty() {
        bail!("unknown eval_split field(s): {}", unknown.join(", "))
    }

    let fraction = match get(mapping, "fraction") {
        None => DEFAULT_EVAL_FRACTION,
        Some(Yaml::Number(number)) => {
            let fraction = number.as_f64().unwrap_or(f64::NAN);
            if !(fraction > 0.0 && fraction <= MAX_EVAL_FRACTION) {
                bail!(
                    "eval_split.fraction must be greater than 0 and at most {}, got {}",
                    float_repr(MAX_EVAL_FRACTION),
                    py_repr(&Yaml::Number(number.clone()))
                )
            }
            fraction
        }
        Some(other) => {
            bail!(
                "eval_split.fraction must be a number, got {}",
                py_repr(other)
            )
        }
    };

    let seed = match get(mapping, "seed") {
        None => DEFAULT_EVAL_SEED,
        Some(Yaml::Number(number)) if number.is_i64() || number.is_u64() => {
            match number.as_i64().filter(|seed| *seed >= 0) {
                Some(seed) => seed,
                None => bail!(
                    "eval_split.seed must be a non-negative integer, got {}",
                    py_repr(&Yaml::Number(number.clone()))
                ),
            }
        }
        Some(other) => {
            bail!(
                "eval_split.seed must be a non-negative integer, got {}",
                py_repr(other)
            )
        }
    };

    Ok(EvalSplit {
        enabled: true,
        fraction: Some(fraction),
        seed: Some(seed),
    })
}

/// Validate the judge section. Absent means the default teacher, on.
fn judge(raw: &serde_yaml::Mapping) -> Result<Judge> {
    let value = get_raw(raw, "judge");
    let value = match value {
        None | Some(Yaml::Null) => return Ok(default_judge()),
        Some(Yaml::Bool(true)) => return Ok(default_judge()),
        Some(Yaml::Bool(false)) => {
            return Ok(Judge {
                enabled: false,
                model: None,
            })
        }
        Some(other) => other,
    };
    let Yaml::Mapping(mapping) = value else {
        bail!(
            "'judge' must be a mapping with 'model', true for the default \
             teacher ({}), or false to skip the verdict",
            brama::DEFAULT_MODEL
        )
    };
    let unknown = unknown_keys(mapping, &JUDGE_KEYS);
    if !unknown.is_empty() {
        bail!("unknown judge field(s): {}", unknown.join(", "))
    }
    let model = match get(mapping, "model") {
        None => brama::DEFAULT_MODEL.to_string(),
        Some(value) => match value.as_str() {
            Some(text) if !text.trim().is_empty() => text.trim().to_string(),
            _ => bail!("judge.model must be a non-empty Brama-routed model id"),
        },
    };
    Ok(Judge {
        enabled: true,
        model: Some(model),
    })
}

/// Python's `datetime.fromisoformat`, restricted to the shapes an operator
/// writes in a job spec, normalized to a UTC-aware ISO string.
fn parse_since(raw: &str) -> Option<String> {
    let text = raw.trim().replace('Z', "+00:00");
    if let Ok(aware) = DateTime::<FixedOffset>::parse_from_rfc3339(&text) {
        return Some(aware.to_rfc3339_opts(SecondsFormat::Secs, false));
    }
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(&text, format) {
            return Some(naive.and_utc().to_rfc3339_opts(SecondsFormat::Secs, false));
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(&text, "%Y-%m-%d") {
        let naive = date.and_hms_opt(0, 0, 0)?;
        return Some(naive.and_utc().to_rfc3339_opts(SecondsFormat::Secs, false));
    }
    None
}

/// Load and fully validate a job spec file.
pub fn load(path: &str) -> Result<Job> {
    let spec_path = Path::new(path);
    if !spec_path.is_file() {
        bail!("job file not found: {path}")
    }
    let contents = std::fs::read_to_string(spec_path)
        .map_err(|error| crate::util::Error(format!("job file {path} is unreadable: {error}")))?;
    let parsed: Yaml = match serde_yaml::from_str(&contents) {
        Ok(parsed) => parsed,
        Err(error) => bail!("job file {path} is not valid YAML: {error}"),
    };
    let Yaml::Mapping(raw) = parsed else {
        bail!("job file {path} must contain a YAML mapping at the top level")
    };

    let unknown = unknown_keys(&raw, &TOP_LEVEL_KEYS);
    if !unknown.is_empty() {
        bail!("unknown job field(s): {}", unknown.join(", "))
    }

    let name = require_string(&raw, "name")?;
    if !name_matches(&name) {
        bail!(
            "'name' {} must match {NAME_PATTERN} (it becomes the artifact directory name)",
            py_repr_str(&name)
        )
    }

    let task = require_string(&raw, "task")?;

    let evaluator = require_string(&raw, "evaluator")?;
    if !source_matches(&evaluator) {
        bail!(
            "'evaluator' {} must match the label-store source grammar {SOURCE_PATTERN} \
             (manual, human, model or brama, with an optional :detail suffix)",
            py_repr_str(&evaluator)
        )
    }

    let model = require_string(&raw, "model")?;

    let Some(Yaml::Mapping(scope)) = get(&raw, "scope") else {
        bail!("'scope' must be a mapping with at least 'aspect'")
    };
    let unknown_scope = unknown_keys(scope, &SCOPE_KEYS);
    if !unknown_scope.is_empty() {
        bail!("unknown scope field(s): {}", unknown_scope.join(", "))
    }

    let aspect = match get(scope, "aspect").and_then(Yaml::as_str) {
        Some(text) if !text.trim().is_empty() => text.trim().to_string(),
        _ => bail!("scope.aspect is required and must be a non-empty string"),
    };

    let since = match get(scope, "since") {
        None => None,
        Some(Yaml::String(text)) => match parse_since(text) {
            Some(normalized) => Some(normalized),
            None => bail!(
                "scope.since {} is not an ISO date, e.g. \"2026-07-01\"",
                py_repr_str(text)
            ),
        },
        Some(other) => {
            bail!(
                "scope.since must be an ISO date string, got {}",
                py_repr(other)
            )
        }
    };

    let min_text_chars = match get(scope, "min_text_chars") {
        None => None,
        Some(Yaml::Number(number)) if number.is_i64() || number.is_u64() => {
            match number.as_i64().filter(|value| *value >= 0) {
                Some(value) => Some(value as u64),
                None => bail!("scope.min_text_chars must be a non-negative integer"),
            }
        }
        Some(_) => bail!("scope.min_text_chars must be a non-negative integer"),
    };

    Ok(Job {
        name,
        task,
        evaluator,
        model,
        scope: Scope {
            aspect,
            runtimes: string_list(scope, "runtimes")?,
            since,
            values: string_list(scope, "values")?,
            min_text_chars,
        },
        eval_split: eval_split(&raw)?,
        judge: judge(&raw)?,
    })
}
