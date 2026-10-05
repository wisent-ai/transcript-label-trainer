use super::*;

/// Validate the eval_split section. It is required: a mapping with both
/// `fraction` and `seed`, or `false` to train on every labeled session. The
/// holdout is the job's to state; nothing here chooses its size or seed.
pub(crate) fn eval_split(raw: &serde_yaml::Mapping) -> Result<EvalSplit> {
    let value = match get_raw(raw, "eval_split") {
        Some(Yaml::Bool(false)) => {
            return Ok(EvalSplit {
                enabled: false,
                fraction: None,
                seed: None,
            })
        }
        Some(Yaml::Mapping(mapping)) => mapping,
        _ => bail!(
            "'eval_split' is required: a mapping with 'fraction' and 'seed', or false \
             to train on every labeled session; the job states its holdout, \
             transcript-label-trainer chooses none"
        ),
    };
    let unknown = unknown_keys(value, &EVAL_SPLIT_KEYS);
    if !unknown.is_empty() {
        bail!("unknown eval_split field(s): {}", unknown.join(", "))
    }

    let fraction = match get(value, "fraction") {
        None => bail!("eval_split.fraction is required (a share between 0 and 1)"),
        Some(Yaml::Number(number)) => {
            let fraction = number.as_f64().unwrap_or(f64::NAN);
            // Both sides must hold sessions, so 0 and 1 are the only bounds.
            if !(fraction > 0.0 && fraction < 1.0) {
                bail!(
                    "eval_split.fraction must be greater than 0 and less than 1, got {}",
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

    let seed = match get(value, "seed") {
        None => bail!("eval_split.seed is required (a non-negative integer)"),
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

/// Validate the training section: required with all four settings for a
/// HuggingFace model, refused for the TF-IDF backend, which takes none.
pub(crate) fn training(raw: &serde_yaml::Mapping, model: &str) -> Result<Option<HfTraining>> {
    let section = get(raw, "training");
    if model == SKLEARN_MODEL {
        if section.is_some() {
            bail!("'training' applies to a HuggingFace model; {SKLEARN_MODEL} takes no training settings")
        }
        return Ok(None);
    }
    let Some(Yaml::Mapping(mapping)) = section else {
        bail!(
            "'training' is required for HuggingFace model {}: a mapping with {}",
            py_repr_str(model),
            TRAINING_KEYS.join(", ")
        )
    };
    let unknown = unknown_keys(mapping, &TRAINING_KEYS);
    if !unknown.is_empty() {
        bail!("unknown training field(s): {}", unknown.join(", "))
    }
    let positive = |key: &str| -> Result<f64> {
        match get(mapping, key).and_then(Yaml::as_f64) {
            Some(value) if value > 0.0 && value.is_finite() => Ok(value),
            Some(value) => bail!("training.{key} must be greater than 0, got {value}"),
            None => bail!("training.{key} is required and must be a number"),
        }
    };
    let whole = |key: &str| -> Result<usize> {
        match get(mapping, key).and_then(Yaml::as_u64) {
            Some(value) if value >= 1 => Ok(value as usize),
            _ => bail!("training.{key} is required and must be a whole number of at least 1"),
        }
    };
    Ok(Some(HfTraining {
        epochs: positive("epochs")?,
        batch_size: whole("batch_size")?,
        learning_rate: positive("learning_rate")?,
        max_length: whole("max_length")?,
    }))
}

/// Validate the judge section. Absent means the default teacher, on.
pub(crate) fn judge(raw: &serde_yaml::Mapping) -> Result<Judge> {
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
pub(crate) fn parse_since(raw: &str) -> Option<String> {
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

    let hf_training = training(&raw, &model)?;
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
        training: hf_training,
    })
}
