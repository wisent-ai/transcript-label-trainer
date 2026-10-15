use super::*;

pub const SPLIT_FILE: &str = "eval-split.json";

pub const JUDGE_FILE: &str = "judge.json";

/// The labels the judge's and the final review's typed choices answer with;
/// what each means is offered beside it (`judge_sessions`).
pub const ACCEPTABLE: &str = "acceptable";
pub(crate) const UNACCEPTABLE: &str = "unacceptable";
pub(crate) const BOTH_SENSIBLE: &str = "both-sensible";
pub(crate) const LABEL_NONSENSICAL: &str = "label-nonsensical";
pub(crate) const JUDGE_NONSENSICAL: &str = "judge-nonsensical";
pub(crate) const BOTH_NONSENSICAL: &str = "both-nonsensical";

/// Where the frozen holdout of one job (or aspect) is persisted.
pub fn split_path(name: &str) -> Result<PathBuf> {
    Ok(model::aspect_dir(name)?.join(SPLIT_FILE))
}

pub fn judge_path(name: &str) -> Result<PathBuf> {
    Ok(model::aspect_dir(name)?.join(JUDGE_FILE))
}

/// A frozen split file as it sits on disk.
#[derive(Debug, Clone)]
pub struct Frozen {
    pub session_ids: Vec<String>,
    pub fraction: Option<f64>,
    pub seed: Option<i64>,
    pub created_at: Option<String>,
}

/// The resolution of one training selection against the frozen split.
#[derive(Debug, Clone)]
pub struct Split {
    pub enabled: bool,
    pub fraction: Option<f64>,
    pub seed: Option<i64>,
    pub created_at: Option<String>,
    pub path: Option<String>,
    /// Indices into the selection that was passed in, row-aligned with the
    /// caller's texts and values.
    pub holdout_index: Vec<usize>,
    pub train_index: Vec<usize>,
    /// Frozen ids that this selection no longer carries.
    pub missing_from_selection: usize,
    pub reused: bool,
    /// The `eval_split` block exactly as it belongs in `metrics.json`.
    pub frozen: Value,
}

/// A float as the Python original printed an optional one: `None` when the
/// split states none.
pub(crate) fn opt_float(value: Option<f64>) -> String {
    match value {
        Some(value) => float_repr(value),
        None => "None".to_string(),
    }
}

/// An integer as the Python original printed an optional one.
pub(crate) fn opt_int(value: Option<i64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "None".to_string(),
    }
}

/// A float as JSON; a value JSON has no number for (NaN, an infinity) is null.
pub(crate) fn number(value: f64) -> Value {
    match serde_json::Number::from_f64(value) {
        Some(number) => Value::Number(number),
        None => Value::Null,
    }
}

/// Python's `round(x, 4)`, the rounding every metrics file on disk carries:
/// half-to-even on the scaled value.
pub(crate) fn round4(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0 // https://docs.python.org/3/library/functions.html#round round(x, 4), as the metrics files on disk were written
}

/// The ChaCha20 stream one class is permuted with: keyed by the SHA-256 of
/// the seed and the class, which is as long as the stream's key.
pub(crate) fn class_rng(seed: i64, class: &str) -> ChaCha20Rng {
    let digest = Sha256::digest(format!("{seed}:{class}").as_bytes());
    let mut key = <ChaCha20Rng as SeedableRng>::Seed::default();
    key.copy_from_slice(&digest);
    ChaCha20Rng::from_seed(key)
}

/// Descending Fisher–Yates, spelled out so the permutation is ours forever.
pub(crate) fn shuffle(items: &mut [String], rng: &mut ChaCha20Rng) {
    let positions = (1..items.len()).rev(); // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm "for i from n−1 down to 1 do"
    for index in positions {
        let pick = (rng.next_u64() % (index as u64 + 1)) as usize; // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm "j ← random integer such that 0 ≤ j ≤ i"
        items.swap(index, pick);
    }
}

/// Pick the holdout, stratified per class and reproducible from the seed.
///
/// A class never loses all of its sessions to the holdout, and any class with
/// at least two sessions contributes at least one — otherwise the holdout could
/// not report per-class counts for it. Shuffling is keyed by seed *and* class,
/// so a class that gains labels does not reshuffle the others.
pub(crate) fn choose_holdout(
    session_ids: &[String],
    values: &[String],
    fraction: f64,
    seed: i64,
) -> Vec<String> {
    let mut by_value: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (session_id, value) in session_ids.iter().zip(values) {
        by_value
            .entry(value.as_str())
            .or_default()
            .push(session_id.clone());
    }
    let mut chosen: Vec<String> = Vec::new();
    for (value, members) in by_value.iter_mut() {
        // A class keeps one session for training and holds out from the rest.
        let Some((_kept, holdable)) = members.split_last() else {
            continue;
        };
        if holdable.is_empty() {
            continue;
        }
        members.sort();
        let count = ((members.len() as f64 * fraction) as usize)
            .max(1) // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/guide/training.md "any class with two or more sessions contributes at least one"
            .min(members.len() - 1); // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/guide/training.md "No class is ever emptied into the holdout"
        let mut shuffled = members.clone();
        shuffle(&mut shuffled, &mut class_rng(seed, value));
        chosen.extend(shuffled.into_iter().take(count));
    }
    chosen.sort();
    chosen
}

/// Read a frozen split file. Fails when it cannot be trusted.
pub fn read_split(name: &str) -> Result<Frozen> {
    let path = split_path(name)?;
    if !path.is_file() {
        return Err(Error(format!(
            "no frozen evaluation split at {}; train '{name}' first, or the run \
             that produced these artifacts had eval_split: false",
            path.display()
        )));
    }
    let parsed: Value = match std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string()))
    {
        Ok(parsed) => parsed,
        Err(detail) => {
            return Err(Error(format!(
                "frozen evaluation split {} is unreadable: {detail}",
                path.display()
            )))
        }
    };
    let session_ids = parsed
        .get("session_ids")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .map(|item| {
                    item.as_str()
                        .filter(|text| !text.is_empty())
                        .map(str::to_string)
                })
                .collect::<Option<Vec<String>>>()
        });
    let Some(session_ids) = session_ids else {
        return Err(Error(format!(
            "frozen evaluation split {} has no usable 'session_ids' list; it is \
             never rewritten automatically — fix or remove it by hand",
            path.display()
        )));
    };
    Ok(Frozen {
        session_ids,
        fraction: parsed.get("fraction").and_then(Value::as_f64),
        seed: parsed.get("seed").and_then(Value::as_i64),
        created_at: parsed
            .get("created_at")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// The part of a split resolution that belongs in `metrics.json`.
pub(crate) fn split_json(split: &Split) -> Value {
    let mut block = Map::new();
    block.insert("enabled".to_string(), Value::Bool(split.enabled));
    block.insert(
        "fraction".to_string(),
        match split.fraction {
            Some(fraction) => number(fraction),
            None => Value::Null,
        },
    );
    block.insert("seed".to_string(), serde_json::json!(split.seed));
    block.insert(
        "created_at".to_string(),
        serde_json::json!(split.created_at),
    );
    block.insert("path".to_string(), serde_json::json!(split.path));
    let frozen_sessions = split.holdout_index.len() + split.missing_from_selection;
    block.insert(
        "frozen_sessions".to_string(),
        Value::Number(frozen_sessions.into()),
    );
    block.insert(
        "holdout_sessions".to_string(),
        Value::Number(split.holdout_index.len().into()),
    );
    block.insert(
        "train_sessions".to_string(),
        Value::Number(split.train_index.len().into()),
    );
    block.insert(
        "missing_from_selection".to_string(),
        Value::Number(split.missing_from_selection.into()),
    );
    block.insert("reused".to_string(), Value::Bool(split.reused));
    Value::Object(block)
}
