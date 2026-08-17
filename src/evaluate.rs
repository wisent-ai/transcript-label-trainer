//! The frozen evaluation split, and the Brama teacher's verdict on it.
//!
//! Three checks live here, and they answer one question — *does the trained
//! model and the evidence used to assess it make sense?*
//!
//! **The frozen split.** Comparing two models over time only means something
//! when both were scored on the same untouched sessions, so the holdout is
//! decided once and written to `$TLT_HOME/models/<name>/eval-split.json` next
//! to the artifacts. Every later run of the same job reads that file back:
//! sessions labeled since then can only join the training side, a session
//! already in the holdout is never trained on, and the file is never rewritten.
//! It is on by default (`eval_split: false` in the job spec turns it off).
//!
//! **The judge.** Accuracy on the holdout says how often the model matched the
//! ground-truth label; it does not say whether the label it chose was
//! defensible for that transcript. So [`evaluate`] additionally sends each
//! holdout session — its text, the model's prediction, the ground truth — to a
//! Brama-routed teacher and asks for one word. A Brama error fails that one
//! session and is counted, exactly like `autolabel`; if no session could be
//! judged at all, the gateway's own error is surfaced verbatim and nothing is
//! written, because a fabricated verdict is worse than no verdict.
//!
//! **The final review.** With `--best`, Brama's `-best` route independently
//! audits both the stored ground-truth label and the first judge's opinion.
//! Any nonsensical or unreviewed record makes the command fail after the
//! evidence has been written to `judge.json`.
//!
//! # The shuffle is deterministic, and it is not Python's
//!
//! The Python original shuffled each class with `random.Random(f"{seed}:{value}")`,
//! whose Mersenne-Twister seeding from a string is a CPython implementation
//! detail that cannot be reproduced here. The scheme used instead, fixed and
//! documented so a first run is reproducible from the seed on any machine and
//! any build:
//!
//! 1. the per-class key is the UTF-8 bytes of `"{seed}:{class}"`, the same
//!    string Python keyed on — so a class that gains labels still does not
//!    reshuffle the others;
//! 2. that string is hashed with SHA-256 and the 32-byte digest becomes the
//!    seed of a ChaCha20 stream (`rand_chacha::ChaCha20Rng::from_seed`);
//! 3. the class members, sorted lexicographically first, are permuted by a
//!    descending Fisher–Yates: for `i` from `len - 1` down to `1`, swap
//!    position `i` with position `next_u64() % (i + 1)`.
//!
//! Fisher–Yates is spelled out here rather than taken from `rand`'s
//! `SliceRandom` so a `rand` upgrade cannot silently pick a different holdout.
//! Splits already on disk are unaffected either way: the file is written once
//! and never rewritten.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::util::{float_repr, json_truthy, now_iso, Error, Result, TrainFailure};
use crate::{brama, jobs, lake, model};

pub const SPLIT_FILE: &str = "eval-split.json";
pub const JUDGE_FILE: &str = "judge.json";

/// The judge answers with exactly one of these.
pub const JUDGE_VALUES: [&str; 2] = ["acceptable", "unacceptable"];
/// Joint audit outcomes from Brama's strongest approved subscription route.
const BEST_REVIEW_VALUES: [&str; 4] = [
    "both-sensible",
    "label-nonsensical",
    "judge-nonsensical",
    "both-nonsensical",
];

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

fn opt_float(value: Option<f64>) -> String {
    value.map(float_repr).unwrap_or_else(|| "None".to_string())
}

fn opt_int(value: Option<i64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "None".to_string())
}

fn number(value: f64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// Python's `round(x, 4)`: half-to-even on the scaled value.
fn round4(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

/// The ChaCha20 stream one class is permuted with. See the module doc.
fn class_rng(seed: i64, class: &str) -> ChaCha20Rng {
    let digest = Sha256::digest(format!("{seed}:{class}").as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    ChaCha20Rng::from_seed(key)
}

/// Descending Fisher–Yates, spelled out so the permutation is ours forever.
fn shuffle(items: &mut [String], rng: &mut ChaCha20Rng) {
    for index in (1..items.len()).rev() {
        let pick = (rng.next_u64() % (index as u64 + 1)) as usize;
        items.swap(index, pick);
    }
}

/// Pick the holdout, stratified per class and reproducible from the seed.
///
/// A class never loses all of its sessions to the holdout, and any class with
/// at least two sessions contributes at least one — otherwise the holdout could
/// not report per-class counts for it. Shuffling is keyed by seed *and* class,
/// so a class that gains labels does not reshuffle the others.
fn choose_holdout(
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
        if members.len() < 2 {
            continue;
        }
        members.sort();
        let count = ((members.len() as f64 * fraction) as usize)
            .max(1)
            .min(members.len() - 1);
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
fn split_json(split: &Split) -> Value {
    let mut block = Map::new();
    block.insert("enabled".to_string(), Value::Bool(split.enabled));
    block.insert(
        "fraction".to_string(),
        split.fraction.map(number).unwrap_or(Value::Null),
    );
    block.insert(
        "seed".to_string(),
        split
            .seed
            .map(|seed| Value::Number(seed.into()))
            .unwrap_or(Value::Null),
    );
    block.insert(
        "created_at".to_string(),
        split
            .created_at
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    block.insert(
        "path".to_string(),
        split.path.clone().map(Value::String).unwrap_or(Value::Null),
    );
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

/// Load or create the frozen holdout for this artifact directory.
///
/// Returns which rows of `sessions` train, which are held out, and the
/// provenance of the file. The file is written only on the first run, and only
/// after the training side is known to clear the minimums, so a run that fails
/// cannot leave a split behind that a later run would inherit.
pub fn resolve_split(
    job: &jobs::Job,
    sessions: &[lake::SessionLabel],
    subject: &str,
) -> Result<Split, TrainFailure> {
    let total = sessions.len();
    let session_ids: Vec<String> = sessions
        .iter()
        .map(|session| session.session_id.clone())
        .collect();
    let values: Vec<String> = sessions
        .iter()
        .map(|session| session.value.clone())
        .collect();

    if !job.eval_split.enabled {
        let mut split = Split {
            enabled: false,
            fraction: None,
            seed: None,
            created_at: None,
            path: None,
            holdout_index: Vec::new(),
            train_index: (0..total).collect(),
            missing_from_selection: 0,
            reused: false,
            frozen: Value::Null,
        };
        split.frozen = split_json(&split);
        return Ok(split);
    }

    let path = split_path(&job.name)?;
    let (holdout_ids, fraction, seed, created_at, reused) = if path.is_file() {
        let frozen = read_split(&job.name)?;
        if frozen.fraction != job.eval_split.fraction || frozen.seed != job.eval_split.seed {
            lake::warn(&format!(
                "eval_split in the spec (fraction={}, seed={}) differs from the \
                 frozen split in {} (fraction={}, seed={}); the frozen file wins \
                 — that is what frozen means",
                opt_float(job.eval_split.fraction),
                opt_int(job.eval_split.seed),
                path.display(),
                opt_float(frozen.fraction),
                opt_int(frozen.seed),
            ));
        }
        (
            frozen.session_ids,
            frozen.fraction,
            frozen.seed,
            frozen.created_at,
            true,
        )
    } else {
        let fraction = job
            .eval_split
            .fraction
            .unwrap_or(jobs::DEFAULT_EVAL_FRACTION);
        let seed = job.eval_split.seed.unwrap_or(jobs::DEFAULT_EVAL_SEED);
        let chosen = choose_holdout(&session_ids, &values, fraction, seed);
        (chosen, Some(fraction), Some(seed), Some(now_iso()), false)
    };

    let holdout: BTreeSet<&str> = holdout_ids.iter().map(String::as_str).collect();
    let mut holdout_index: Vec<usize> = Vec::new();
    let mut train_index: Vec<usize> = Vec::new();
    for (index, session_id) in session_ids.iter().enumerate() {
        if holdout.contains(session_id.as_str()) {
            holdout_index.push(index);
        } else {
            train_index.push(index);
        }
    }

    let train_values: BTreeSet<&str> = train_index
        .iter()
        .map(|index| values[*index].as_str())
        .collect();
    if train_index.len() < model::MIN_LABELED_SESSIONS || train_values.len() < 2 {
        return Err(TrainFailure::NotEnoughData(format!(
            "{subject} has {total} usable labeled session(s), of which {} are held \
             out by the frozen evaluation split (fraction={}, seed={}), leaving {} \
             session(s) across {} distinct value(s) to train on; at least {} \
             sessions and 2 distinct values are required. Add labels with \
             'transcript-lake label add', or set 'eval_split: false' in the job \
             spec to train on every labeled session.",
            holdout_index.len(),
            opt_float(fraction),
            opt_int(seed),
            train_index.len(),
            train_values.len(),
            model::MIN_LABELED_SESSIONS,
        )));
    }

    if !reused {
        let directory = model::aspect_dir(&job.name)?;
        std::fs::create_dir_all(&directory).map_err(Error::from)?;
        let mut record = Map::new();
        record.insert(
            "fraction".to_string(),
            fraction.map(number).unwrap_or(Value::Null),
        );
        record.insert(
            "seed".to_string(),
            seed.map(|seed| Value::Number(seed.into()))
                .unwrap_or(Value::Null),
        );
        record.insert(
            "created_at".to_string(),
            created_at.clone().map(Value::String).unwrap_or(Value::Null),
        );
        record.insert(
            "session_ids".to_string(),
            Value::Array(holdout_ids.iter().cloned().map(Value::String).collect()),
        );
        let body = serde_json::to_string_pretty(&Value::Object(record)).map_err(Error::from)?;
        std::fs::write(&path, body + "\n").map_err(Error::from)?;
    }

    let mut split = Split {
        enabled: true,
        fraction,
        seed,
        created_at,
        path: Some(path.to_string_lossy().into_owned()),
        missing_from_selection: holdout_ids.len() - holdout_index.len(),
        holdout_index,
        train_index,
        reused,
        frozen: Value::Null,
    };
    split.frozen = split_json(&split);
    Ok(split)
}

/// Accuracy, per-class counts and confusion pairs on the frozen holdout.
///
/// This is not the HF backend's `in_training_eval`: that one is a stratified
/// slice of the training side, resplit on every run. This one is the frozen
/// set, identical across runs and across backends.
pub fn holdout_report(gold: &[String], predictions: &[(String, f64)]) -> Value {
    let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
    let mut correct_per_class: BTreeMap<&str, u64> = BTreeMap::new();
    let mut confusion: BTreeMap<(&str, &str), u64> = BTreeMap::new();
    let mut correct = 0u64;
    for (actual, (predicted, _confidence)) in gold.iter().zip(predictions) {
        *counts.entry(actual.as_str()).or_insert(0) += 1;
        correct_per_class.entry(actual.as_str()).or_insert(0);
        if predicted == actual {
            *correct_per_class.entry(actual.as_str()).or_insert(0) += 1;
            correct += 1;
        } else {
            *confusion
                .entry((actual.as_str(), predicted.as_str()))
                .or_insert(0) += 1;
        }
    }
    let total = gold.len();

    let mut ordered: Vec<((&str, &str), u64)> =
        confusion.into_iter().map(|(pair, n)| (pair, n)).collect();
    ordered.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let pairs: Vec<Value> = ordered
        .into_iter()
        .map(|((actual, predicted), n)| {
            let mut pair = Map::new();
            pair.insert("gold".to_string(), Value::String(actual.to_string()));
            pair.insert(
                "predicted".to_string(),
                Value::String(predicted.to_string()),
            );
            pair.insert("n".to_string(), Value::Number(n.into()));
            Value::Object(pair)
        })
        .collect();

    let as_object = |source: BTreeMap<&str, u64>| {
        let mut object = Map::new();
        for (key, value) in source {
            object.insert(key.to_string(), Value::Number(value.into()));
        }
        Value::Object(object)
    };

    let mut report = Map::new();
    report.insert("n_sessions".to_string(), Value::Number(total.into()));
    report.insert(
        "accuracy".to_string(),
        if total > 0 {
            number(round4(correct as f64 / total as f64))
        } else {
            Value::Null
        },
    );
    report.insert("counts".to_string(), as_object(counts));
    report.insert("correct".to_string(), as_object(correct_per_class));
    report.insert("confusion".to_string(), Value::Array(pairs));
    Value::Object(report)
}

// ---------------------------------------------------------------------------
// The Brama teacher's verdict
// ---------------------------------------------------------------------------

/// Ask the teacher whether one prediction is defensible for one session.
pub fn judge_prompt(
    aspect: &str,
    task: Option<&str>,
    gold: &str,
    prediction: &str,
    text: &str,
) -> Vec<brama::Message> {
    let purpose = match task {
        Some(task) if !task.is_empty() => format!("\nWhat the classifier is for: {task}"),
        _ => String::new(),
    };
    vec![
        brama::Message {
            role: "system".to_string(),
            content: format!(
                "You audit a small classifier that assigns aspect labels to \
                 coding-agent session transcripts. Answer with exactly one \
                 word, {} or {}, and nothing else.",
                JUDGE_VALUES[0], JUDGE_VALUES[1]
            ),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}{purpose}\n\
                 Ground-truth label recorded by the evaluator: {gold}\n\
                 Label predicted by the classifier: {prediction}\n\n\
                 Is the predicted label a defensible reading of this transcript \
                 on this aspect? Answer {} if it is (including when it differs \
                 from the ground truth but the session genuinely supports it), \
                 {} if it is not.\n\n\
                 Transcript:\n{text}",
                JUDGE_VALUES[0], JUDGE_VALUES[1]
            ),
        },
    ]
}

fn failure(session_id: &str, error: &str) -> Value {
    let mut record = Map::new();
    record.insert(
        "session_id".to_string(),
        Value::String(session_id.to_string()),
    );
    record.insert("error".to_string(), Value::String(error.to_string()));
    Value::Object(record)
}

/// One verdict per holdout session; a Brama error fails only its session.
fn judge_sessions(
    client: &brama::BramaClient,
    judge_model: &str,
    aspect: &str,
    task: Option<&str>,
    sessions: &[Value],
    texts: &HashMap<String, lake::SessionText>,
) -> (Vec<Value>, Vec<Value>) {
    let mut records: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    for session in sessions {
        let session_id = session
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let gold = session
            .get("gold")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let prediction = session
            .get("prediction")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = texts
            .get(session_id)
            .map(|entry| entry.text.as_str())
            .unwrap_or_default();
        let prompt = judge_prompt(aspect, task, gold, prediction, text);
        match client.chat(judge_model, &prompt) {
            Ok(answer) => match brama::parse_answer(&answer, &JUDGE_VALUES[..]) {
                Some((verdict, _exact)) => {
                    let mut record = session.clone();
                    if let Some(object) = record.as_object_mut() {
                        object.insert("verdict".to_string(), Value::String(verdict));
                    }
                    records.push(record);
                }
                None => failures.push(failure(
                    session_id,
                    &format!(
                        "unparseable judge answer: {}",
                        jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                    ),
                )),
            },
            Err(error) => failures.push(failure(session_id, &error.0)),
        }
    }
    (records, failures)
}

fn best_review_prompt(
    aspect: &str,
    task: Option<&str>,
    gold: &str,
    prediction: &str,
    verdict: &str,
    text: &str,
) -> Vec<brama::Message> {
    let purpose = match task {
        Some(task) if !task.is_empty() => format!("\nWhat the classifier is for: {task}"),
        _ => String::new(),
    };
    vec![
        brama::Message {
            role: "system".to_string(),
            content: format!(
                "You are the final semantic auditor for transcript labels and \
                 another judge's opinion. Answer with exactly one of: {}.",
                BEST_REVIEW_VALUES.join(", ")
            ),
        },
        brama::Message {
            role: "user".to_string(),
            content: format!(
                "Aspect: {aspect}{purpose}\n\
                 Recorded ground-truth label: {gold}\n\
                 Classifier prediction: {prediction}\n\
                 Earlier judge verdict on that prediction: {verdict}\n\n\
                 Decide independently whether (1) the recorded label is a \
                 sensible reading of the transcript for this aspect, and \
                 (2) the earlier judge verdict is sensible given the prediction \
                 and transcript. Answer both-sensible when both are sound, \
                 label-nonsensical when only the label is unsound, \
                 judge-nonsensical when only the earlier verdict is unsound, or \
                 both-nonsensical when neither is sound.\n\n\
                 Transcript:\n{text}"
            ),
        },
    ]
}

fn best_review_sessions(
    client: &brama::BramaClient,
    aspect: &str,
    task: Option<&str>,
    sessions: &mut [Value],
    texts: &HashMap<String, lake::SessionText>,
) -> Vec<Value> {
    let mut failures = Vec::new();
    for session in sessions {
        let session_id = session
            .get("session_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let gold = session
            .get("gold")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let prediction = session
            .get("prediction")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let verdict = session
            .get("verdict")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let text = texts
            .get(&session_id)
            .map(|entry| entry.text.as_str())
            .unwrap_or_default();
        let prompt = best_review_prompt(aspect, task, &gold, &prediction, &verdict, text);
        match client.chat(brama::BEST_MODEL, &prompt) {
            Ok(answer) => match brama::parse_answer(&answer, &BEST_REVIEW_VALUES[..]) {
                Some((review, _exact)) => {
                    if let Some(object) = session.as_object_mut() {
                        object.insert("best_review".to_string(), Value::String(review));
                    }
                }
                None => failures.push(failure(
                    &session_id,
                    &format!(
                        "unparseable final review answer: {}",
                        jobs::py_repr_str(&brama::truncate_chars(&answer, 80))
                    ),
                )),
            },
            Err(error) => failures.push(failure(&session_id, &error.0)),
        }
    }
    failures
}

/// Score the trained model on its frozen holdout and have Brama judge it.
///
/// `name` is a job name or a bare aspect — both are directory names under
/// `$TLT_HOME/models/`. Fails when there is no frozen holdout, when nothing is
/// trained, and when the gateway could not judge a single session.
pub fn evaluate(
    name: &str,
    judge: Option<bool>,
    judge_model: Option<&str>,
    best_review: bool,
) -> Result<Value> {
    let artifact = model::active_artifact(name)?;
    let metrics = &artifact.metrics;
    let aspect = metrics
        .get("aspect")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let empty = Value::Object(Map::new());
    let job_meta = match metrics.get("job") {
        Some(value) if json_truthy(value) => value.clone(),
        _ => empty.clone(),
    };
    let spec_judge = match job_meta.get("judge") {
        Some(value) if json_truthy(value) => value.clone(),
        _ => serde_json::to_value(jobs::default_judge())?,
    };
    let judge_enabled = judge.unwrap_or_else(|| {
        spec_judge
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    });
    let model_id = judge_model
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            spec_judge
                .get("model")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| brama::DEFAULT_MODEL.to_string());
    if best_review && !judge_enabled {
        return Err(Error(
            "--best requires the Brama judge to be enabled".to_string(),
        ));
    }

    let frozen = read_split(name)?;
    let labels = model::labels_for_artifact(metrics)?;
    let by_id: HashMap<&str, &lake::SessionLabel> = labels
        .iter()
        .map(|label| (label.session_id.as_str(), label))
        .collect();
    let holdout_ids: Vec<String> = frozen
        .session_ids
        .iter()
        .filter(|session_id| by_id.contains_key(session_id.as_str()))
        .cloned()
        .collect();
    let missing = frozen.session_ids.len() - holdout_ids.len();
    if holdout_ids.is_empty() {
        return Err(Error(format!(
            "none of the {} frozen holdout session(s) in {} still carry a \
             ground-truth label for aspect '{aspect}'; there is nothing to evaluate",
            frozen.session_ids.len(),
            split_path(name)?.display()
        )));
    }

    let texts = lake::session_texts(&holdout_ids)?;
    let usable: Vec<String> = holdout_ids
        .iter()
        .filter(|session_id| {
            texts
                .get(session_id.as_str())
                .is_some_and(|entry| !entry.text.trim().is_empty())
        })
        .cloned()
        .collect();
    let no_text = holdout_ids.len() - usable.len();
    if no_text > 0 {
        lake::warn(&format!(
            "{no_text} frozen holdout session(s) had no text in the lake"
        ));
    }
    if usable.is_empty() {
        return Err(Error(format!(
            "none of the {} frozen holdout session(s) have text in the lake; \
             there is nothing to evaluate",
            holdout_ids.len()
        )));
    }

    let gold: Vec<String> = usable
        .iter()
        .map(|session_id| by_id[session_id.as_str()].value.clone())
        .collect();
    let holdout_texts: Vec<String> = usable
        .iter()
        .map(|session_id| texts[session_id.as_str()].text.clone())
        .collect();
    let predictions = model::predict(&artifact, &holdout_texts)?;
    let report = holdout_report(&gold, &predictions);

    let sessions: Vec<Value> = usable
        .iter()
        .zip(&gold)
        .zip(&predictions)
        .map(|((session_id, actual), (predicted, confidence))| {
            let mut record = Map::new();
            record.insert("session_id".to_string(), Value::String(session_id.clone()));
            record.insert(
                "runtime".to_string(),
                texts[session_id.as_str()]
                    .runtime
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            record.insert("gold".to_string(), Value::String(actual.clone()));
            record.insert("prediction".to_string(), Value::String(predicted.clone()));
            record.insert("confidence".to_string(), number(round4(*confidence)));
            record.insert("correct".to_string(), Value::Bool(predicted == actual));
            Value::Object(record)
        })
        .collect();

    let mut eval_split = Map::new();
    eval_split.insert(
        "path".to_string(),
        Value::String(split_path(name)?.to_string_lossy().into_owned()),
    );
    eval_split.insert(
        "fraction".to_string(),
        frozen.fraction.map(number).unwrap_or(Value::Null),
    );
    eval_split.insert(
        "seed".to_string(),
        frozen
            .seed
            .map(|seed| Value::Number(seed.into()))
            .unwrap_or(Value::Null),
    );
    eval_split.insert(
        "created_at".to_string(),
        frozen
            .created_at
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    eval_split.insert(
        "frozen_sessions".to_string(),
        Value::Number(frozen.session_ids.len().into()),
    );
    eval_split.insert(
        "missing_ground_truth".to_string(),
        Value::Number(missing.into()),
    );
    eval_split.insert("skipped_no_text".to_string(), Value::Number(no_text.into()));

    let mut result = Map::new();
    result.insert("name".to_string(), Value::String(name.to_string()));
    result.insert("aspect".to_string(), Value::String(aspect.clone()));
    result.insert(
        "backend".to_string(),
        Value::String(artifact.backend.clone()),
    );
    result.insert(
        "model_path".to_string(),
        metrics.get("model_path").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "trained_at".to_string(),
        metrics.get("trained_at").cloned().unwrap_or(Value::Null),
    );
    result.insert("evaluated_at".to_string(), Value::String(now_iso()));
    result.insert("eval_split".to_string(), Value::Object(eval_split));
    result.insert("holdout_evaluation".to_string(), report);

    if !judge_enabled {
        let mut block = Map::new();
        block.insert("enabled".to_string(), Value::Bool(false));
        result.insert("judge".to_string(), Value::Object(block));
        result.insert("sessions".to_string(), Value::Array(sessions));
        return Ok(Value::Object(result));
    }

    let client = brama::BramaClient::from_env()?;
    let task = job_meta.get("task").and_then(Value::as_str);
    let (mut records, failures) =
        judge_sessions(&client, &model_id, &aspect, task, &sessions, &texts);
    if records.is_empty() {
        // No usable provider route: surface the gateway's own words and write
        // nothing. A verdict nobody produced is not a verdict.
        let first = failures
            .first()
            .and_then(|entry| entry.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("(no sessions to judge)")
            .to_string();
        return Err(Error(format!(
            "the Brama judge ({model_id}) could not judge any of the {} holdout \
             session(s); first error: {first}",
            sessions.len()
        )));
    }

    let acceptable = records
        .iter()
        .filter(|record| record.get("verdict").and_then(Value::as_str) == Some(JUDGE_VALUES[0]))
        .count();
    let mut block = Map::new();
    block.insert("enabled".to_string(), Value::Bool(true));
    block.insert("model".to_string(), Value::String(model_id.clone()));
    block.insert("judged".to_string(), Value::Number(records.len().into()));
    block.insert("acceptable".to_string(), Value::Number(acceptable.into()));
    block.insert(
        "unacceptable".to_string(),
        Value::Number((records.len() - acceptable).into()),
    );
    block.insert("failed".to_string(), Value::Number(failures.len().into()));
    block.insert(
        "agreement_rate".to_string(),
        number(round4(acceptable as f64 / records.len() as f64)),
    );
    result.insert("judge".to_string(), Value::Object(block));
    if best_review {
        let review_failures = best_review_sessions(&client, &aspect, task, &mut records, &texts);
        let reviewed = records
            .iter()
            .filter(|record| record.get("best_review").and_then(Value::as_str).is_some())
            .count();
        if reviewed == 0 {
            let first = review_failures
                .first()
                .and_then(|entry| entry.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("(no sessions to review)");
            return Err(Error(format!(
                "the final Brama reviewer ({}) could not review any of the {} \
                 judge record(s); first error: {first}",
                brama::BEST_MODEL,
                records.len()
            )));
        }
        let both_sensible = records
            .iter()
            .filter(|record| {
                record.get("best_review").and_then(Value::as_str) == Some(BEST_REVIEW_VALUES[0])
            })
            .count();
        let label_nonsensical = records
            .iter()
            .filter(|record| {
                matches!(
                    record.get("best_review").and_then(Value::as_str),
                    Some("label-nonsensical" | "both-nonsensical")
                )
            })
            .count();
        let judge_nonsensical = records
            .iter()
            .filter(|record| {
                matches!(
                    record.get("best_review").and_then(Value::as_str),
                    Some("judge-nonsensical" | "both-nonsensical")
                )
            })
            .count();
        let sensible =
            failures.is_empty() && review_failures.is_empty() && both_sensible == records.len();
        result.insert(
            "best_review".to_string(),
            serde_json::json!({
                "enabled": true,
                "model": brama::BEST_MODEL,
                "reviewed": reviewed,
                "both_sensible": both_sensible,
                "label_nonsensical": label_nonsensical,
                "judge_nonsensical": judge_nonsensical,
                "failed": review_failures.len(),
                "sensible": sensible,
            }),
        );
        result.insert(
            "best_review_failures".to_string(),
            Value::Array(review_failures),
        );
    }
    result.insert("sessions".to_string(), Value::Array(records));
    result.insert("failures".to_string(), Value::Array(failures));

    let path = judge_path(name)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&Value::Object(result.clone()))?;
    std::fs::write(&path, body + "\n")?;
    result.insert(
        "judge_path".to_string(),
        Value::String(path.to_string_lossy().into_owned()),
    );
    Ok(Value::Object(result))
}
