//! What preparation and audit share: reading JSONL, asking Brama as many
//! times as the caller states, pulling the JSON object out of an answer, and
//! fanning rows out over worker threads while reporting progress.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::brama::{BramaClient, Message};
use crate::util::{Error, Result};

/// Every non-blank line of `path`, parsed as `T`; the line number names the
/// row that does not parse.
pub(crate) fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path)
        .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line)
            .map_err(|error| Error(format!("{} line {}: {error}", path.display(), index + 1)))?;
        rows.push(row);
    }
    if rows.is_empty() {
        crate::bail!("{} holds no rows", path.display());
    }
    Ok(rows)
}

/// The JSON object between the first `{` and the last `}` of a model answer.
pub(crate) fn json_object(answer: &str, what: &str) -> Result<serde_json::Map<String, Value>> {
    let start = answer.find('{');
    let end = answer.rfind('}');
    let (Some(start), Some(end)) = (start, end) else {
        crate::bail!("{what} is not JSON: {answer}");
    };
    if end < start {
        crate::bail!("{what} is not JSON: {answer}");
    }
    match serde_json::from_str::<Value>(&answer[start..=end])? {
        Value::Object(object) => Ok(object),
        other => crate::bail!("{what} is not a JSON object: {other}"),
    }
}

/// Ask `model` once per attempt, `attempts` times at most, until `accept`
/// takes the answer. The count is the caller's (`--attempts`); a row that
/// fails every attempt is counted with its last error, never dropped silently.
pub(crate) fn ask<T>(
    client: &BramaClient,
    model: &str,
    attempts: usize,
    system: &str,
    user: String,
    accept: impl Fn(&str) -> Result<T>,
) -> Result<T> {
    let messages = [
        Message::new("system", system.to_string()),
        Message::new("user", user),
    ];
    let mut last = Error(format!("{model} was never asked"));
    for _ in 0..attempts {
        let answer = client.chat(model, &messages);
        let answer = answer.and_then(|content| match content.is_empty() {
            true => Err(Error("Brama returned empty content".to_string())),
            false => Ok(content),
        });
        match answer.and_then(|content| accept(&content)) {
            Ok(value) => return Ok(value),
            Err(error) => last = error,
        }
    }
    Err(last)
}

/// `work` applied to every row on `workers` threads, results in row order.
/// `stage` names the progress line printed to stderr every `every` rows and
/// at the end.
pub(crate) fn fan_out<T: Sync, R: Send>(
    rows: &[T],
    workers: usize,
    stage: &str,
    every: usize,
    work: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..rows.len()).map(|_| None).collect());
    thread::scope(|scope| {
        for _ in 0..workers.clamp(1, rows.len().max(1)) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(row) = rows.get(index) else {
                    break;
                };
                let result = work(row);
                results.lock().expect("humanizer result lock")[index] = Some(result);
                let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                if finished % every == 0 || finished == rows.len() {
                    eprintln!("{stage} {finished}/{}", rows.len());
                }
            });
        }
    });
    results
        .into_inner()
        .expect("humanizer result lock")
        .into_iter()
        .flatten()
        .collect()
}
