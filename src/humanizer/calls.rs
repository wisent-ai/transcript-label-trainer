//! What preparation and audit share: reading JSONL, asking Brama one
//! question for text or for the arguments of one required function call,
//! and fanning rows out over worker threads while reporting progress.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Mutex;
use std::thread;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::brama::{BramaClient, Message};
use crate::util::{Error, Result};

/// Every non-blank line of `path`, parsed as `T`; a row that does not parse
/// is refused quoting it, with the parser's reason.
pub(crate) fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path)
        .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let line =
            line.map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line).map_err(|error| {
            Error(format!(
                "{} holds a row that is not one ({error}): {line}",
                path.display()
            ))
        })?;
        rows.push(row);
    }
    if rows.is_empty() {
        crate::bail!("{} holds no rows", path.display());
    }
    Ok(rows)
}

/// Ask `model` once for the arguments of one required call of the function
/// `name`, whose JSON Schema is `parameters`, and hand them to `accept`. The
/// answer is that object; it is never searched for inside prose, and an
/// answer without the call is the row's error with what the model sent.
pub(crate) fn ask_call<T>(
    client: &BramaClient,
    model: &str,
    system: &str,
    user: String,
    (name, description, parameters): (&str, &str, Value),
    accept: impl Fn(serde_json::Map<String, Value>) -> Result<T>,
) -> Result<T> {
    let messages = [
        Message::new("system", system.to_string()),
        Message::new("user", user),
    ];
    accept(client.call(model, &messages, name, description, parameters)?)
}

/// Ask `model` once and hand the answer to `accept`. No retry count is
/// stated, so none is made: a failed or unacceptable answer is the row's
/// error, counted with its reason, never dropped silently.
pub(crate) fn ask<T>(
    client: &BramaClient,
    model: &str,
    system: &str,
    user: String,
    accept: impl Fn(&str) -> Result<T>,
) -> Result<T> {
    let messages = [
        Message::new("system", system.to_string()),
        Message::new("user", user),
    ];
    let content = client.chat(model, &messages)?;
    if content.is_empty() {
        return Err(Error(format!("{model} returned empty content")));
    }
    accept(&content)
}

/// `work` applied to every row on as many threads as `workers` allows and
/// there are rows, results in row order. `stage` names the progress line
/// printed to stderr as each row finishes.
pub(crate) fn fan_out<T: Sync, R: Send>(
    rows: &[T],
    workers: NonZeroUsize,
    stage: &str,
    work: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let queue = Mutex::new(rows.iter().enumerate());
    let finished = Mutex::new(Vec::with_capacity(rows.len()));
    let results: Mutex<Vec<Option<R>>> = Mutex::new(rows.iter().map(|_| None).collect());
    thread::scope(|scope| {
        for () in std::iter::repeat_n((), workers.get().min(rows.len())) {
            scope.spawn(|| loop {
                let next = queue.lock().expect("humanizer row queue").next();
                let Some((index, row)) = next else {
                    break;
                };
                let result = work(row);
                results.lock().expect("humanizer result lock")[index] = Some(result);
                let mut done = finished.lock().expect("humanizer progress lock");
                done.push(index);
                eprintln!("{stage} {}/{}", done.len(), rows.len());
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
