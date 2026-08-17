//! Privacy-masked corpus export for Echo's personal-voice humanizer.
//!
//! Transcript Lake owns source parsing and masking. This module selects only
//! likely human-authored user turns, deduplicates them, and caps each session so
//! one conversation cannot dominate the fine-tune.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::lake;
use crate::util::{Error, Result};

const MAX_PER_SESSION: usize = 6;
const FETCH_MULTIPLIER: usize = 12;

#[derive(Clone, Serialize)]
struct TargetRow {
    id: String,
    session_id: String,
    runtime: String,
    target: String,
}

fn field(row: &Value, name: &str) -> String {
    row.get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn normalized(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn digest(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

fn likely_authored(value: &str) -> bool {
    let text = value.trim();
    let chars = text.chars().count();
    if !(20..=2_000).contains(&chars) || text.lines().count() > 8 || text.starts_with('/') {
        return false;
    }
    let lower = text.to_lowercase();
    let rejected_prefixes = [
        "<system",
        "kontynuuj dokładnie przerwaną pracę",
        "you are a strict gate",
        "on your first completion attempt",
        "use the read tool",
        "use bash to run exactly",
        "api validation error:",
    ];
    let rejected_fragments = [
        "skip to content",
        "begin private key",
        "authorization: bearer",
        "[masked:",
        "claude-code-hint",
        "<system-reminder>",
        "<system-notice>",
        "github_pat_",
        "sk-ant-",
        "sk-proj-",
    ];
    if rejected_prefixes
        .iter()
        .any(|prefix| lower.starts_with(prefix))
        || rejected_fragments
            .iter()
            .any(|fragment| lower.contains(fragment))
    {
        return false;
    }
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .count();
    if words < 4 {
        return false;
    }
    let meaningful = text
        .chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .count();
    meaningful * 100 / chars.max(1) >= 65
}

pub fn export_targets(path: &Path, limit: usize) -> Result<Value> {
    if limit < 1_000 {
        return Err(Error(
            "humanizer corpus requires at least 1000 targets".to_string(),
        ));
    }
    let fetch = limit.saturating_mul(FETCH_MULTIPLIER);
    let sql = format!(
        r#"
SELECT session_id, runtime, text AS target
FROM events
WHERE event_type = 'user'
  AND text IS NOT NULL
  AND runtime IN ('omp', 'claude', 'codex', 'droid', 'kimi')
  AND length(text) BETWEEN 20 AND 2000
ORDER BY hash(session_id || ':' || text)
LIMIT {fetch}
"#
    );
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let mut per_session: HashMap<String, usize> = HashMap::new();
    for value in lake::query(&sql)? {
        let session_id = field(&value, "session_id");
        let runtime = field(&value, "runtime");
        let target = field(&value, "target");
        if session_id.is_empty() || runtime.is_empty() || !likely_authored(&target) {
            continue;
        }
        let identity = normalized(&target);
        if !seen.insert(identity) {
            continue;
        }
        let count = per_session.entry(session_id.clone()).or_default();
        if *count >= MAX_PER_SESSION {
            continue;
        }
        *count += 1;
        rows.push(TargetRow {
            id: digest(&format!("{runtime}:{session_id}:{target}")),
            session_id,
            runtime,
            target,
        });
        if rows.len() == limit {
            break;
        }
    }
    if rows.len() < 1_000 {
        return Err(Error(format!(
            "humanizer corpus produced only {} clean targets; need at least 1000",
            rows.len()
        )));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(File::create(path)?);
    for row in &rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(json!({
        "targets": rows.len(),
        "sessions": per_session.len(),
        "path": path,
        "sha256": hex::encode(Sha256::digest(std::fs::read(path)?)),
        "source": "transcript-lake:masked-user-events",
        "max_per_session": MAX_PER_SESSION,
    }))
}
