//! Echo's personal-voice humanizer: the privacy-masked corpus export, the
//! Brama-built inverse style-transfer dataset (`prepare`), and the independent
//! Brama audit of the trained adapter (`audit`).
//!
//! Transcript Lake owns source parsing and masking. The export selects only
//! likely human-authored user turns, deduplicates them, and caps each session
//! so one conversation cannot dominate the fine-tune.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::lake;
use crate::util::{Error, Result};

mod anchors;
mod audit;
mod calls;
mod prepare;
mod prompts;
mod publication;
mod manifest;
mod served;

pub use audit::{audit_outputs, AuditGate};
pub use manifest::manifest;
pub use anchors::LengthRatio;
pub use prepare::{prepare_dataset, HeldOut, PreparationBounds};
pub use publication::{publish_adapter, Publication};
pub(crate) use served::{evaluate_gguf, export_examples, HumanizerEvaluation};

/// What the humanizer corpus takes, every value the caller's to state: at
/// most `limit` targets and at most `max_per_session` from one session, the
/// export refused below `minimum`; a target counts as authored when its
/// trimmed text has `min_chars` to `max_chars` characters, at most
/// `max_lines` lines, at least `min_words` words holding a letter, and at
/// least `min_meaningful_share` of its characters letters, digits or spaces.
pub struct CorpusBounds {
    pub limit: usize,
    pub minimum: usize,
    pub max_per_session: usize,
    pub min_chars: usize,
    pub max_chars: usize,
    pub max_lines: usize,
    pub min_words: usize,
    pub min_meaningful_share: f64,
}

/// Version of the `preparation.json`, `audit.json` and job-output
/// `model-manifest.json` record layouts.
const REPORT_SCHEMA_VERSION: u32 = 1;
const MODEL_CONTRACT: &str = "echo-lukasz-humanizer-v1";

#[derive(Clone, Deserialize, Serialize)]
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

fn likely_authored(value: &str, bounds: &CorpusBounds) -> bool {
    let text = value.trim();
    let chars = text.chars().count();
    if !(bounds.min_chars..=bounds.max_chars).contains(&chars)
        || text.lines().count() > bounds.max_lines
        || text.starts_with('/')
    {
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
    if words < bounds.min_words {
        return false;
    }
    let meaningful = text
        .chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .count();
    meaningful as f64 / chars as f64 >= bounds.min_meaningful_share
}

pub fn export_targets(path: &Path, bounds: &CorpusBounds) -> Result<Value> {
    if bounds.min_chars > bounds.max_chars {
        return Err(Error(format!(
            "--min-target-chars {} exceeds --max-target-chars {}: no target could qualify",
            bounds.min_chars, bounds.max_chars
        )));
    }
    if bounds.minimum > bounds.limit {
        return Err(Error(format!(
            "--min-targets {} exceeds --limit {}: the corpus could never be large enough",
            bounds.minimum, bounds.limit
        )));
    }
    // Every candidate is read in its stable order and the loop stops at the
    // limit, so how many rows the filters reject never needs guessing.
    let sql = r#"
SELECT session_id, runtime, text AS target
FROM events
WHERE event_type = 'user'
  AND text IS NOT NULL
  AND runtime IN ('omp', 'claude', 'codex', 'droid', 'kimi')
ORDER BY hash(session_id || ':' || text)
"#;
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    let mut per_session: HashMap<String, usize> = HashMap::new();
    for value in lake::query(sql)? {
        let session_id = field(&value, "session_id");
        let runtime = field(&value, "runtime");
        let target = field(&value, "target");
        if session_id.is_empty() || runtime.is_empty() || !likely_authored(&target, bounds) {
            continue;
        }
        let identity = normalized(&target);
        if !seen.insert(identity) {
            continue;
        }
        let count = per_session.entry(session_id.clone()).or_default();
        if *count >= bounds.max_per_session {
            continue;
        }
        *count += 1;
        rows.push(TargetRow {
            id: digest(&format!("{runtime}:{session_id}:{target}")),
            session_id,
            runtime,
            target,
        });
        if rows.len() == bounds.limit {
            break;
        }
    }
    if rows.len() < bounds.minimum {
        return Err(Error(format!(
            "humanizer corpus produced only {} clean targets; --min-targets asks for {}",
            rows.len(),
            bounds.minimum
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
        "max_per_session": bounds.max_per_session,
        "limit": bounds.limit,
        "minimum": bounds.minimum,
        "authored": {
            "min_chars": bounds.min_chars,
            "max_chars": bounds.max_chars,
            "max_lines": bounds.max_lines,
            "min_words": bounds.min_words,
            "min_meaningful_share": bounds.min_meaningful_share,
        },
    }))
}
