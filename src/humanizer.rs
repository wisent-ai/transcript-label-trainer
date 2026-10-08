//! Echo's personal-voice humanizer: the privacy-masked corpus export, the
//! Brama-built inverse style-transfer dataset (`prepare`), and the independent
//! Brama audit of the trained adapter (`audit`).
//!
//! Transcript Lake owns source parsing and masking. The export takes every
//! distinct user turn that holds a word; whether a turn is a person's own
//! typed words (and not pasted output or injected harness text) is the
//! independent Brama review's judgement in `prepare`, not a size or keyword rule.

use std::collections::HashSet;
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

pub use audit::audit_outputs;
pub use manifest::manifest;
pub use prepare::prepare_dataset;
pub use publication::{publish_adapter, Publication};
pub(crate) use served::{evaluate_gguf, export_examples, HumanizerEvaluation};

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

pub fn export_targets(path: &Path) -> Result<Value> {
    // Every candidate is read in its stable order.
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
    let mut sessions: HashSet<String> = HashSet::new();
    for value in lake::query(sql)? {
        let session_id = field(&value, "session_id");
        let runtime = field(&value, "runtime");
        let target = field(&value, "target");
        if session_id.is_empty() || runtime.is_empty() || !target.split_whitespace().any(|word| word.chars().any(char::is_alphabetic)) {
            continue;
        }
        let identity = normalized(&target);
        if !seen.insert(identity) {
            continue;
        }
        sessions.insert(session_id.clone());
        rows.push(TargetRow {
            id: digest(&format!("{runtime}:{session_id}:{target}")),
            session_id,
            runtime,
            target,
        });
    }
    if rows.is_empty() {
        return Err(Error("Transcript Lake holds no user turn with a word in it; the humanizer corpus would be empty".to_string()));
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
        "sessions": sessions.len(),
        "path": path,
        "sha256": hex::encode(Sha256::digest(std::fs::read(path)?)),
        "source": "transcript-lake:masked-user-events",
        "authorship": "judged per pair by the independent review in humanizer-prepare",
    }))
}
