//! The humanizer job's `model-manifest.json`: the published adapter's
//! repository and revision, the metrics and audit beside it, and every
//! evidence file's size and SHA-256. `qualified` holds only when the
//! publication recorded a qualified adapter and the independent audit passed.

use std::path::Path;

use serde_json::{json, Map, Value};

use super::{MODEL_CONTRACT, REPORT_SCHEMA_VERSION};
use crate::util::{Error, Result};

fn read(output: &Path, name: &str) -> Result<Value> {
    let path = output.join(name);
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
    serde_json::from_str(&text).map_err(|error| Error(format!("{}: {error}", path.display())))
}

/// The manifest for the job output `output`, whose files `evidence` lists.
pub fn manifest(output: &Path, evidence: &Map<String, Value>) -> Result<Value> {
    let publication = read(output, "publication.json")?;
    let metrics = read(output, "metrics.json")?;
    let audit = read(output, "audit.json")?;
    let field = |document: &Value, key: &str| document.get(key).cloned().unwrap_or(Value::Null);
    let qualified = publication.get("qualified").and_then(Value::as_bool) == Some(true)
        && audit.get("passed").and_then(Value::as_bool) == Some(true);
    Ok(json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "product": "Echo Łukasz humanizer",
        "contract": MODEL_CONTRACT,
        "format": "vLLM LoRA adapter",
        "base_model": field(&metrics, "base_model"),
        "base_revision": field(&metrics, "base_revision"),
        "repo_id": field(&publication, "repo_id"),
        "revision": field(&publication, "revision"),
        "private": field(&publication, "private"),
        "qualified": qualified,
        "publication_qualified": field(&publication, "qualified"),
        "audit_passed": field(&audit, "passed"),
        "metrics": {
            "base": field(&metrics, "base"),
            "student": field(&metrics, "student"),
            "target_chrf_gain": field(&metrics, "target_chrf_gain"),
            "audit_base": field(&audit, "base"),
            "audit_student": field(&audit, "student"),
            "voice_match_gain": field(&audit, "voice_match_gain"),
            "semantic_fidelity_delta": field(&audit, "semantic_fidelity_delta"),
        },
        "evidence": evidence,
    }))
}
