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
/// Every field it states is read from the evidence beside it; one the
/// evidence does not hold is refused naming the file and the field, never
/// written as null into a manifest that is published.
pub fn manifest(output: &Path, evidence: &Map<String, Value>) -> Result<Value> {
    let publication = read(output, "publication.json")?;
    let metrics = read(output, "metrics.json")?;
    let audit = read(output, "audit.json")?;
    let field = |document: &Value, file: &str, key: &str| -> Result<Value> {
        match document.get(key) {
            Some(value) => Ok(value.clone()),
            None => Err(Error(format!(
                "{} holds no {key}, so the model manifest cannot state it",
                output.join(file).display()
            ))),
        }
    };
    let from_publication = |key: &str| field(&publication, "publication.json", key);
    let from_metrics = |key: &str| field(&metrics, "metrics.json", key);
    let from_audit = |key: &str| field(&audit, "audit.json", key);
    let published = publication.get("qualified").and_then(Value::as_bool) == Some(true);
    let qualified = published && audit.get("passed").and_then(Value::as_bool) == Some(true);
    // A refused publication records the privacy it observed and names no
    // revision, because nothing was published; a qualified one records both.
    let (revision, private) = if published {
        (from_publication("revision")?, from_publication("private")?)
    } else {
        (Value::Null, from_publication("observed_private")?)
    };
    Ok(json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "product": "Echo Łukasz humanizer",
        "contract": MODEL_CONTRACT,
        "format": "vLLM LoRA adapter",
        "base_model": from_metrics("base_model")?,
        "base_revision": from_metrics("base_revision")?,
        "repo_id": from_publication("repo_id")?,
        "revision": revision,
        "private": private,
        "qualified": qualified,
        "publication_qualified": from_publication("qualified")?,
        "audit_passed": from_audit("passed")?,
        "metrics": {
            "base": from_metrics("base")?,
            "student": from_metrics("student")?,
            "target_chrf_gain": from_metrics("target_chrf_gain")?,
            "audit_base": from_audit("base")?,
            "audit_student": from_audit("student")?,
            "voice_match_gain": from_audit("voice_match_gain")?,
            "semantic_fidelity_delta": from_audit("semantic_fidelity_delta")?,
        },
        "evidence": evidence,
    }))
}
