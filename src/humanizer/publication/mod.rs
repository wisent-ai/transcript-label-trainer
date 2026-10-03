use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::MODEL_CONTRACT;
use crate::util::{Error, Result};

mod files;
use files::Signature;
mod hub;
use hub::{Hub, ModelInfo};

pub struct Publication<'a> {
    pub model: &'a Path,
    pub metrics: &'a Path,
    pub audit: &'a Path,
    pub preparation: &'a Path,
    pub repository: &'a str,
}

#[derive(Deserialize)]
struct Qualification<'a> {
    #[serde(borrow)]
    contract: Option<&'a str>,
    passed: Option<bool>,
}

enum Gate {
    Metrics,
    Audit,
}

fn evidence(path: &Path, gate: Gate) -> Result<Signature> {
    let bytes = fs::read(path)
        .map_err(|error| Error(format!("cannot read humanizer evidence {}: {error}", path.display())))?;
    let record: Qualification<'_> = serde_json::from_slice(&bytes)
        .map_err(|error| Error(format!("invalid humanizer evidence {}: {error}", path.display())))?;
    match gate {
        Gate::Metrics if record.contract != Some(MODEL_CONTRACT) => {
            return Err(Error(format!("humanizer metrics carry the wrong contract: {:?}", record.contract)));
        }
        Gate::Audit if record.passed != Some(true) => {
            return Err(Error(format!("refusing an unqualified humanizer: audit passed={:?}", record.passed)));
        }
        _ => {}
    }
    Ok(Signature { size: bytes.len() as u64, sha256: hex::encode(Sha256::digest(&bytes)) })
}

fn refusal(repository: &str, info: &ModelInfo) -> Option<Value> {
    (!info.private).then(|| json!({
        "qualified": false,
        "error": "repository_not_private",
        "operation": "model_info",
        "repo_id": repository,
        "observed_private": info.private,
        "observed_revision": info.sha,
    }))
}

pub fn publish_adapter(request: Publication<'_>) -> Result<Value> {
    let repository = request.repository.trim();
    let hub = Hub::new()?;
    let exists = match hub.info(repository, None, false)? {
        Some(info) => {
            if let Some(report) = refusal(repository, &info) {
                return Ok(report);
            }
            true
        }
        None => false,
    };
    let metrics = evidence(request.metrics, Gate::Metrics)?;
    let audit = evidence(request.audit, Gate::Audit)?;
    if !request.model.is_dir() {
        return Err(Error(format!("humanizer adapter directory is missing: {}", request.model.display())));
    }
    let mut signatures = BTreeMap::from([
        ("evaluation/metrics.json", metrics),
        ("evaluation/audit.json", audit),
        ("evaluation/preparation.json", Signature::read(request.preparation)?),
    ]);
    for name in ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"] {
        signatures.insert(name, Signature::read(&request.model.join(name))?);
    }
    if !exists {
        hub.create_private(repository)?;
    }

    // The vendor CLI returns a mutable file URL, not its commit id. Give each
    // publication its own branch so readback cannot name another writer's head.
    let branch = format!("humanizer-publication-{}", hex::encode(rand::random::<[u8; 16]>()));
    for (source, destination) in [
        (request.model, ""),
        (request.metrics, "evaluation/metrics.json"),
        (request.audit, "evaluation/audit.json"),
        (request.preparation, "evaluation/preparation.json"),
    ] {
        let info = hub.info(repository, None, false)?
            .ok_or_else(|| Error(format!("Hugging Face repository is unavailable before upload: {repository}")))?;
        if let Some(report) = refusal(repository, &info) {
            return Ok(report);
        }
        hub.upload(repository, &branch, source, destination)
            .map_err(|error| Error(format!("publication {repository}@{branch}, path {destination:?}: {error}")))?;
    }
    let info = hub.info(repository, Some(&branch), true)?
        .ok_or_else(|| Error(format!("published Hugging Face branch is unavailable: {repository}@{branch}")))?;
    if let Some(report) = refusal(repository, &info) {
        return Ok(report);
    }
    let revision = info.sha.filter(|value| !value.is_empty())
        .ok_or_else(|| Error("Hugging Face model_info returned no immutable revision".into()))?;
    let mut files = BTreeMap::new();
    for file in info.siblings {
        if let Some(signature) = signatures.get(file.rfilename.as_str()) {
            hub.verify(repository, &revision, signature, &file)?;
        }
        let size = file.size.ok_or_else(|| Error(format!(
            "Hugging Face model_info returned no size for {}", file.rfilename
        )))?;
        files.insert(file.rfilename, size);
    }
    for required in signatures.keys() {
        if !files.contains_key(*required) {
            return Err(Error(format!("published humanizer revision is missing {required}")));
        }
    }
    Ok(json!({
        "schema_version": 1,
        "contract": MODEL_CONTRACT,
        "repo_id": repository,
        "revision": revision,
        "branch": branch,
        "private": info.private,
        "files": files,
        "metrics_sha256": signatures["evaluation/metrics.json"].sha256,
        "audit_sha256": signatures["evaluation/audit.json"].sha256,
        "verified_files": signatures,
        "qualified": true,
    }))
}
