//! `model-manifest`: the `model-manifest.json` a training job writes into its
//! Stado output beside the evidence, which `release-publish` reads and checks
//! file by file before anything is published.
//!
//! It replaces the three inline Python programs the job scripts ran with the
//! training virtualenv's interpreter. What it records is measured here: each
//! evidence file's size and SHA-256, the ordered model parts and the
//! assembled artifact they rebuild, and `qualified`, read from the
//! independent gate the model declares. A missing or unreadable input is a
//! refusal naming the file, never a manifest with the field left out.

use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};

use super::release::read_json;
use crate::hub::Signature;
use crate::util::{Error, Result};

/// The file the manifest is written to, and never counted as its own evidence.
pub(crate) const MANIFEST: &str = "model-manifest.json";
const JUDGE: &str = "final-judge.json";
/// The base checkpoint both GGUF fine-tunes start from.
const BASE_MODEL: &str = "Qwen/Qwen3-4B";
const BASE_REVISION: &str = "1cfa9a7208912126459214e8b04321603b3df60c";

/// A fine-tune whose job output carries a manifest.
#[derive(Clone, Copy)]
pub(crate) enum ManifestModel {
    /// Jeden's goal model: a GGUF shipped through the release channel.
    Goal,
    /// Oko's goal-lifecycle model: a GGUF served from a GPU host.
    Lifecycle,
    /// Echo's humanizer: a LoRA adapter in its own private repository.
    Humanizer,
}

impl ManifestModel {
    pub(crate) const EVERY: &'static [ManifestModel] = &[Self::Goal, Self::Lifecycle, Self::Humanizer];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Goal => "goal",
            Self::Lifecycle => "lifecycle",
            Self::Humanizer => "humanizer",
        }
    }

    /// Every model's name, as the refusals and the help list them.
    pub(crate) fn names() -> String {
        Self::EVERY.iter().map(|model| model.name()).collect::<Vec<_>>().join(", ")
    }

    /// The model `name` names, refused with every model that takes a manifest.
    pub(crate) fn named(name: &str) -> Result<Self> {
        Self::EVERY
            .iter()
            .copied()
            .find(|model| model.name() == name)
            .ok_or_else(|| Error(format!("--model must be one of {}, not {name:?}", Self::names())))
    }
}

/// Write `output/model-manifest.json` for `model` and answer the manifest.
/// `artifact` is the assembled GGUF the job split into `<name>.part-*` parts;
/// the humanizer is a LoRA adapter published to its own repository and takes
/// none.
pub(crate) fn write_manifest(model: ManifestModel, output: &Path, artifact: Option<&Path>) -> Result<Value> {
    let manifest = match (model, artifact) {
        (ManifestModel::Goal, Some(artifact)) => goal(output, artifact)?,
        (ManifestModel::Lifecycle, Some(artifact)) => lifecycle(output, artifact)?,
        (ManifestModel::Humanizer, None) => crate::humanizer::manifest(output, &evidence(output, false)?)?,
        (ManifestModel::Goal | ManifestModel::Lifecycle, None) => {
            return Err(Error(format!(
                "the {} manifest describes a GGUF artifact: name it with --artifact",
                model.name()
            )))
        }
        (ManifestModel::Humanizer, Some(artifact)) => {
            return Err(Error(format!(
                "the humanizer is a LoRA adapter published to its private repository; --artifact {} names a \
                 file its manifest does not carry",
                artifact.display()
            )))
        }
    };
    let path = output.join(MANIFEST);
    fs::write(&path, serde_json::to_string_pretty(&manifest)? + "\n")
        .map_err(|error| Error(format!("cannot write {}: {error}", path.display())))?;
    // The GGUF jobs end on their judge's own exit status; the humanizer job
    // has no later step, so its gate is enforced here, after the manifest is
    // staged for the record.
    if matches!(model, ManifestModel::Humanizer) && manifest["qualified"].as_bool() != Some(true) {
        return Err(Error(format!(
            "the humanizer adapter did not pass its publication gate: {} records publication.json qualified \
             {} and audit.json passed {}",
            path.display(),
            manifest["publication_qualified"],
            manifest["audit_passed"]
        )));
    }
    Ok(manifest)
}

/// Every file directly in `output` except the manifest, by name, with its
/// size and SHA-256. `parts_too` keeps the model parts in the listing, as the
/// GGUF manifests have always listed them.
pub(crate) fn evidence(output: &Path, parts_too: bool) -> Result<Map<String, Value>> {
    let entries = fs::read_dir(output)
        .map_err(|error| Error(format!("cannot list the job output {}: {error}", output.display())))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| Error(format!("cannot list {}: {error}", output.display())))?;
        let is_file = entry
            .file_type()
            .map_err(|error| Error(format!("cannot read {}: {error}", entry.path().display())))?
            .is_file();
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_file && name != MANIFEST && (parts_too || !name.contains(".part-")) {
            names.push(name);
        }
    }
    names.sort();
    let mut files = Map::new();
    for name in names {
        let signature = Signature::read(&output.join(&name))?;
        files.insert(name, json!({ "bytes": signature.size, "sha256": signature.sha256 }));
    }
    Ok(files)
}

/// The ordered parts of `artifact` in `output` and the artifact they rebuild.
fn transport(output: &Path, artifact: &Path, files: &Map<String, Value>) -> Result<Value> {
    let name = file_name(artifact)?;
    let prefix = format!("{name}.part-");
    let parts: Vec<&String> = files.keys().filter(|file| file.starts_with(&prefix)).collect();
    if parts.is_empty() {
        return Err(Error(format!(
            "{} holds no {prefix}* parts of {}; split the artifact into the output before writing its manifest",
            output.display(),
            artifact.display()
        )));
    }
    let whole = Signature::read(artifact)?;
    Ok(json!({
        "kind": "ordered-parts",
        "parts": parts,
        "assembled_bytes": whole.size,
        "assembled_sha256": whole.sha256,
    }))
}

fn file_name(artifact: &Path) -> Result<String> {
    artifact
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| Error(format!("--artifact {} names no file", artifact.display())))
}

/// The judge's verdict: qualified only when it says `passed: true`.
fn judged(output: &Path) -> Result<(bool, Value)> {
    let path = output.join(JUDGE);
    let judge = read_json(&path)?;
    let passed = judge
        .get("passed")
        .and_then(Value::as_bool)
        .ok_or_else(|| Error(format!("{} carries no boolean passed verdict", path.display())))?;
    Ok((passed, judge.get("review_model").cloned().unwrap_or(Value::Null)))
}

fn goal(output: &Path, artifact: &Path) -> Result<Value> {
    let (qualified, review_model) = judged(output)?;
    let served = read_json(&output.join("metrics-gguf.json"))?;
    let trained = read_json(&output.join("metrics.json"))?;
    let files = evidence(output, true)?;
    Ok(json!({
        "product": "Jeden goal model",
        "format": "GGUF",
        "default_artifact": file_name(artifact)?,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "required_quality_gate": JUDGE,
        "evaluation_surface": "served Q4_K_M GGUF through Jeden's goal chat, decoding constrained to <goal/> or one <goal> line",
        "qualified": qualified,
        "review_model": review_model,
        "metrics": served,
        "training_metrics": trained,
        "transport": transport(output, artifact, &files)?,
        "files": files,
    }))
}

/// The lifecycle manifest gates on the independent judge alone, like the
/// goal model: the served model's measured rates are recorded beside it, and
/// the per-rate floors the Python wrote in (nobody stated them) are gone.
fn lifecycle(output: &Path, artifact: &Path) -> Result<Value> {
    let (qualified, review_model) = judged(output)?;
    let served = read_json(&output.join("metrics-gguf.json"))?;
    let trained = read_json(&output.join("metrics.json"))?;
    let files = evidence(output, true)?;
    Ok(json!({
        "product": "Oko goal lifecycle model",
        "contract": "oko-goal-lifecycle-v1",
        "format": "GGUF",
        "default_artifact": file_name(artifact)?,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "required_quality_gate": JUDGE,
        "evaluation_surface": "served Q4_K_M GGUF through Oko's loopback chat contract, decoding constrained to lifecycle-output-schema.json",
        "qualified": qualified,
        "review_model": review_model,
        "metrics": served,
        "training_metrics": trained,
        "transport": transport(output, artifact, &files)?,
        "files": files,
    }))
}
