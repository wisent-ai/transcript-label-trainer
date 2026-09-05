//! Adoption of the trainer's existing canonical dataset-bundle format.
//!
//! Transcript Lake still owns live labels. An adopted bundle is an immutable,
//! self-contained training input under the trainer placement root; training
//! reads the selected bundle through the same `lake::load_labels` boundary used
//! by pinned Stado jobs.
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::placement::resolve_placement;
use crate::util::{now_iso, Error, Result};

const BUNDLE_SCHEMA: u32 = 1;
const REGISTRY_SCHEMA: u32 = 1;
const CORPORA_DIR: &str = "corpora";
const REGISTRY_FILE: &str = "registry.json";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusBundle {
    schema_version: u32,
    aspect: String,
    labels: Vec<CorpusRow>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusRow {
    session_id: String,
    value: String,
    source: String,
    ts: String,
    runtime: Option<String>,
    text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AdoptedCorpus {
    pub id: String,
    pub sha256: String,
    pub aspect: String,
    pub records: usize,
    #[serde(rename = "bundlePath")]
    pub bundle_path: PathBuf,
    #[serde(rename = "sourcePath")]
    pub source_path: PathBuf,
    #[serde(rename = "adoptedAt")]
    pub adopted_at: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct CorpusRegistry {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    #[serde(rename = "selectedCorpusId")]
    selected_corpus_id: String,
    corpora: Vec<AdoptedCorpus>,
}

pub fn adopt(path: &Path) -> Result<Value> {
    let source_path = fs::canonicalize(path).map_err(|error| {
        Error(format!("could not resolve corpus {}: {error}", path.display()))
    })?;
    if !source_path.is_file() {
        return Err(Error(format!(
            "corpus is not a regular file: {}",
            source_path.display()
        )));
    }
    let raw = fs::read(&source_path)?;
    let bundle: CorpusBundle = serde_json::from_slice(&raw).map_err(|error| {
        Error(format!(
            "invalid dataset bundle {}: {error}; no corpus state was changed",
            source_path.display()
        ))
    })?;
    validate_bundle(&bundle, &source_path)?;
    let mut canonical = serde_json::to_vec_pretty(&bundle)?;
    canonical.push(b'\n');
    let sha256 = format!("{:x}", Sha256::digest(&canonical));
    let id = format!("dataset-bundle:{sha256}");
    let root = corpora_root();
    let registry_path = root.join(REGISTRY_FILE);
    let mut registry = read_registry(&registry_path)?.unwrap_or(CorpusRegistry {
        schema_version: REGISTRY_SCHEMA,
        selected_corpus_id: id.clone(),
        corpora: Vec::new(),
    });
    let bundle_path = root.join(format!("{sha256}.json"));
    let already_present = bundle_path.exists();
    if already_present {
        if fs::read(&bundle_path)? != canonical {
            return Err(Error(format!(
                "corpus identity conflict at {}; no corpus state was changed",
                bundle_path.display()
            )));
        }
    } else {
        durable_create(&bundle_path, &canonical)?;
    }
    let adopted = match registry.corpora.iter().find(|entry| entry.id == id) {
        Some(existing) => existing.clone(),
        None => {
            let entry = AdoptedCorpus {
                id: id.clone(),
                sha256: sha256.clone(),
                aspect: bundle.aspect.clone(),
                records: bundle.labels.len(),
                bundle_path: bundle_path.clone(),
                source_path,
                adopted_at: now_iso(),
            };
            registry.corpora.push(entry.clone());
            entry
        }
    };
    registry.selected_corpus_id = id;
    registry.corpora.sort_by(|left, right| left.id.cmp(&right.id));
    durable_replace(&registry_path, &serde_json::to_vec_pretty(&registry)?)?;
    let records = bundle.labels.len();
    Ok(json!({
        "status": if already_present { "unchanged" } else { "adopted" },
        "corpusId": adopted.id,
        "aspect": adopted.aspect,
        "sourcePath": adopted.source_path,
        "bundlePath": adopted.bundle_path,
        "selected": true,
        "records": records,
        "imported": if already_present { 0 } else { records },
        "unchanged": if already_present { records } else { 0 },
        "conflicting": 0,
        "rejected": 0,
    }))
}

pub fn status() -> Result<Value> {
    let registry_path = corpora_root().join(REGISTRY_FILE);
    let registry = read_registry(&registry_path)?;
    let selected = registry.as_ref().and_then(|registry| {
        registry
            .corpora
            .iter()
            .find(|entry| entry.id == registry.selected_corpus_id)
            .cloned()
    });
    let corpora = registry
        .map(|registry| registry.corpora)
        .unwrap_or_default();
    Ok(json!({
        "registry": registry_path,
        "selected": selected,
        "corpora": corpora,
    }))
}

pub fn selected_bundle_path() -> Result<Option<PathBuf>> {
    let path = corpora_root().join(REGISTRY_FILE);
    let Some(registry) = read_registry(&path)? else {
        return Ok(None);
    };
    let selected_id = registry.selected_corpus_id;
    let selected = registry
        .corpora
        .into_iter()
        .find(|entry| entry.id == selected_id)
        .ok_or_else(|| {
            Error(format!(
                "corpus registry {} selects unknown corpus {}",
                path.display(),
                selected_id
            ))
        })?;
    if !selected.bundle_path.is_file() {
        return Err(Error(format!(
            "selected corpus bundle is unavailable: {}",
            selected.bundle_path.display()
        )));
    }
    let digest = format!("{:x}", Sha256::digest(fs::read(&selected.bundle_path)?));
    if digest != selected.sha256 {
        return Err(Error(format!(
            "selected corpus bundle changed after adoption: {}",
            selected.bundle_path.display()
        )));
    }
    Ok(Some(selected.bundle_path))
}

fn corpora_root() -> PathBuf {
    resolve_placement().training_root.join(CORPORA_DIR)
}

fn validate_bundle(bundle: &CorpusBundle, path: &Path) -> Result<()> {
    if bundle.schema_version != BUNDLE_SCHEMA {
        return Err(Error(format!(
            "unsupported dataset bundle schema {} in {}; expected {}",
            bundle.schema_version,
            path.display(),
            BUNDLE_SCHEMA
        )));
    }
    crate::model::aspect_dir(&bundle.aspect)?;
    if bundle.labels.is_empty() {
        return Err(Error(format!(
            "dataset bundle {} contains no label records",
            path.display()
        )));
    }
    let mut sessions = BTreeSet::new();
    for (index, row) in bundle.labels.iter().enumerate() {
        let number = index + 1;
        if row.session_id.trim().is_empty()
            || row.value.trim().is_empty()
            || row.source.trim().is_empty()
            || row.text.trim().is_empty()
        {
            return Err(Error(format!(
                "dataset bundle record {number} requires nonempty session_id, value, source, and text; no corpus state was changed"
            )));
        }
        if chrono::DateTime::parse_from_rfc3339(&row.ts).is_err() {
            return Err(Error(format!(
                "dataset bundle record {number} has invalid RFC 3339 ts; no corpus state was changed"
            )));
        }
        if let Some(runtime) = &row.runtime {
            if runtime.trim().is_empty() {
                return Err(Error(format!(
                    "dataset bundle record {number} has an empty runtime; use null when unknown"
                )));
            }
        }
        if !sessions.insert(row.session_id.as_str()) {
            return Err(Error(format!(
                "dataset bundle repeats native session_id '{}' at record {number}; no corpus state was changed",
                row.session_id
            )));
        }
    }
    Ok(())
}

fn read_registry(path: &Path) -> Result<Option<CorpusRegistry>> {
    if !path.exists() {
        return Ok(None);
    }
    let registry: CorpusRegistry = serde_json::from_slice(&fs::read(path)?).map_err(|error| {
        Error(format!("invalid corpus registry {}: {error}", path.display()))
    })?;
    if registry.schema_version != REGISTRY_SCHEMA {
        return Err(Error(format!(
            "unsupported corpus registry schema {} in {}; expected {}",
            registry.schema_version,
            path.display(),
            REGISTRY_SCHEMA
        )));
    }
    let mut ids = BTreeSet::new();
    for corpus in &registry.corpora {
        if !ids.insert(corpus.id.as_str())
            || corpus.id != format!("dataset-bundle:{}", corpus.sha256)
            || !corpus.bundle_path.is_absolute()
        {
            return Err(Error(format!(
                "corpus registry {} carries an invalid or duplicate identity",
                path.display()
            )));
        }
    }
    if !ids.contains(registry.selected_corpus_id.as_str()) {
        return Err(Error(format!(
            "corpus registry {} selects unknown corpus {}",
            path.display(),
            registry.selected_corpus_id
        )));
    }
    Ok(Some(registry))
}

fn durable_create(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn durable_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".registry.{}.tmp", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}
