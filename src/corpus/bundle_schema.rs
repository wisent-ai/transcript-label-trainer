use super::*;

pub(crate) const BUNDLE_SCHEMA: u32 = 1;

pub(crate) const REGISTRY_SCHEMA: u32 = 1;

pub(crate) const CORPORA_DIR: &str = "corpora";

pub(crate) const REGISTRY_FILE: &str = "registry.json";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorpusBundle {
    pub(crate) schema_version: u32,
    pub(crate) aspect: String,
    pub(crate) labels: Vec<CorpusRow>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorpusRow {
    pub(crate) session_id: String,
    pub(crate) value: String,
    pub(crate) source: String,
    pub(crate) ts: String,
    pub(crate) runtime: Option<String>,
    pub(crate) text: String,
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
    #[serde(rename = "sourceName", default, skip_serializing_if = "Option::is_none")]
    pub source_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CorpusRegistry {
    #[serde(rename = "schemaVersion")]
    pub(crate) schema_version: u32,
    #[serde(rename = "selectedCorpusId")]
    pub(crate) selected_corpus_id: String,
    pub(crate) corpora: Vec<AdoptedCorpus>,
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
    adopt_raw(&raw, source_path, None)
}

/// Adopt bytes selected in the browser without staging them in a temporary
/// directory. The raw-content identity is stable across GUI sessions and the
/// canonical bundle is retained by the same operation the path-based CLI uses.
pub fn adopt_upload(file_name: &str, raw: &[u8]) -> Result<Value> {
    validate_upload_name(file_name)?;
    let source_sha256 = format!("{:x}", Sha256::digest(raw));
    let source_identity = PathBuf::from(format!("browser-upload:sha256:{source_sha256}"));
    adopt_raw(raw, source_identity, Some(file_name.to_string()))
}

pub(crate) fn adopt_raw(
    raw: &[u8],
    source_identity: PathBuf,
    source_name: Option<String>,
) -> Result<Value> {
    let bundle: CorpusBundle = serde_json::from_slice(raw).map_err(|error| {
        Error(format!(
            "invalid dataset bundle {}: {error}; no corpus state was changed",
            source_identity.display()
        ))
    })?;
    validate_bundle(&bundle, &source_identity)?;
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
                source_path: source_identity.clone(),
                adopted_at: now_iso(),
                source_name: source_name.clone(),
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
        "sourceIdentity": source_identity,
        "sourceName": source_name,
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

pub(crate) fn validate_upload_name(file_name: &str) -> Result<()> {
    if file_name.is_empty()
        || file_name.len() > 255
        || file_name == "."
        || file_name == ".."
        || file_name
            .chars()
            .any(|character| character.is_control() || character == '/' || character == '\\')
    {
        return Err(Error(
            "uploaded corpus filename must be a 1-255 character basename".to_string(),
        ));
    }
    Ok(())
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

pub(crate) fn corpora_root() -> PathBuf {
    resolve_placement().training_root.join(CORPORA_DIR)
}

pub(crate) fn validate_bundle(bundle: &CorpusBundle, path: &Path) -> Result<()> {
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
