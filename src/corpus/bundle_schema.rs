use super::*;

/// The dataset-bundle format this binary reads and writes.
pub(crate) const BUNDLE_SCHEMA: u32 = 1; // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/corpus-import.html "schema_version must be 1"

/// The corpus registry format this binary reads and writes.
pub(crate) const REGISTRY_SCHEMA: u32 = 1; // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/corpus-import.html "unsupported corpus registry schema"

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
    #[serde(
        rename = "sourceName",
        default,
        skip_serializing_if = "Option::is_none"
    )]
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
        Error(format!(
            "could not resolve corpus {}: {error}",
            path.display()
        ))
    })?;
    if !source_path.is_file() {
        return Err(Error(format!(
            "corpus is not a regular file: {}",
            source_path.display()
        )));
    }
    let raw = crate::util::read_bytes(&source_path)?;
    Ok(adopt_raw(&raw, source_path, None)?)
}

/// Adopt bytes selected in the browser without staging them in a temporary
/// directory. The raw-content identity is stable across GUI sessions and the
/// canonical bundle is retained by the same operation the path-based CLI uses.
pub fn adopt_upload(file_name: &str, raw: &[u8]) -> std::result::Result<Value, AdoptRefusal> {
    validate_upload_name(file_name)?;
    let source_sha256 = format!("{:x}", Sha256::digest(raw));
    let source_identity = PathBuf::from(format!("browser-upload:sha256:{source_sha256}"));
    adopt_raw(raw, source_identity, Some(file_name.to_string()))
}

pub(crate) fn adopt_raw(
    raw: &[u8],
    source_identity: PathBuf,
    source_name: Option<String>,
) -> std::result::Result<Value, AdoptRefusal> {
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
    // The first adoption starts the registry the later ones extend.
    let mut registry = match read_registry(&registry_path)? {
        Some(registry) => registry,
        None => CorpusRegistry {
            schema_version: REGISTRY_SCHEMA,
            selected_corpus_id: id.clone(),
            corpora: Vec::new(),
        },
    };
    let bundle_path = root.join(format!("{sha256}.json"));
    let already_present = bundle_path.exists();
    if already_present {
        if crate::util::read_bytes(&bundle_path)? != canonical {
            return Err(AdoptRefusal::IdentityConflict(Error(format!(
                "corpus identity conflict at {}; no corpus state was changed",
                bundle_path.display()
            ))));
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
    registry
        .corpora
        .sort_by(|left, right| left.id.cmp(&right.id));
    durable_replace(&registry_path, &serde_json::to_vec_pretty(&registry)?)?;
    // Every record of an adopted bundle is either new to the store or was
    // already retained with it; an adoption conflicts and rejects nothing.
    let (imported, unchanged): (Vec<&CorpusRow>, Vec<&CorpusRow>) =
        bundle.labels.iter().partition(|_| !already_present);
    let refused: Vec<&CorpusRow> = Vec::new();
    Ok(json!({
        "status": if already_present { "unchanged" } else { "adopted" },
        "corpusId": adopted.id,
        "aspect": adopted.aspect,
        "sourceIdentity": source_identity,
        "sourceName": source_name,
        "sourcePath": adopted.source_path,
        "bundlePath": adopted.bundle_path,
        "selected": true,
        "records": bundle.labels.len(),
        "imported": imported.len(),
        "unchanged": unchanged.len(),
        "conflicting": refused.len(),
        "rejected": refused.len(),
    }))
}

pub fn status() -> Result<Value> {
    let registry_path = corpora_root().join(REGISTRY_FILE);
    let (selected, corpora) = match read_registry(&registry_path)? {
        Some(registry) => {
            let selected = registry
                .corpora
                .iter()
                .find(|entry| entry.id == registry.selected_corpus_id)
                .cloned();
            (selected, registry.corpora)
        }
        // No corpus was ever adopted here.
        None => (None, Vec::new()),
    };
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
    let digest = format!(
        "{:x}",
        Sha256::digest(crate::util::read_bytes(&selected.bundle_path)?)
    );
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
