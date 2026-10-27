use super::*;

pub(crate) fn read_registry(path: &Path) -> Result<Option<CorpusRegistry>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|error| failed("read the corpus registry", path, error))?;
    let registry: CorpusRegistry = serde_json::from_slice(&bytes).map_err(|error| {
        Error(format!(
            "invalid corpus registry {}: {error}",
            path.display()
        ))
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

/// The directory a durable write syncs after it writes `path`; a path that
/// names no directory has nothing to sync and is refused.
fn parent_of(path: &Path) -> Result<&Path> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => Ok(parent),
        _ => Err(Error(format!(
            "{} names no directory to write it durably into",
            path.display()
        ))),
    }
}

/// `step` failed on `path`: the sentence a durable write answers with.
fn failed(step: &str, path: &Path, error: std::io::Error) -> Error {
    Error(format!("cannot {step} {}: {error}", path.display()))
}

pub(crate) fn durable_create(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = parent_of(path)?;
    fs::create_dir_all(parent).map_err(|error| failed("create the directory", parent, error))?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| failed("create", path, error))?;
    file.write_all(bytes)
        .map_err(|error| failed("write", path, error))?;
    file.sync_all()
        .map_err(|error| failed("flush to disk", path, error))?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| failed("flush to disk the directory", parent, error))?;
    Ok(())
}

/// The registry written whole and durably, ending in a newline, through the
/// program's one whole-file writer (`util::write_whole`).
pub(crate) fn durable_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut document = bytes.to_vec();
    document.push(b'\n');
    crate::util::write_whole(path, &document)
}
