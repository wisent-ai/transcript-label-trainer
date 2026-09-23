use super::*;

pub(crate) fn read_registry(path: &Path) -> Result<Option<CorpusRegistry>> {
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

pub(crate) fn durable_create(path: &Path, bytes: &[u8]) -> Result<()> {
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

pub(crate) fn durable_replace(path: &Path, bytes: &[u8]) -> Result<()> {
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
