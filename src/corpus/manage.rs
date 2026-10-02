//! What can be done to a corpus already adopted: select it as the input
//! train, infer and evaluate read, or remove it with its retained copy. The
//! counterparts of `adopt`, which retains a bundle and selects it.

use super::*;

fn registry_or_refuse() -> Result<(PathBuf, CorpusRegistry)> {
    let path = corpora_root().join(REGISTRY_FILE);
    let registry = read_registry(&path)?
        .ok_or_else(|| Error(format!("no corpus has been adopted; {} does not exist", path.display())))?;
    Ok((path, registry))
}

fn refuse_unknown(registry: &CorpusRegistry, id: &str) -> Error {
    let held: Vec<&str> = registry.corpora.iter().map(|entry| entry.id.as_str()).collect();
    Error(format!("no adopted corpus {id}; the store holds {}", held.join(", ")))
}

/// Make `id` the corpus train, infer and evaluate read.
pub fn select(id: &str) -> Result<Value> {
    let (path, mut registry) = registry_or_refuse()?;
    if !registry.corpora.iter().any(|entry| entry.id == id) {
        return Err(refuse_unknown(&registry, id));
    }
    if registry.selected_corpus_id == id {
        return Err(Error(format!("corpus {id} is already selected")));
    }
    registry.selected_corpus_id = id.to_string();
    durable_replace(&path, &serde_json::to_vec_pretty(&registry)?)?;
    status()
}

/// Remove `id` and its retained bundle. The selected corpus is refused while
/// others remain, so training never reads a selection nobody made; removing
/// the last corpus removes the registry with it.
pub fn remove(id: &str) -> Result<Value> {
    let (path, mut registry) = registry_or_refuse()?;
    let position = registry
        .corpora
        .iter()
        .position(|entry| entry.id == id)
        .ok_or_else(|| refuse_unknown(&registry, id))?;
    if registry.selected_corpus_id == id && registry.corpora.len() > 1 {
        return Err(Error(format!(
            "corpus {id} is selected; select another with corpus-select before removing it"
        )));
    }
    let entry = registry.corpora.remove(position);
    if registry.corpora.is_empty() {
        fs::remove_file(&path)?;
    } else {
        durable_replace(&path, &serde_json::to_vec_pretty(&registry)?)?;
    }
    match fs::remove_file(&entry.bundle_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(Error(format!(
                "corpus {id} left the registry but its bundle {} was not deleted: {error}",
                entry.bundle_path.display()
            )))
        }
    }
    let mut answer = status()?;
    answer["removed"] = json!(id);
    Ok(answer)
}
