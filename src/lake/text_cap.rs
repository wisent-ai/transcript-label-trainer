use super::*;

/// Characters of concatenated session text kept per session.
pub(crate) const TEXT_CAP: usize = 12_000;

/// The lake CLI is a Rust binary now; `cargo install --path .` puts it on
/// PATH under this name.
pub(crate) const LAKE_BINARY: &str = "transcript-lake";

/// One label-store record: the latest one on a session for one aspect.
///
/// `text` is empty here — the label store carries no transcript text, and
/// [`session_texts`] is what fills it in for the sessions a caller needs.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SessionLabel {
    pub session_id: String,
    pub value: String,
    pub source: String,
    pub ts: String,
    pub runtime: Option<String>,
    pub text: String,
}

/// Reconstructed transcript text for one session.
#[derive(Clone, Debug, Default)]
pub struct SessionText {
    pub runtime: Option<String>,
    pub text: String,
}

/// One row of the lake's `sessions` view.
#[derive(Clone, Debug, Default)]
pub struct SessionRow {
    pub session_id: String,
    pub runtime: Option<String>,
}

pub(crate) const DATASET_BUNDLE_ENV: &str = "TLT_DATASET_BUNDLE";

pub(crate) const DATASET_BUNDLE_SCHEMA: u32 = 1;

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct DatasetBundle {
    pub(crate) schema_version: u32,
    pub(crate) aspect: String,
    pub(crate) labels: Vec<SessionLabel>,
}

pub(crate) fn read_bundle_at(path: &Path) -> Result<DatasetBundle> {
    let bundle: DatasetBundle =
        serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
            Error(format!(
                "invalid dataset bundle {}: {error}",
                path.display()
            ))
        })?;
    if bundle.schema_version != DATASET_BUNDLE_SCHEMA {
        bail!(
            "unsupported dataset bundle schema {} in {}; expected {}",
            bundle.schema_version,
            path.display(),
            DATASET_BUNDLE_SCHEMA
        );
    }
    Ok(bundle)
}

pub(crate) fn read_pinned_bundle() -> Result<Option<DatasetBundle>> {
    let Some(path) = std::env::var_os(DATASET_BUNDLE_ENV).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    Ok(Some(read_bundle_at(&PathBuf::from(path))?))
}

pub(crate) fn read_training_bundle() -> Result<Option<DatasetBundle>> {
    if let Some(bundle) = read_pinned_bundle()? {
        return Ok(Some(bundle));
    }
    let Some(path) = crate::corpus::selected_bundle_path()? else {
        return Ok(None);
    };
    Ok(Some(read_bundle_at(&path)?))
}

/// Materialize only the selected labels and their transcript text for a remote
/// Stado run. The bundle is read-only and contains no unrelated lake sessions.
pub fn export_bundle(aspect: &str, labels: &[SessionLabel], path: &Path) -> Result<()> {
    let ids: Vec<String> = labels
        .iter()
        .map(|label| label.session_id.clone())
        .collect();
    let mut texts = session_texts(&ids)?;
    let mut bundled = Vec::with_capacity(labels.len());
    for label in labels {
        let mut label = label.clone();
        if let Some(text) = texts.remove(&label.session_id) {
            label.runtime = label.runtime.or(text.runtime);
            label.text = text.text;
        }
        bundled.push(label);
    }
    let bundle = DatasetBundle {
        schema_version: DATASET_BUNDLE_SCHEMA,
        aspect: aspect.to_string(),
        labels: bundled,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&bundle)?)?;
    Ok(())
}

/// Command prefix that invokes the lake CLI.
///
/// `TLT_LAKE_CLI` overrides it and is split on whitespace into argv, exactly
/// as before. The default is the bare binary name resolved on PATH; only when
/// the lake is not installed does this fall back to the release build in the
/// neighbouring checkout, so a developer with just the two working copies
/// still gets a working trainer.
pub fn lake_cli() -> Vec<String> {
    if let Ok(override_value) = std::env::var("TLT_LAKE_CLI") {
        if !override_value.trim().is_empty() {
            return override_value
                .split_whitespace()
                .map(str::to_string)
                .collect();
        }
    }
    if find_on_path(LAKE_BINARY).is_some() {
        return vec![LAKE_BINARY.to_string()];
    }
    vec![checkout_lake_binary().to_string_lossy().into_owned()]
}

/// Latest label record per session for one aspect.
///
/// The store is append-only, so the record with the newest `ts` wins for each
/// `session_id`. A missing labels directory simply means zero labels. Records
/// keep the order in which their session was first seen, which is the order
/// the files themselves are read in — and the files are read in sorted name
/// order, because the directory listing order must not decide which record
/// a tie on `ts` resolves to.
pub fn load_labels(aspect: &str) -> Result<Vec<SessionLabel>> {
    if let Some(bundle) = read_training_bundle()? {
        if bundle.aspect != aspect {
            bail!(
                "dataset bundle carries aspect '{}', not requested aspect '{aspect}'",
                bundle.aspect
            );
        }
        return Ok(bundle.labels);
    }
    let labels_dir = resolve_placement().storage_root.join("labels");
    let mut latest: Vec<SessionLabel> = Vec::new();
    if !labels_dir.is_dir() {
        return Ok(latest);
    }

    let mut names: Vec<std::ffi::OsString> = Vec::new();
    for entry in std::fs::read_dir(&labels_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if Path::new(&name).extension().and_then(|ext| ext.to_str()) == Some("ndjson") {
            names.push(name);
        }
    }
    names.sort_unstable();

    for name in names {
        let text = std::fs::read_to_string(labels_dir.join(&name))?;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if record.get("aspect").and_then(Value::as_str) != Some(aspect) {
                continue;
            }
            let Some(session_id) = record.get("session_id").filter(|id| json_truthy(id)) else {
                continue;
            };
            let session_id = json_text(session_id);
            let label = SessionLabel {
                session_id: session_id.clone(),
                value: record.get("value").map(json_text).unwrap_or_default(),
                source: record.get("source").map(json_text).unwrap_or_default(),
                ts: record
                    .get("ts")
                    .filter(|ts| !ts.is_null())
                    .map(json_text)
                    .unwrap_or_default(),
                runtime: record
                    .get("runtime")
                    .filter(|runtime| !runtime.is_null())
                    .map(json_text),
                text: String::new(),
            };
            match latest.iter_mut().find(|seen| seen.session_id == session_id) {
                // Newest ts wins, ties to the later record: the store is
                // append-only, so a second record with the same second is a
                // correction of the first.
                Some(current) if label.ts >= current.ts => *current = label,
                Some(_) => {}
                None => latest.push(label),
            }
        }
    }
    Ok(latest)
}

/// Run SQL over the lake views and return the rows.
pub fn query(sql: &str) -> Result<Vec<Value>> {
    let out = run_lake(&["query", "--json", sql])?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let detail = if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            stderr.trim().to_string()
        };
        bail!("lake query failed: {detail}");
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stdout = stdout.trim();
    if stdout.is_empty() {
        return Ok(Vec::new());
    }
    match serde_json::from_str::<Value>(stdout)? {
        Value::Array(rows) => Ok(rows),
        _ => bail!("lake query did not return a JSON array"),
    }
}

/// Every session known to the lake.
pub fn all_sessions() -> Result<Vec<SessionRow>> {
    if let Some(bundle) = read_pinned_bundle()? {
        return Ok(bundle
            .labels
            .into_iter()
            .map(|label| SessionRow {
                session_id: label.session_id,
                runtime: label.runtime,
            })
            .collect());
    }
    let rows = query("SELECT runtime, session_id FROM sessions ORDER BY last_ts DESC")?;
    Ok(rows
        .into_iter()
        .map(|row| SessionRow {
            session_id: row.get("session_id").map(json_text).unwrap_or_default(),
            runtime: row
                .get("runtime")
                .filter(|runtime| !runtime.is_null())
                .map(json_text),
        })
        .collect())
}
