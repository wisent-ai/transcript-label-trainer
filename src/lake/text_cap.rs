use super::*;

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

/// The dataset bundle this module writes for a Stado run and reads back is
/// the corpus bundle the corpus module declares; its schema version is that
/// declaration's, never a second copy.
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
    if bundle.schema_version != crate::corpus::BUNDLE_SCHEMA {
        bail!(
            "unsupported dataset bundle schema {} in {}; expected {}",
            bundle.schema_version,
            path.display(),
            crate::corpus::BUNDLE_SCHEMA
        );
    }
    Ok(bundle)
}

/// The bundle a pinned Stado job was handed in `TLT_DATASET_BUNDLE`; `None`
/// only when the variable is unset. A set but empty value is refused: it
/// names no file, and reading it as unset would send the job back to the
/// source machine's lake.
pub(crate) fn read_pinned_bundle() -> Result<Option<DatasetBundle>> {
    let Some(path) = std::env::var_os(DATASET_BUNDLE_ENV) else {
        return Ok(None);
    };
    if path.is_empty() {
        bail!("{DATASET_BUNDLE_ENV} is set but empty; name the dataset bundle file or unset it");
    }
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
        schema_version: crate::corpus::BUNDLE_SCHEMA,
        aspect: aspect.to_string(),
        labels: bundled,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&bundle)?)?;
    Ok(())
}

/// Command prefix that invokes the lake CLI: `TLT_LAKE_CLI` split on
/// whitespace into a command and its arguments when it is set, otherwise the
/// lake binary's name, which the system looks up on PATH when it runs. A lake
/// that is not installed is refused there, naming the program; a set but
/// empty `TLT_LAKE_CLI` is refused by `run_lake`, never read as unset.
pub fn lake_cli() -> Result<Vec<String>> {
    match std::env::var("TLT_LAKE_CLI") {
        Ok(value) => Ok(value.split_whitespace().map(str::to_string).collect()),
        Err(std::env::VarError::NotPresent) => Ok(vec![LAKE_BINARY.to_string()]),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error("TLT_LAKE_CLI is set but is not UTF-8".to_string()))
        }
    }
}

/// A text field every label-store record carries, or the record refused
/// with the file it came from.
fn record_text(record: &Value, field: &str, file: &Path, line: &str) -> Result<String> {
    match record.get(field) {
        Some(value) if !value.is_null() => Ok(json_text(value)),
        _ => Err(Error(format!(
            "label store {} holds a record without {field}: {line}",
            file.display()
        ))),
    }
}

/// Latest label in force per session for one aspect.
///
/// The store is append-only, so the record with the newest `ts` decides each
/// `session_id`: a value assigns it, and the lake's withdrawal record, whose
/// value is null, leaves the session unlabeled. A store with no labels
/// directory holds no labels. Sessions keep the order they were first seen
/// in, files are read in sorted name order so the listing order cannot
/// decide a tie on `ts`, and a line that is not JSON or a record of this
/// aspect without its session, value, source or time is refused with its
/// file: skipping it would train on a store that silently lost labels.
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
    // Each session's newest record, with the value it assigns, if any.
    let mut latest: Vec<(SessionLabel, Option<String>)> = Vec::new();
    let entries = match std::fs::read_dir(&labels_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            bail!("cannot list the label store {}: {error}", labels_dir.display())
        }
    };

    let mut names: Vec<std::ffi::OsString> = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if Path::new(&name).extension().and_then(|ext| ext.to_str()) == Some("ndjson") {
            names.push(name);
        }
    }
    names.sort_unstable();

    for name in names {
        let file = labels_dir.join(&name);
        let text = std::fs::read_to_string(&file)?;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record = serde_json::from_str::<Value>(line).map_err(|error| {
                Error(format!(
                    "label store {} holds a line that is not JSON ({error}): {line}",
                    file.display()
                ))
            })?;
            if record.get("aspect").and_then(Value::as_str) != Some(aspect) {
                continue;
            }
            let session_id = record_text(&record, "session_id", &file, line)?;
            let assigned = match record.get("value") {
                Some(Value::Null) => None,
                Some(value) => Some(json_text(value)),
                None => bail!("label store {} holds a record without value: {line}", file.display()),
            };
            let label = SessionLabel {
                session_id: session_id.clone(),
                value: String::new(),
                source: record_text(&record, "source", &file, line)?,
                ts: record_text(&record, "ts", &file, line)?,
                runtime: record
                    .get("runtime")
                    .filter(|runtime| !runtime.is_null())
                    .map(json_text),
                text: String::new(),
            };
            match latest.iter_mut().find(|(seen, _)| seen.session_id == session_id) {
                // Newest ts wins, ties to the later record: the store is
                // append-only, so a second record with the same second is a
                // correction of the first.
                Some(current) if label.ts >= current.0.ts => *current = (label, assigned),
                Some(_) => {}
                None => latest.push((label, assigned)),
            }
        }
    }
    Ok(latest
        .into_iter()
        .filter_map(|(label, assigned)| assigned.map(|value| SessionLabel { value, ..label }))
        .collect())
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
    // DuckDB's JSON mode, which the lake's query runs in, prints nothing for
    // a result without rows.
    if stdout.is_empty() {
        return Ok(Vec::new());
    }
    match serde_json::from_str::<Value>(stdout)? {
        Value::Array(rows) => Ok(rows),
        _ => bail!("lake query did not return a JSON array"),
    }
}

/// A column the lake's own query selected, or the row refused naming it.
pub(crate) fn row_field<'row>(row: &'row Value, column: &str, view: &str) -> Result<&'row Value> {
    row.get(column).ok_or_else(|| {
        Error(format!(
            "the lake's {view} query answered a row without {column}: {row}"
        ))
    })
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
    rows.iter()
        .map(|row| {
            let runtime = row_field(row, "runtime", "sessions")?;
            Ok(SessionRow {
                session_id: json_text(row_field(row, "session_id", "sessions")?),
                runtime: (!runtime.is_null()).then(|| json_text(runtime)),
            })
        })
        .collect()
}
