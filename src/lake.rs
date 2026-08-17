//! Read-only access to Transcript Lake state.
//!
//! Two contracts are consumed here, neither is reimplemented:
//!
//! - the append-only label store at `<storage root>/labels/*.ndjson`, owned by
//!   `transcript-lake label` — this module only ever reads it;
//! - the canonical `events`/`sessions` DuckDB views, reached by shelling out
//!   to the lake CLI (`query --json`) so the SQL setup in sql/views.sql stays
//!   the lake's own code.
//!
//! The one write path is [`label_add`], and it is a write the lake performs:
//! `autolabel` hands the lake CLI a `label add`, and the lake validates the
//! session and appends the record. Nothing here opens the label store for
//! writing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bail;
use crate::placement::resolve_placement;
use crate::util::{home_dir, json_text, json_truthy, Error, Result};

/// Characters of concatenated session text kept per session.
const TEXT_CAP: usize = 12_000;

/// The lake CLI is a Rust binary now; `cargo install --path .` puts it on
/// PATH under this name.
const LAKE_BINARY: &str = "transcript-lake";

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

const DATASET_BUNDLE_ENV: &str = "TLT_DATASET_BUNDLE";
const DATASET_BUNDLE_SCHEMA: u32 = 1;

#[derive(Debug, Deserialize, Serialize)]
struct DatasetBundle {
    schema_version: u32,
    aspect: String,
    labels: Vec<SessionLabel>,
}

fn read_bundle() -> Result<Option<DatasetBundle>> {
    let Some(path) = std::env::var_os(DATASET_BUNDLE_ENV).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let bundle: DatasetBundle =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
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
    Ok(Some(bundle))
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
    if let Some(bundle) = read_bundle()? {
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
    if let Some(bundle) = read_bundle()? {
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

/// Concatenated user+assistant text per session, ordered by ts.
///
/// Only sessions with at least one text event appear; each is capped at
/// [`TEXT_CAP`] characters.
pub fn session_texts(session_ids: &[String]) -> Result<HashMap<String, SessionText>> {
    if let Some(bundle) = read_bundle()? {
        let wanted: std::collections::HashSet<&str> =
            session_ids.iter().map(String::as_str).collect();
        return Ok(bundle
            .labels
            .into_iter()
            .filter(|label| wanted.contains(label.session_id.as_str()))
            .map(|label| {
                (
                    label.session_id,
                    SessionText {
                        runtime: label.runtime,
                        text: label.text,
                    },
                )
            })
            .collect());
    }
    if session_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let wanted = session_ids
        .iter()
        .map(|id| quote(id))
        .collect::<Vec<_>>()
        .join(", ");
    let rows = query(&format!(
        "SELECT session_id, runtime, ts, text FROM events \
         WHERE event_type IN ('user', 'assistant') AND text IS NOT NULL \
         AND session_id IN ({wanted}) \
         ORDER BY ts"
    ))?;

    let mut runtimes: HashMap<String, Option<String>> = HashMap::new();
    let mut parts: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows {
        let session_id = row.get("session_id").map(json_text).unwrap_or_default();
        // The runtime of the first row for a session wins, as it did when the
        // Python used setdefault on the accumulator.
        runtimes.entry(session_id.clone()).or_insert_with(|| {
            row.get("runtime")
                .filter(|runtime| !runtime.is_null())
                .map(json_text)
        });
        parts
            .entry(session_id)
            .or_default()
            .push(row.get("text").map(json_text).unwrap_or_default());
    }

    let mut texts = HashMap::with_capacity(parts.len());
    for (session_id, parts) in parts {
        let joined = parts.join("\n");
        // The cap is on characters, not bytes: Polish transcripts would
        // otherwise be cut to a different length than the Python cut them.
        let text = match joined.char_indices().nth(TEXT_CAP) {
            Some((end, _)) => joined[..end].to_string(),
            None => joined,
        };
        let runtime = runtimes.remove(&session_id).flatten();
        texts.insert(session_id, SessionText { runtime, text });
    }
    Ok(texts)
}

/// Apply one label through the lake CLI, which validates it and owns the write.
///
/// This is the single write path in the whole product, and it is deliberately
/// not a file append: the lake refuses labels for sessions it does not know,
/// and that check is the reason the boundary exists.
pub fn label_add(
    session_id: &str,
    aspect: &str,
    value: &str,
    source: &str,
    note: &str,
) -> Result<()> {
    let out = run_lake(&[
        "label", "add", session_id, "--aspect", aspect, "--value", value, "--source", source,
        "--note", note,
    ])?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let detail = if stderr.is_empty() {
        String::from_utf8_lossy(&out.stdout).into_owned()
    } else {
        stderr.into_owned()
    };
    let detail = detail.trim();
    let detail: String = detail.chars().take(200).collect();
    bail!("lake label add failed: {detail}");
}

/// One operator-facing warning line on stderr, prefixed with the product name.
pub fn warn(message: &str) {
    eprintln!("transcript-label-trainer: {message}");
}

/// Invoke the lake CLI with the resolved storage root in `LAKE_DATA`.
fn run_lake(args: &[&str]) -> Result<Output> {
    let cli = lake_cli();
    let Some((program, prefix)) = cli.split_first() else {
        bail!("TLT_LAKE_CLI is set but names no command to run");
    };
    let storage_root = resolve_placement().storage_root;
    Command::new(program)
        .args(prefix)
        .args(args)
        .env("LAKE_DATA", &storage_root)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| Error(format!("could not run the lake CLI '{program}': {error}")))
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn checkout_lake_binary() -> PathBuf {
    home_dir()
        .join("Documents")
        .join("CodingProjects")
        .join("Wisent")
        .join("transcript-lake")
        .join("target")
        .join("release")
        .join(LAKE_BINARY)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}
