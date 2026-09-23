use super::*;

/// Concatenated user+assistant text per session, ordered by ts.
///
/// Only sessions with at least one text event appear; each is capped at
/// [`TEXT_CAP`] characters.
pub fn session_texts(session_ids: &[String]) -> Result<HashMap<String, SessionText>> {
    if let Some(bundle) = read_pinned_bundle()? {
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
    if let Some(bundle) = read_training_bundle()? {
        let wanted: std::collections::HashSet<&str> =
            session_ids.iter().map(String::as_str).collect();
        let available: HashMap<String, SessionText> = bundle
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
            .collect();
        if available.len() == wanted.len() {
            return Ok(available);
        }
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
pub(crate) fn run_lake(args: &[&str]) -> Result<Output> {
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

pub(crate) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

pub(crate) fn checkout_lake_binary() -> PathBuf {
    home_dir()
        .join("Documents")
        .join("CodingProjects")
        .join("Wisent")
        .join("transcript-lake")
        .join("target")
        .join("release")
        .join(LAKE_BINARY)
}

pub(crate) fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

pub(crate) fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}
