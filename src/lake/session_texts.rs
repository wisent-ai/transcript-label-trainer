use super::*;

/// Concatenated user+assistant text per session, ordered by ts.
///
/// Only sessions with at least one text event appear; each text is the whole
/// session. A model or teacher that cannot take it answers so itself.
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
    for row in &rows {
        let session_id = json_text(row_field(row, "session_id", "events")?);
        let runtime = row_field(row, "runtime", "events")?;
        // A session's runtime is the one its first row in time names.
        runtimes
            .entry(session_id.clone())
            .or_insert_with(|| (!runtime.is_null()).then(|| json_text(runtime)));
        parts
            .entry(session_id)
            .or_default()
            .push(json_text(row_field(row, "text", "events")?));
    }

    let mut texts = HashMap::with_capacity(parts.len());
    for (session_id, parts) in parts {
        let text = parts.join("\n");
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
    bail!("lake label add failed: {}", detail.trim());
}

/// One operator-facing warning line on stderr, prefixed with the product name.
pub fn warn(message: &str) {
    eprintln!("transcript-label-trainer: {message}");
}

/// Invoke the lake CLI with the resolved storage root in `LAKE_DATA`.
pub(crate) fn run_lake(args: &[&str]) -> Result<Output> {
    let cli = lake_cli()?;
    let Some((program, prefix)) = cli.split_first() else {
        bail!("TLT_LAKE_CLI is set but names no command to run");
    };
    let storage_root = resolve_placement().storage_root;
    stado_wait::output(
        Command::new(program)
            .args(prefix)
            .args(args)
            .env("LAKE_DATA", &storage_root)
            .stdin(Stdio::null()),
    )
    .map_err(|error| Error(format!("could not run the lake CLI '{program}': {error}")))
}

pub(crate) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
