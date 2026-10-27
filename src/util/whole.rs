//! Writing an output file whole: the bytes go to a sibling file that is
//! renamed into place, so a reader never sees half of them. Every failure
//! names the path and the step it failed at, and a failed write removes the
//! sibling it left, or says that it could not.
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use super::{Error, Result};

/// The sibling `output` is written through before it is renamed into place.
/// It carries this process's id, so two runs writing the same output never
/// write through, or rename away, each other's file.
fn sibling(output: &Path) -> PathBuf {
    let mut temporary = OsString::from(output.as_os_str());
    temporary.push(format!(".tmp-{}", std::process::id()));
    PathBuf::from(temporary)
}

/// Write `bytes` to `output` whole and durably, creating its directory first:
/// the sibling is flushed to disk before it is renamed into place, and the
/// directory after, so a machine that stops right after the write still
/// holds the new file and not the old one or nothing. Every writer of a whole
/// file in this program writes through here.
pub(crate) fn write_whole(output: &Path, bytes: &[u8]) -> Result<()> {
    let parent = match output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        Some(parent) => parent,
        None => Path::new("."),
    };
    fs::create_dir_all(parent).map_err(|error| {
        Error(format!(
            "cannot create {} to write {} into: {error}",
            parent.display(),
            output.display()
        ))
    })?;
    let temporary = sibling(output);
    let step = |what: &str, error: std::io::Error| {
        Error(format!(
            "cannot {what} {} on the way to {}: {error}",
            temporary.display(),
            output.display()
        ))
    };
    let written = fs::File::create(&temporary)
        .map_err(|error| step("create", error))
        .and_then(|mut file| {
            use std::io::Write;
            file.write_all(bytes).map_err(|error| step("write", error))?;
            file.sync_all().map_err(|error| step("flush to disk", error))
        })
        .and_then(|()| {
            fs::rename(&temporary, output).map_err(|error| {
                Error(format!(
                    "cannot move {} into place as {}: {error}",
                    temporary.display(),
                    output.display()
                ))
            })
        });
    match written {
        Ok(()) => fs::File::open(parent).and_then(|directory| directory.sync_all()).map_err(|error| {
            Error(format!(
                "{} was written, but its directory {} could not be flushed to disk: {error}",
                output.display(),
                parent.display()
            ))
        }),
        Err(error) => Err(match fs::remove_file(&temporary) {
            Ok(()) => error,
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => error,
            Err(cleanup) => Error(format!(
                "{error}; {} was left behind and cannot be removed: {cleanup}",
                temporary.display()
            )),
        }),
    }
}

/// Write each of `rows` as one JSON line of `output`, whole.
pub(crate) fn write_jsonl<'a>(
    output: &Path,
    rows: impl IntoIterator<Item = &'a serde_json::Value>,
) -> Result<()> {
    let mut lines = String::new();
    for row in rows {
        lines.push_str(&serde_json::to_string(row)?);
        lines.push('\n');
    }
    write_whole(output, lines.as_bytes())
}

/// Write `value` to `output` whole as indented JSON ending in a newline.
pub(crate) fn write_json_pretty(output: &Path, value: &impl serde::Serialize) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    write_whole(output, text.as_bytes())
}
