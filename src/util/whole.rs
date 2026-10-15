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

/// Write `bytes` to `output` whole, creating its directory first.
pub(crate) fn write_whole(output: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            Error(format!(
                "cannot create {} to write {} into: {error}",
                parent.display(),
                output.display()
            ))
        })?;
    }
    let temporary = sibling(output);
    let written = fs::write(&temporary, bytes)
        .map_err(|error| {
            Error(format!(
                "cannot write {} on the way to {}: {error}",
                temporary.display(),
                output.display()
            ))
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
        Ok(()) => Ok(()),
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
