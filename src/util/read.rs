//! Reading an input file whole. A failed read names the file, so a refusal
//! says which of a run's inputs could not be read, not only why.
use std::fs;
use std::path::Path;

use super::{Error, Result};

/// The bytes of `path`, or an error naming it.
pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|error| Error(format!("cannot read {}: {error}", path.display())))
}

/// The text of `path`, or an error naming it; a file that is not UTF-8 is
/// refused the same way.
pub(crate) fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))
}
