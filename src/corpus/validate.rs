//! What an adoption refuses before it changes anything, and how it says why.

use super::*;

/// The longest file name, in bytes, the filesystems corpora are kept on take.
const NAME_MAX: usize = 255; // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/limits.h.html {NAME_MAX}, which macOS's and Linux's <limits.h> define as 255

/// Why an adoption changed nothing: the retained store already holds this
/// content's identity with other bytes, or anything else refused it.
#[derive(Debug)]
pub enum AdoptRefusal {
    IdentityConflict(Error),
    Refused(Error),
}

impl From<AdoptRefusal> for Error {
    fn from(refusal: AdoptRefusal) -> Self {
        match refusal {
            AdoptRefusal::IdentityConflict(error) | AdoptRefusal::Refused(error) => error,
        }
    }
}

impl From<Error> for AdoptRefusal {
    fn from(error: Error) -> Self {
        AdoptRefusal::Refused(error)
    }
}

impl From<std::io::Error> for AdoptRefusal {
    fn from(error: std::io::Error) -> Self {
        AdoptRefusal::Refused(Error::from(error))
    }
}

impl From<serde_json::Error> for AdoptRefusal {
    fn from(error: serde_json::Error) -> Self {
        AdoptRefusal::Refused(Error::from(error))
    }
}

pub(crate) fn validate_upload_name(file_name: &str) -> Result<()> {
    if file_name.is_empty()
        || file_name.len() > NAME_MAX
        || file_name == "."
        || file_name == ".."
        || file_name
            .chars()
            .any(|character| character.is_control() || character == '/' || character == '\\')
    {
        return Err(Error(format!(
            "uploaded corpus filename must be a 1-{NAME_MAX} character basename"
        )));
    }
    Ok(())
}

pub(crate) fn validate_bundle(bundle: &CorpusBundle, path: &Path) -> Result<()> {
    if bundle.schema_version != BUNDLE_SCHEMA {
        return Err(Error(format!(
            "unsupported dataset bundle schema {} in {}; expected {}",
            bundle.schema_version,
            path.display(),
            BUNDLE_SCHEMA
        )));
    }
    crate::model::aspect_dir(&bundle.aspect)?;
    if bundle.labels.is_empty() {
        return Err(Error(format!(
            "dataset bundle {} contains no label records",
            path.display()
        )));
    }
    let mut sessions = BTreeSet::new();
    for row in &bundle.labels {
        // A record is named by its native session id, which it must carry.
        if row.session_id.trim().is_empty() {
            return Err(Error(format!(
                "dataset bundle {} holds a record that requires nonempty session_id, value, source, and text: its session_id is empty; no corpus state was changed",
                path.display()
            )));
        }
        let record = &row.session_id;
        if row.value.trim().is_empty() || row.source.trim().is_empty() || row.text.trim().is_empty()
        {
            return Err(Error(format!(
                "dataset bundle record {record} requires nonempty session_id, value, source, and text; no corpus state was changed"
            )));
        }
        if let Err(error) = chrono::DateTime::parse_from_rfc3339(&row.ts) {
            return Err(Error(format!(
                "dataset bundle record {record} has invalid RFC 3339 ts {:?} ({error}); no corpus state was changed",
                row.ts
            )));
        }
        if let Some(runtime) = &row.runtime {
            if runtime.trim().is_empty() {
                return Err(Error(format!(
                    "dataset bundle record {record} has an empty runtime; use null when unknown"
                )));
            }
        }
        if !sessions.insert(row.session_id.as_str()) {
            return Err(Error(format!(
                "dataset bundle repeats native session_id '{record}'; no corpus state was changed"
            )));
        }
    }
    Ok(())
}
