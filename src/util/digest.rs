//! The content identity every record of the trainer names a file, a dataset
//! or a run by: the SHA-256 of its bytes, written as lowercase hex.
use sha2::{Digest, Sha256};

/// The lowercase hex SHA-256 of `bytes`.
pub(crate) fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes))
}
