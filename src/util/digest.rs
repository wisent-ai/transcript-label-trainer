//! The content identity every record of the trainer names a file, a dataset
//! or a run by: the SHA-256 of its bytes, written as lowercase hex. Bytes in
//! memory and bytes streamed from readers take the same digest here, so no
//! other module builds a SHA-256 of its own.
use sha2::{Digest, Sha256};

/// The lowercase hex SHA-256 of `bytes`.
pub(crate) fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// The lowercase hex SHA-256 of everything `readers` yield, read in order
/// as one stream, and the number of bytes read. A read that fails is
/// answered with the index of the reader it failed on, so the caller can
/// name the file.
pub(crate) fn sha256_hex_stream<R: std::io::Read>(
    readers: impl IntoIterator<Item = R>,
) -> Result<(String, u64), (usize, std::io::Error)> {
    let mut hasher = Sha256::new();
    let mut bytes = 0;
    for (index, mut reader) in readers.into_iter().enumerate() {
        bytes += std::io::copy(&mut reader, &mut hasher).map_err(|error| (index, error))?;
    }
    Ok((hex::encode(hasher.finalize()), bytes))
}
