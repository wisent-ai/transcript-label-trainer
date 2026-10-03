use std::fs::File;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::util::{Error, Result};

#[derive(Serialize)]
pub(super) struct Signature {
    pub size: u64,
    pub sha256: String,
}

impl Signature {
    pub fn read(path: &Path) -> Result<Self> {
        let mut input = File::open(path)
            .map_err(|error| Error(format!("cannot open publication input {}: {error}", path.display())))?;
        Self::stream(&mut input)
            .map_err(|error| Error(format!("cannot hash publication input {}: {error}", path.display())))
    }

    pub fn stream(input: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut hash = Sha256::new();
        let size = std::io::copy(input, &mut hash)?;
        Ok(Self { size, sha256: hex::encode(hash.finalize()) })
    }
}
