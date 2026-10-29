//! Publishing a qualified model as one immutable revision of a private
//! Hugging Face repository, verified file by file after the upload.

use super::*;

/// Join the verified parts into the qualified artifact and upload it, the
/// manifest and every evidence file as one new branch of the private
/// repository `repo`, then read that branch's immutable revision back and
/// check each file's size and SHA-256 against what was verified here. A
/// repository that exists and is public is refused before anything uploads.
pub(super) fn publish_private(
    model: &ReleaseModel,
    repo: &str,
    root: &Path,
    parts: &[PathBuf],
    artifact: &str,
    (digest, bytes): (&str, u64),
) -> Result<Value> {
    use crate::hub::{Hub, Signature};
    let hub = Hub::new()?;
    let public = |info: &crate::hub::ModelInfo| -> Result<()> {
        if !info.private {
            crate::bail!("Hugging Face repository {repo} is public; a model is published only to a private one");
        }
        Ok(())
    };
    match hub.info(repo, None, false)? {
        Some(info) => public(&info)?,
        None => hub.create_private(repo)?,
    }
    let assembled = root.join(artifact);
    {
        let mut output = File::create(&assembled)
            .map_err(|error| Error(format!("cannot create {}: {error}", assembled.display())))?;
        for part in parts {
            File::open(part)
                .and_then(|mut input| io::copy(&mut input, &mut output))
                .map_err(|error| {
                    Error(format!(
                        "cannot append {} to {}: {error}",
                        part.display(),
                        assembled.display()
                    ))
                })?;
            std::fs::remove_file(part).map_err(|error| {
                Error(format!(
                    "cannot remove the joined part {}: {error}",
                    part.display()
                ))
            })?;
        }
    }
    let mut signatures = std::collections::BTreeMap::from([(
        artifact.to_string(),
        Signature {
            size: bytes,
            sha256: digest.to_string(),
        },
    )]);
    for name in model.metadata.iter().chain(["model-manifest.json"].iter()) {
        signatures.insert(name.to_string(), Signature::read(&root.join(name))?);
    }
    // A branch of its own per publication, named by a random 128-bit value.
    let branch = format!("{}-publication-{}", model.name, rand::random::<u128>());
    let message = format!("Publish qualified {} model {digest}", model.name);
    for name in signatures.keys() {
        hub.upload(repo, &branch, &root.join(name), name, &message)
            .map_err(|error| {
                Error(format!(
                    "publication {repo}@{branch}, path {name:?}: {error}"
                ))
            })?;
    }
    let info = hub.info(repo, Some(&branch), true)?.ok_or_else(|| {
        Error(format!(
            "published Hugging Face branch is unavailable: {repo}@{branch}"
        ))
    })?;
    public(&info)?;
    let revision = info
        .sha
        .clone()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error("Hugging Face model_info returned no immutable revision".into()))?;
    for (name, signature) in &signatures {
        let file = info
            .siblings
            .iter()
            .find(|file| &file.rfilename == name)
            .ok_or_else(|| {
                Error(format!(
                    "published revision {repo}@{revision} is missing {name}"
                ))
            })?;
        hub.verify(repo, &revision, signature, file)?;
    }
    Ok(json!({
        "repo_id": repo,
        "revision": revision,
        "branch": branch,
        "artifact": artifact,
        "digest": digest,
        "bytes": bytes,
        "verified_files": signatures,
    }))
}
