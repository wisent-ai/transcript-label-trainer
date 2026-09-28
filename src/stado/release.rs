use super::*;

use std::fs::{self, File};

use serde_json::{json, Value};

/// A fine-tune whose qualified Stado job output may become a release.
pub(crate) struct ReleaseModel {
    pub(crate) name: &'static str,
    /// Release prefix; the assembled artifact's SHA-256 is appended.
    destination: &'static str,
    /// The decision contract the manifest must name, when the model has one.
    contract: Option<&'static str>,
    /// Evidence published beside the model, each checked against the manifest.
    metadata: &'static [&'static str],
}

pub(crate) const RELEASE_MODELS: [ReleaseModel; 2] = [
    ReleaseModel {
        name: "goal",
        destination: "stado://releases/jeden-desktop/models/goal-qwen3-4b",
        contract: None,
        metadata: &[
            "final-judge.json",
            "metrics.json",
            "predictions.jsonl",
            "goal-system-prompt.md",
            "python-requirements.lock",
        ],
    },
    ReleaseModel {
        name: "lifecycle",
        destination: "stado://releases/oko/models/lifecycle-qwen3-4b",
        contract: Some("oko-goal-lifecycle-v1"),
        metadata: &[
            "final-judge.json",
            "metrics.json",
            "predictions.jsonl",
            "lifecycle-system-prompt.txt",
            "lifecycle-output-schema.json",
            "python-requirements.lock",
        ],
    },
];

const GATE: &str = "final-judge.json";
const CONTENT_TYPE: &str = "application/octet-stream";

fn fetch(stado: &Path, uri: &str, destination: &Path) -> Result<()> {
    let args = [
        OsString::from("storage"),
        OsString::from("get"),
        OsString::from(uri),
        destination.as_os_str().to_owned(),
    ];
    let output = run(stado, &args)?;
    match output.status.success() {
        true => Ok(()),
        false => Err(command_error(stado, &format!("downloading {uri}"), &output)),
    }
}

fn present(stado: &Path, uri: &str) -> Result<bool> {
    let args = [
        OsString::from("storage"),
        OsString::from("stat"),
        OsString::from(uri),
        OsString::from("--json"),
    ];
    let output = run(stado, &args)?;
    if !output.status.success() {
        return Err(command_error(stado, &format!("checking {uri}"), &output));
    }
    let state: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        Error(format!(
            "stado storage stat {uri} did not answer JSON: {error}"
        ))
    })?;
    Ok(state.get("state").and_then(Value::as_str) == Some("present"))
}

/// Upload unless the immutable coordinate already holds the object.
fn publish(stado: &Path, uri: &str, path: &Path) -> Result<()> {
    match present(stado, uri)? {
        true => Ok(()),
        false => upload(stado, uri, path, CONTENT_TYPE),
    }
}

/// Stream `paths` in order through one SHA-256; returns digest and byte count.
fn hash_files(paths: &[PathBuf]) -> Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut bytes = 0;
    for path in paths {
        let mut file = File::open(path)
            .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
        bytes += io::copy(&mut file, &mut hasher)?;
    }
    Ok((hex::encode(hasher.finalize()), bytes))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .map_err(|error| Error(format!("cannot read {}: {error}", path.display())))?;
    serde_json::from_str(&text).map_err(|error| Error(format!("{}: {error}", path.display())))
}

fn text<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| Error(format!("model manifest has no {pointer}")))
}

/// Fetch a qualified job output from `source`, verify every file against the
/// manifest and the passed final judge, check that the ordered parts rebuild
/// the qualified artifact, and publish it under its digest. Objects already
/// present at the coordinate are left as they are.
pub(crate) fn publish_release(model: &ReleaseModel, source: &str) -> Result<Value> {
    let stado = stado_bin();
    let source = source.trim_end_matches('/');
    let work = TempDir::create()?;
    let root = &work.0;
    let manifest_path = root.join("model-manifest.json");
    fetch(
        &stado,
        &format!("{source}/model-manifest.json"),
        &manifest_path,
    )?;
    let manifest = read_json(&manifest_path)?;
    if manifest.get("qualified") != Some(&Value::Bool(true)) {
        crate::bail!("{} model manifest is not qualified", model.name);
    }
    if manifest
        .get("required_quality_gate")
        .and_then(Value::as_str)
        != Some(GATE)
    {
        crate::bail!("{} model manifest does not require {GATE}", model.name);
    }
    if let Some(contract) = model.contract {
        if manifest.get("contract").and_then(Value::as_str) != Some(contract) {
            crate::bail!(
                "{} model manifest is not the {contract} contract",
                model.name
            );
        }
    }
    let parts: Vec<String> = manifest
        .pointer("/transport/parts")
        .and_then(Value::as_array)
        .ok_or_else(|| Error("model manifest has no /transport/parts".to_string()))?
        .iter()
        .map(|part| part.as_str().map(str::to_string))
        .collect::<Option<_>>()
        .ok_or_else(|| Error("model manifest names a part that is not a string".to_string()))?;
    let names: Vec<&str> = model
        .metadata
        .iter()
        .copied()
        .chain(parts.iter().map(String::as_str))
        .collect();
    for name in &names {
        let path = root.join(name);
        fetch(&stado, &format!("{source}/{name}"), &path)?;
        let expected = text(&manifest, &format!("/files/{name}/sha256"))?;
        let (actual, _) = hash_files(&[path])?;
        if actual != expected {
            crate::bail!("sha256 mismatch for {name}: expected {expected}, found {actual}");
        }
    }
    let judge = read_json(&root.join(GATE))?;
    if judge.get("passed") != Some(&Value::Bool(true)) {
        crate::bail!("{} final judge did not pass", model.name);
    }
    if judge
        .get("complete")
        .is_some_and(|complete| complete != &Value::Bool(true))
    {
        crate::bail!("{} final judge is incomplete", model.name);
    }
    let part_paths: Vec<PathBuf> = parts.iter().map(|name| root.join(name)).collect();
    let (digest, bytes) = hash_files(&part_paths)?;
    let expected_bytes = manifest
        .pointer("/transport/assembled_bytes")
        .and_then(Value::as_u64);
    if digest != text(&manifest, "/transport/assembled_sha256")? || expected_bytes != Some(bytes) {
        crate::bail!("ordered model parts do not reconstruct the qualified artifact");
    }
    let base = format!("{}/{digest}", model.destination);
    for name in &parts {
        publish(
            &stado,
            &format!("{base}/large-output/{name}"),
            &root.join(name),
        )?;
    }
    for name in model.metadata {
        publish(&stado, &format!("{base}/{name}"), &root.join(name))?;
    }
    let chunk_manifest = json!({
        "filename": text(&manifest, "/default_artifact")?,
        "sha256": digest,
        "bytes": bytes,
        "part_count": parts.len(),
        "parts": parts,
    });
    let chunk_path = root.join("large-output-manifest.json");
    fs::write(
        &chunk_path,
        serde_json::to_string_pretty(&chunk_manifest)? + "\n",
    )?;
    publish(
        &stado,
        &format!("{base}/large-output/manifest-v2.json"),
        &chunk_path,
    )?;
    publish(
        &stado,
        &format!("{base}/model-manifest-v2.json"),
        &manifest_path,
    )?;
    Ok(json!({"base": base, "digest": digest, "parts": parts.len()}))
}
