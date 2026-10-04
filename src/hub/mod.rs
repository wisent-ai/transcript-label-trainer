//! The private Hugging Face repository every qualified model of this trainer
//! is published to: one client for metadata reads, digest checks and the
//! supported `hf` CLI uploads, shared by the humanizer adapter and the
//! goal-lifecycle model. A model is served from the host that runs it, which
//! fetches the immutable revision into its own model cache; the fleet's
//! release store is not a model store.
use std::ffi::OsString;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use reqwest::blocking::Client;
use reqwest::{StatusCode, Url};
use serde::Deserialize;

use crate::util::{Error, Result};

mod files;
pub(crate) use files::Signature;

#[derive(Deserialize)]
pub(crate) struct ModelInfo {
    pub private: bool,
    pub sha: Option<String>,
    #[serde(default)]
    pub siblings: Vec<RemoteFile>,
}

#[derive(Deserialize)]
pub(crate) struct RemoteFile {
    pub rfilename: String,
    pub size: Option<u64>,
    pub lfs: Option<LfsObject>,
}

#[derive(Deserialize)]
pub(crate) struct LfsObject {
    sha256: String,
}

pub(crate) struct Hub {
    client: Client,
    endpoint: Url,
    executable: OsString,
    token: String,
}

impl Hub {
    pub fn new() -> Result<Self> {
        let token = std::env::var("HF_TOKEN")
            .map_err(|error| Error(format!("cannot read required HF_TOKEN: {error}")))?;
        if token.trim().is_empty() {
            return Err(Error("HF_TOKEN is empty".into()));
        }
        let endpoint = match std::env::var("HF_ENDPOINT") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "https://huggingface.co".to_string(),
            Err(error) => return Err(Error(format!("cannot read HF_ENDPOINT: {error}"))),
        };
        let endpoint = Url::parse(&endpoint)
            .map_err(|error| Error(format!("invalid HF_ENDPOINT: {error}")))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Error("HF_ENDPOINT must be an HTTP base URL without a query or fragment".into()));
        }
        let client = Client::builder()
            .build()
            .map_err(|error| Error(format!("cannot initialize Hugging Face client: {error}")))?;
        Ok(Self {
            client,
            endpoint,
            executable: std::env::var_os("HF_BIN").unwrap_or_else(|| "hf".into()),
            token,
        })
    }

    pub fn info(&self, repo: &str, revision: Option<&str>, files: bool) -> Result<Option<ModelInfo>> {
        let (owner, name) = repo.split_once('/').ok_or_else(|| {
            Error("Hugging Face repository must explicitly name OWNER/REPOSITORY".into())
        })?;
        for component in [owner, name] {
            if component.is_empty()
                || component == "."
                || component == ".."
                || !component.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            {
                return Err(Error(format!("invalid Hugging Face repository: {repo:?}")));
            }
        }
        let mut url = self.endpoint.clone();
        {
            let mut segments = url.path_segments_mut()
                .map_err(|_| Error("HF_ENDPOINT cannot hold API paths".into()))?;
            segments.pop_if_empty().extend(["api", "models", owner, name]);
            if let Some(revision) = revision {
                segments.extend(["revision", revision]);
            }
        }
        if files {
            url.query_pairs_mut().append_pair("blobs", "true");
        }
        let response = self.client.get(url).bearer_auth(self.token.trim()).send()
            .map_err(|error| Error(format!("Hugging Face model_info {repo} failed: {error}")))?;
        let status = response.status();
        let repository_missing = response.headers().get("x-error-code")
            .and_then(|value| value.to_str().ok()) == Some("RepoNotFound");
        if status == StatusCode::NOT_FOUND || (status == StatusCode::UNAUTHORIZED && repository_missing) {
            return Ok(None);
        }
        if !status.is_success() {
            let detail = response.text().map_err(|error| Error(format!(
                "Hugging Face model_info {repo}: HTTP {status}; cannot read response: {error}"
            )))?;
            return Err(Error(format!("Hugging Face model_info {repo}: HTTP {status}: {}", detail.trim())));
        }
        response.json().map(Some)
            .map_err(|error| Error(format!("Hugging Face model_info {repo} returned invalid metadata: {error}")))
    }

    pub fn verify(&self, repo: &str, revision: &str, expected: &Signature, file: &RemoteFile) -> Result<()> {
        if file.size != Some(expected.size) {
            return Err(Error(format!(
                "published file {repo}@{revision}/{}: expected {} bytes, observed {:?}",
                file.rfilename, expected.size, file.size
            )));
        }
        let observed: std::borrow::Cow<'_, str> = if let Some(object) = &file.lfs {
            // LFS metadata names the content-addressed object. Do not download
            // model weights again merely to calculate the same digest.
            object.sha256.as_str().into()
        } else {
            let mut url = self.endpoint.clone();
            url.path_segments_mut()
                .map_err(|_| Error("HF_ENDPOINT cannot hold file paths".into()))?
                .pop_if_empty().extend(repo.split('/'))
                .extend(["resolve", revision]).extend(file.rfilename.split('/'));
            let mut response = self.client.get(url).bearer_auth(self.token.trim()).send()
                .map_err(|error| Error(format!("cannot read published file {}: {error}", file.rfilename)))?;
            let status = response.status();
            if !status.is_success() {
                let detail = response.text().map_err(|error| Error(format!(
                    "cannot read published file {}: HTTP {status}; cannot read response: {error}",
                    file.rfilename
                )))?;
                return Err(Error(format!("cannot read published file {}: HTTP {status}: {detail}", file.rfilename)));
            }
            let signature = Signature::stream(&mut response)
                .map_err(|error| Error(format!("cannot hash published file {}: {error}", file.rfilename)))?;
            if signature.size != expected.size {
                return Err(Error(format!("published file {} returned {} bytes, expected {}",
                    file.rfilename, signature.size, expected.size)));
            }
            std::borrow::Cow::Owned(signature.sha256)
        };
        if observed != expected.sha256 {
            return Err(Error(format!("published file {} checksum mismatch: expected {}, observed {observed}",
                file.rfilename, expected.sha256)));
        }
        Ok(())
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command.env("HF_TOKEN", self.token.trim())
            .env("HF_ENDPOINT", self.endpoint.as_str().trim_end_matches('/'))
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .stdout(Stdio::null());
        command
    }

    fn execute(operation: &str, repo: &str, mut command: Command) -> Result<()> {
        let output = command.output()
            .map_err(|error| Error(format!("cannot execute hf {operation} for {repo}: {error}")))?;
        if !output.status.success() {
            return Err(Error(format!(
                "hf {operation} for {repo} failed ({}): {}",
                output.status, String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        std::io::stderr().write_all(&output.stderr)?;
        Ok(())
    }

    pub fn create_private(&self, repo: &str) -> Result<()> {
        let mut command = self.command();
        command.args(["repo", "create", repo, "--repo-type", "model", "--private", "--exist-ok"]);
        Self::execute("repo create", repo, command)
    }

    pub fn upload(
        &self,
        repo: &str,
        branch: &str,
        source: &Path,
        destination: &str,
        message: &str,
    ) -> Result<()> {
        let mut command = self.command();
        command.args(["upload", repo]).arg(source).arg(destination).args([
            "--repo-type", "model", "--revision", branch, "--private", "--quiet",
            "--commit-message", message,
        ]);
        Self::execute("upload", repo, command)
    }
}
