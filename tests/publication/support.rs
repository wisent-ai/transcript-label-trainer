use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use reqwest::blocking::Client;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;
pub const ENDPOINT: &str = "https://huggingface.co";
pub const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");

#[derive(Debug)]
pub struct Blocked(pub String);
impl fmt::Display for Blocked {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for Blocked {}

pub fn blocked<T>(message: impl Into<String>) -> Result<T> {
    Err(Box::new(Blocked(message.into())))
}

pub fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

pub fn digest(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut input, &mut hash)?;
    Ok(hex::encode(hash.finalize()))
}

pub struct Context {
    pub work: PathBuf,
    pub owner: String,
    pub commands: Vec<Value>,
    pub requests: Vec<Value>,
    client: Client,
    token: String,
    cache: PathBuf,
    hf_bin: std::ffi::OsString,
}

impl Context {
    /// The hf executable is chosen as the product's publisher chooses it:
    /// HF_BIN when set, refused when set but empty, and otherwise the `hf`
    /// command huggingface_hub documents
    /// (https://huggingface.co/docs/huggingface_hub/guides/cli).
    pub fn new(work: PathBuf, token: String) -> Result<Self> {
        let cache = work.join("hf-home");
        let hf_bin = match std::env::var_os("HF_BIN") {
            Some(value) if value.is_empty() => {
                return Err("HF_BIN is set but empty; name the hf executable or unset it".into())
            }
            Some(value) => value,
            None => "hf".into(),
        };
        Ok(Self {
            work,
            owner: String::new(),
            commands: Vec::new(),
            requests: Vec::new(),
            client: Client::builder().build()?,
            token,
            cache,
            hf_bin,
        })
    }

    fn request(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(StatusCode, String, bool)> {
        let mut request = self
            .client
            .request(method.clone(), format!("{ENDPOINT}{path}"))
            .bearer_auth(&self.token);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .map_err(|error| format!("{method} {path}: {error}"))?;
        let status = response.status();
        let code = match response.headers().get("x-error-code") {
            None => None,
            Some(value) => Some(value.to_str().map_err(|error| {
                format!("{method} {path}: HTTP {status} carries an x-error-code that is not text: {error}")
            })?),
        };
        let repository_missing = code == Some("RepoNotFound");
        self.requests
            .push(json!({"method": method.as_str(), "path": path,
            "status": status.as_u16(), "error_code": code}));
        let text = response
            .text()
            .map_err(|error| format!("{method} {path}: HTTP {status}; {error}"))?;
        Ok((status, text, repository_missing))
    }

    pub fn get(&mut self, path: &str) -> Result<Option<Value>> {
        let (status, body, repository_missing) = self.request(Method::GET, path, None)?;
        if status == StatusCode::NOT_FOUND
            || (status == StatusCode::UNAUTHORIZED && repository_missing)
        {
            return Ok(None);
        }
        require(
            status.is_success(),
            format!("GET {path}: HTTP {status}: {body}"),
        )?;
        Ok(Some(serde_json::from_str(&body)?))
    }

    pub fn info(&mut self, repo: &str, revision: Option<&str>) -> Result<Option<Value>> {
        let path = match revision {
            Some(revision) => format!("/api/models/{repo}/revision/{revision}?blobs=true"),
            None => format!("/api/models/{repo}?blobs=true"),
        };
        self.get(&path)
    }

    fn mutate(&mut self, method: Method, path: &str, body: &Value) -> Result<()> {
        let (status, text, _) = self.request(method.clone(), path, Some(body))?;
        require(
            status.is_success(),
            format!("{method} {path}: HTTP {status}: {text}"),
        )
    }

    pub fn with_repository(
        &mut self,
        private: Option<bool>,
        journey: impl FnOnce(&mut Self, &str) -> Result<Value>,
    ) -> Result<Value> {
        let name = format!("tlt-publication-{:x}", rand::random::<u128>());
        let repo = format!("{}/{name}", self.owner);
        require(
            self.info(&repo, None)?.is_none(),
            format!("refusing an existing fixture repository: {repo}"),
        )?;
        let body = json!({"organization": self.owner, "name": name, "type": "model"});
        let result = (|| {
            if let Some(private) = private {
                let mut creation = body.clone();
                creation["private"] = json!(private);
                self.mutate(Method::POST, "/api/repos/create", &creation)?;
            }
            journey(self, &repo)
        })();
        let cleanup = (|| -> Result<()> {
            if self.info(&repo, None)?.is_some() {
                self.mutate(Method::DELETE, "/api/repos/delete", &body)?;
            }
            require(
                self.info(&repo, None)?.is_none(),
                format!("fixture remains after cleanup: {repo}"),
            )
        })();
        match (result, cleanup) {
            (Ok(report), Ok(())) => Ok(report),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => {
                Err(format!("fixture cleanup failed for {repo}: {error}").into())
            }
            (Err(error), Err(cleanup)) => {
                Err(format!("{error}; fixture cleanup failed for {repo}: {cleanup}").into())
            }
        }
    }

    fn command(&self, executable: impl AsRef<OsStr>) -> Command {
        let mut command = Command::new(executable);
        command
            .current_dir(&self.work)
            .env("HF_TOKEN", &self.token)
            .env("HF_ENDPOINT", ENDPOINT)
            .env("HF_HOME", &self.cache)
            .env("HF_HUB_CACHE", self.cache.join("hub"))
            .env("HF_XET_CACHE", self.cache.join("xet"))
            .env("HF_ASSETS_CACHE", self.cache.join("assets"))
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1");
        command
    }

    pub fn product(&self) -> Command {
        self.command(BINARY)
    }

    pub fn hf(&self) -> Command {
        self.command(&self.hf_bin)
    }

    pub fn run(&mut self, label: &str, mut command: Command) -> Result<Output> {
        let argv: Vec<String> = std::iter::once(command.get_program())
            .chain(command.get_args())
            .map(|part| part.to_string_lossy().into_owned())
            .collect();
        let output = command
            .output()
            .map_err(|error| format!("cannot execute {label}: {error}"))?;
        let prefix = self.work.join(format!("{}-{label}", self.commands.len()));
        let stdout = prefix.with_extension("stdout");
        let stderr = prefix.with_extension("stderr");
        fs::write(&stdout, &output.stdout)?;
        fs::write(&stderr, &output.stderr)?;
        self.commands.push(json!({
            "argv": argv, "cwd": self.work, "exit_status": output.status.code(),
            "status": output.status.to_string(), "stdout": stdout, "stderr": stderr,
        }));
        Ok(output)
    }
}
