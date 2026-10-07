//! A quantized model served the way production serves it: `llama-server` on a
//! loopback port the system assigns, asked over its OpenAI chat route. Every
//! GGUF evaluation (the lifecycle and goal models) starts it through here, so
//! there is one harness and one set of waiting rules.
//!
//! Nothing here waits by the clock. The server is ready when its health
//! answer is ready; between probes the harness reads the server's next log
//! line, so it waits on the server's own progress, and a server that exits
//! first is a failure naming its status and log. Each request waits for its
//! answer; a request that fails is the caller's failure, never retried on a
//! guessed schedule.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;

use serde_json::{json, Value};

use crate::util::{Error, Result};

/// How the model is served; every field is the caller's.
pub(crate) struct Serving {
    pub(crate) server: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) server_log: PathBuf,
    pub(crate) parallel: usize,
    pub(crate) slot_context: usize,
    pub(crate) gpu_layers: String,
}

/// A running server: the chat endpoint, the alias every request names, and
/// a client that waits for each answer.
pub(crate) struct Server {
    child: Child,
    pub(crate) endpoint: String,
    pub(crate) alias: String,
    pub(crate) client: reqwest::blocking::Client,
}

impl Server {
    /// Start `serving` and return once it answers its health check. A server
    /// or model file that does not exist is refused before anything starts;
    /// the slot and context counts are the caller's, stated at least one on
    /// the command line.
    pub(crate) fn start(serving: &Serving) -> Result<Self> {
        for (what, path) in [("model", &serving.model), ("server", &serving.server)] {
            if !path.is_file() {
                return Err(Error(format!("the served {what} {} does not exist", path.display())));
            }
        }
        let port = free_port()?;
        let child = start_server(serving, port)?;
        Ok(Self {
            child,
            endpoint: format!("http://{}:{port}/v1/chat/completions", Ipv4Addr::LOCALHOST),
            alias: alias(&serving.model),
            client: waiting_client()?,
        })
    }

    /// Stop the server once every request has its answer.
    pub(crate) fn stop(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Ask for the answer to one conversation — `system` then `user` — with
    /// thinking off and greedy decoding, constrained by `constraint` (the
    /// request field llama-server reads it from, and its value), and return
    /// the answer's text. `id` names the request in a failure.
    pub(crate) fn answer(&self, id: &str, system: &str, user: &str, constraint: (&str, Value)) -> Result<String> {
    let mut body = json!({
        "model": self.alias,
        "messages": [{ "role": "system", "content": system }, { "role": "user", "content": user }],
        "temperature": 0,
        "stream": false,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    body[constraint.0] = constraint.1;
    let response = self.client.post(&self.endpoint).json(&body).send().map_err(|error| Error(format!("{id} inference failed: {error}")))?;
    let status = response.status();
    let answer: Value = response.json().map_err(|error| Error(format!("{id} answered {status} without JSON: {error}")))?;
    Ok(answer.pointer("/choices/0/message/content").and_then(Value::as_str).ok_or_else(|| Error(format!("{id} answered {status} without a message: {answer}")))?.trim().to_string())
    }
}

/// The served alias every request names.
fn alias(model: &Path) -> String {
    model.file_stem().map(|stem| stem.to_string_lossy().into_owned()).unwrap_or_default()
}

/// An HTTP client that waits for the server's answer: reqwest's blocking
/// client otherwise gives up after its own 30 s.
fn waiting_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder().timeout(None).build().map_err(|error| Error(format!("the HTTP client could not be built: {error}")))
}

/// A loopback port the operating system has free right now.
fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?.local_addr()?.port())
}

/// Start the server and return once its health answer is ready.
fn start_server(run: &Serving, port: u16) -> Result<Child> {
    let mut child = Command::new(&run.server)
        .arg("--model")
        .arg(&run.model)
        .args(["--host", &Ipv4Addr::LOCALHOST.to_string(), "--port", &port.to_string(), "--alias", &alias(&run.model)])
        .args(["--ctx-size", &(run.slot_context * run.parallel).to_string(), "--gpu-layers", &run.gpu_layers])
        .args(["--parallel", &run.parallel.to_string(), "--no-webui"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error(format!("{} could not start: {error}", run.server.display())))?;
    let mut log = fs::File::create(&run.server_log)?;
    let stderr = child.stderr.take().ok_or("llama-server has no stderr")?;
    let stdout = child.stdout.take().ok_or("llama-server has no stdout")?;
    let mut copy_out = fs::File::create(run.server_log.with_extension("stdout.log"))?;
    thread::spawn(move || std::io::copy(&mut BufReader::new(stdout), &mut copy_out));
    let health = format!("http://{}:{port}/health", Ipv4Addr::LOCALHOST);
    let client = waiting_client()?;
    let mut lines = BufReader::new(stderr).lines();
    loop {
        if client.get(&health).send().map(|response| response.status().is_success()).unwrap_or(false) {
            thread::spawn(move || {
                for line in lines.map_while(std::result::Result::ok) {
                    let _ = writeln!(log, "{line}");
                }
            });
            return Ok(child);
        }
        match lines.next() {
            Some(line) => writeln!(log, "{}", line?)?,
            None => {
                let status = child.wait()?;
                return Err(Error(format!("llama-server exited {status} before it was ready; its log is {}", run.server_log.display())));
            }
        }
    }
}

