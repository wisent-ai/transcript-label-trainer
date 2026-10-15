//! A quantized model served the way production serves it: `llama-server` on a
//! loopback port the system assigns, asked over its OpenAI chat route. Every
//! GGUF evaluation (the goal, lifecycle and humanizer models) starts it
//! through here, so there is one harness and one set of waiting rules.
//!
//! Nothing here waits by the clock. The server is ready when its health
//! answer is ready; between probes the harness reads the server's next log
//! line, so it waits on the server's own progress, and a server that exits
//! first is a failure naming its status and log. Each request waits for its
//! answer and says so on stderr; a request that fails is the caller's
//! failure, never retried.

mod decoding;
pub(crate) use decoding::Decoding;

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;

use serde_json::{json, Value};

use crate::util::{Error, Result};

/// Which server serves which model, how it decodes, and where its log goes.
/// How many slots it runs, how much context it holds and how many layers it
/// offloads are llama-server's own to size from the model and the device:
/// its documented defaults are `--parallel -1` (auto), `--ctx-size 0`
/// (loaded from the model, fitted to device memory by `--fit on`) and
/// `--gpu-layers auto`
/// (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
pub(crate) struct Serving {
    pub(crate) server: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) server_log: PathBuf,
    pub(crate) decoding: Decoding,
}

/// A running server: the chat endpoint, the alias every request names, the
/// slots it chose (its `/props` `total_slots`), how it decodes, and a client
/// that waits for each answer.
pub(crate) struct Server {
    child: Child,
    pub(crate) endpoint: String,
    pub(crate) alias: String,
    pub(crate) slots: usize,
    decoding: Decoding,
    pub(crate) client: reqwest::blocking::Client,
}

impl Server {
    /// Start `serving` and return once it answers its health check and has
    /// said how many slots it runs. A server or model file that does not
    /// exist is refused before anything starts.
    pub(crate) fn start(serving: &Serving) -> Result<Self> {
        for (what, path) in [("model", &serving.model), ("server", &serving.server)] {
            if !path.is_file() {
                return Err(Error(format!("the served {what} {} does not exist", path.display())));
            }
        }
        let alias = alias(&serving.model)?;
        let client = waiting_client()?;
        let port = free_port()?;
        let mut child = start_server(serving, port, &alias)?;
        let slots = match served_slots(&client, port) {
            Ok(slots) => slots,
            Err(error) => return Err(Error(format!("{error}; {}", end(&mut child)))),
        };
        Ok(Self {
            child,
            endpoint: format!("http://{}:{port}/v1/chat/completions", Ipv4Addr::LOCALHOST),
            alias,
            slots,
            decoding: serving.decoding,
            client,
        })
    }

    /// Stop the server once every request has its answer, and say how its
    /// end went.
    pub(crate) fn stop(mut self) {
        let ended = end(&mut self.child);
        eprintln!("transcript-label-trainer: {ended}");
    }

    /// Ask for the answer to one conversation — `system` then `user` — with
    /// thinking off and the serving's decoding, constrained by `constraint`
    /// when one is given (the request field llama-server reads it from, and
    /// its value), and return the answer's text. `id` names the request in a
    /// failure.
    pub(crate) fn answer(&self, id: &str, system: &str, user: &str, constraint: Option<(&str, Value)>) -> Result<String> {
        let mut body = json!({
            "model": self.alias,
            "messages": [{ "role": "system", "content": system }, { "role": "user", "content": user }],
            "stream": false,
            "chat_template_kwargs": { "enable_thinking": false },
        });
        for (field, value) in self.decoding.fields() {
            body[field] = value;
        }
        if let Some((field, value)) = constraint {
            body[field] = value;
        }
        let response = stado_wait::request_blocking(self.client.post(&self.endpoint).json(&body))
            .map_err(|error| Error(format!("{id} inference failed: {error}")))?;
        let status = response.status();
        let text = response.text().map_err(|error| Error(format!("{id} answered {status} but its body could not be read: {error}")))?;
        if !status.is_success() {
            return Err(Error(format!("{id} inference failed at {} with {status}: {text}", self.endpoint)));
        }
        let answer: Value = serde_json::from_str(&text).map_err(|error| Error(format!("{id} answered {status} without JSON: {error}; body: {text}")))?;
        let choice = answer
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| Error(format!("{id} answered {status} without a choice: {answer}")))?;
        if matches!(
            choice.get("finish_reason").and_then(Value::as_str),
            Some("length" | "max_tokens")
        ) {
            return Err(Error(format!("{id} answered {status} with a truncated completion: {answer}")));
        }
        let content = choice
            .pointer("/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| Error(format!("{id} answered {status} without a message: {answer}")))?
            .trim();
        if content.is_empty() {
            return Err(Error(format!("{id} answered {status} with an empty message: {answer}")));
        }
        Ok(content.to_owned())
    }
}

/// Wait for the server to end, saying so on stderr.
fn reap(child: &mut Child) -> std::io::Result<ExitStatus> {
    let pid = child.id();
    stado_wait::blocking(
        stado_wait::Kind::Process,
        format!("llama-server pid {pid} to end"),
        format!("pid {pid}"),
        || child.wait(),
    )
}

/// Kill the server and reap it, and say how that went: a kill that failed or
/// a reap that failed is named, never dropped, because a server left running
/// keeps its port and its device memory.
fn end(child: &mut Child) -> String {
    let pid = child.id();
    let killed = child.kill();
    match (killed, reap(child)) {
        (Ok(()), Ok(status)) => format!("llama-server pid {pid} stopped ({status})"),
        (Err(kill), Ok(status)) => format!("llama-server pid {pid} had already ended ({status}); kill answered {kill}"),
        (Ok(()), Err(failed)) => format!("llama-server pid {pid} was killed but could not be reaped: {failed}"),
        (Err(kill), Err(failed)) => format!("llama-server pid {pid} could not be killed ({kill}) or reaped ({failed}); it may still hold its port"),
    }
}

/// The slots the server chose, from its `/props` answer's `total_slots`.
fn served_slots(client: &reqwest::blocking::Client, port: u16) -> Result<usize> {
    let props = format!("http://{}:{port}/props", Ipv4Addr::LOCALHOST);
    let response = stado_wait::request_blocking(client.get(&props))
        .map_err(|error| Error(format!("llama-server request to {props} failed: {error}")))?;
    let status = response.status();
    let text = response.text().map_err(|error| {
        Error(format!("llama-server {props} answered {status}, but its body could not be read: {error}"))
    })?;
    if !status.is_success() {
        return Err(Error(format!("llama-server {props} refused with {status}: {text}")));
    }
    let answer: Value = serde_json::from_str(&text).map_err(|error| {
        Error(format!("llama-server {props} answered {status} without valid JSON: {error}; body: {text}"))
    })?;
    let stated = answer
        .get("total_slots")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error(format!("llama-server's {props} states no total_slots: {answer}")))?;
    let slots = usize::try_from(stated).map_err(|error| {
        Error(format!("llama-server's {props} states {stated} slots, more than this host can address: {error}"))
    })?;
    std::num::NonZeroUsize::new(slots)
        .map(std::num::NonZeroUsize::get)
        .ok_or_else(|| Error(format!("llama-server's {props} states no positive total_slots: {answer}")))
}

/// The served alias every request names: the model file's name without its
/// extension. A model path without a file name is refused.
fn alias(model: &Path) -> Result<String> {
    model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .ok_or_else(|| Error(format!("the served model {} names no file", model.display())))
}

/// An HTTP client that waits for the server's answer: reqwest's blocking
/// client otherwise gives up on its own after a cut-off it chose.
fn waiting_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder().timeout(None).build().map_err(|error| Error(format!("the HTTP client could not be built: {error}")))
}

/// A loopback port the operating system has free right now.
fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?.local_addr()?.port()) // port 0 asks the system for a free port: https://doc.rust-lang.org/std/net/struct.TcpListener.html#method.bind
}

/// Whether the server answers its health route as ready. Before it listens,
/// the connection is refused, which is the state being waited out, not a
/// failure; any other failure to ask is.
fn healthy(client: &reqwest::blocking::Client, health: &str) -> Result<bool> {
    match stado_wait::request_blocking(client.get(health)) {
        Ok(response) => Ok(response.status().is_success()),
        Err(error) if error.is_connect() => Ok(false),
        Err(error) => Err(Error(format!("llama-server health {health} could not be asked: {error}"))),
    }
}

/// Start the server and return once its health answer is ready.
fn start_server(run: &Serving, port: u16, alias: &str) -> Result<Child> {
    let mut log = fs::File::create(&run.server_log).map_err(|error| {
        Error(format!("cannot create server stderr log {}: {error}", run.server_log.display()))
    })?;
    let stdout_log = run.server_log.with_extension("stdout.log");
    let mut copy_out = fs::File::create(&stdout_log).map_err(|error| {
        Error(format!("cannot create server stdout log {}: {error}", stdout_log.display()))
    })?;
    let client = waiting_client()?;
    let mut child = Command::new(&run.server)
        .arg("--model")
        .arg(&run.model)
        .args(["--host", &Ipv4Addr::LOCALHOST.to_string(), "--port", &port.to_string(), "--alias", alias])
        .args(["--no-webui"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| Error(format!("{} could not start: {error}", run.server.display())))?;
    let stderr = child.stderr.take().ok_or("llama-server has no stderr")?;
    let stdout = child.stdout.take().ok_or("llama-server has no stdout")?;
    thread::spawn(move || {
        if let Err(error) = std::io::copy(&mut BufReader::new(stdout), &mut copy_out) {
            eprintln!("llama-server's stdout log {} stops here: {error}", stdout_log.display());
        }
    });
    let health = format!("http://{}:{port}/health", Ipv4Addr::LOCALHOST);
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match healthy(&client, &health) {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) => return Err(Error(format!("{}; {}", error.0, end(&mut child)))),
        }
        match lines.next() {
            Some(line) => writeln!(log, "{}", line?)?,
            None => {
                let status = reap(&mut child)?;
                return Err(Error(format!(
                    "llama-server exited {status} before it was ready; its log is {}",
                    run.server_log.display()
                )));
            }
        }
    }
    let server_log = run.server_log.clone();
    // The first failure ends the forwarder and is said: silent, the log just
    // stopped and the server looked quiet.
    thread::spawn(move || {
        for line in lines {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("llama-server's stderr can no longer be read; {} stops here: {error}", server_log.display());
                    return;
                }
            };
            if let Err(error) = writeln!(log, "{line}") {
                eprintln!("llama-server's log {} can no longer be written: {error}", server_log.display());
                return;
            }
        }
    });
    Ok(child)
}
