//! Brama client: HMAC-signed OpenAI-compatible chat completions.
//!
//! Brama is Wisent's authenticated, provider-neutral LLM gateway. All LLM
//! inference goes through it — never direct provider keys.
//!
//! Auth contract mirrored from jeden (rust/model_router.rs::hmac_headers,
//! rust/control_plane/brama.rs, scripts/run-with-stado.sh):
//!
//! - POST {BRAMA_URL}/v1/chat/completions, OpenAI-compatible body;
//! - HMAC identity headers: x-agent-id, x-agent-timestamp (unix seconds),
//!   x-agent-body-sha256 (hex SHA-256 of the raw body, empty string for an
//!   empty body), x-agent-signature (hex HMAC-SHA256 of
//!   "{agent_id}:{timestamp}:{body_hash}" keyed by the shared secret);
//! - bearer Authorization when the gateway demands one (the fleet gateway does).
//!
//! Credential resolution mirrors jeden's launcher: environment first
//! (WISENT_APP_AGENT_AUTH_SECRET / BRAMA_TOKEN), then Skarbiec — the signing
//! secret is item `agent:wisent-app` field `value` (vault-first, stado
//! fallback), the bearer is item `jeden-model-router` field `token`. Secret
//! values are only ever held in memory — never printed, never logged.
//!
//! Endpoint resolution: BRAMA_URL, then JEDEN_BRAMA_URL, then the BRAMA_URL
//! line of jeden's own config (~/.jeden/.env — the same value jeden uses), then
//! Stado's service directory, which derives the address from where the gateway
//! is placed.
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::bail;
use crate::util::{home_dir, Error, Result};

pub const DEFAULT_STADO_BIN: &str = "stado";
pub const DEFAULT_AGENT_ITEM: &str = "agent:wisent-app";
pub const DEFAULT_TOKEN_ITEM: &str = "jeden-model-router";

// Being listed by /v1/models is not the same as being servable: that list is
// the public models.dev catalogue, several thousand ids wide, and on 2026-08-09
// only 59 of 6244 came back with `available: true` for this agent. The previous
// default, `302ai/claude-haiku-4-5`, was chosen off that list and answered 503
// `direct '302ai' credential is unavailable` on every call, because the fleet
// vault has never held a 302ai credential.
//
// This one is billed to an existing subscription rather than per-token credits,
// handles the mixed Polish/English transcripts, and was measured answering
// through this client. There is no free local fallback for label work:
// `wisent-backend/chat/primary` is an unrelated product route, it labelled
// 1,617 curriculum rows against the held-out convention before 2026-08-18, and
// naming it here is what made that look allowed. Override with --brama-model.
pub const DEFAULT_MODEL: &str = "codex/gpt-5.6-sol";
/// Strongest active operator subscription route exposed by Brama.
pub const BEST_MODEL: &str = "-best";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const STADO_TIMEOUT: Duration = Duration::from_secs(20);
const ANSWER_MAX_TOKENS: u32 = 64;

/// One OpenAI chat message. Field order is `role` then `content`, matching the
/// dicts the Python client built and hashed.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    fn new(role: &str, content: String) -> Self {
        Message {
            role: role.to_string(),
            content,
        }
    }
}

/// Run a child process, capture stdout, drain stderr, and optionally give up
/// after `timeout` the way `subprocess.run(..., timeout=)` does. `None` means
/// the process could not be started, could not be waited on, or timed out.
fn run_capture(
    program: &str,
    args: &[&str],
    extra_env: &[(&str, String)],
    timeout: Option<Duration>,
) -> Option<(bool, String)> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command.spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    let mut stderr = child.stderr.take()?;
    let collect = thread::spawn(move || {
        let mut buffer = String::new();
        let _ = stdout.read_to_string(&mut buffer);
        buffer
    });
    // Drained so a chatty child cannot deadlock on a full stderr pipe.
    thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = stderr.read_to_end(&mut sink);
    });
    let status = match timeout {
        None => child.wait().ok()?,
        Some(limit) => {
            let deadline = Instant::now() + limit;
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => break status,
                    Ok(None) => {
                        if Instant::now() >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            return None;
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => return None,
                }
            }
        }
    };
    Some((status.success(), collect.join().ok()?))
}

fn env_trimmed(key: &str) -> String {
    std::env::var(key).unwrap_or_default().trim().to_string()
}

/// Where Stado says Brama is actually placed, or an empty string.
///
/// `stado service directory connect brama` derives the address from the
/// placement rather than from a name someone wrote down, and verifies that
/// something answers there before reporting it.
fn stado_url() -> String {
    let stado = match env_trimmed("TLT_STADO_BIN") {
        value if value.is_empty() => DEFAULT_STADO_BIN.to_string(),
        value => value,
    };
    let Some((ok, stdout)) = run_capture(
        &stado,
        &["service", "directory", "connect", "--json", "brama"],
        &[],
        Some(STADO_TIMEOUT),
    ) else {
        return String::new();
    };
    if !ok {
        return String::new();
    }
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&stdout) else {
        return String::new();
    };
    parsed
        .get("url")
        .and_then(|url| url.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn resolve_url() -> Result<String> {
    for var in ["BRAMA_URL", "JEDEN_BRAMA_URL"] {
        let value = env_trimmed(var);
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let jeden_env = home_dir().join(".jeden").join(".env");
    if jeden_env.is_file() {
        if let Ok(contents) = std::fs::read_to_string(&jeden_env) {
            for line in contents.lines() {
                if let Some(value) = line.strip_prefix("BRAMA_URL=") {
                    let value = value.trim();
                    if !value.is_empty() {
                        return Ok(value.to_string());
                    }
                }
            }
        }
    }
    // There used to be a public default here, `https://brama.wisent.com`. That
    // hostname resolves to the company website, so every call that reached this
    // line got a 404 page from Vercel and a parse error blaming Brama. The
    // fleet keeps the real placement in Stado's service directory, which is
    // also the one answer that follows the gateway when it moves.
    let url = stado_url();
    if !url.is_empty() {
        return Ok(url);
    }
    bail!(
        "no Brama endpoint: set BRAMA_URL, or make `stado service directory \
         connect brama` resolve (it reports where the gateway is placed)"
    )
}

/// Read one field of one Skarbiec item; empty string when unavailable.
///
/// Vault-first (the gateway verifies against the vault's current revision),
/// then the managed `stado credentials get` path as fallback — the same order
/// as jeden's run-with-stado.sh.
fn skarbiec_read(item: &str, field: &str) -> String {
    let home = home_dir();
    let vault = match env_trimmed("SKARBIEC_VAULT_FILE") {
        value if value.is_empty() => home
            .join(".stado")
            .join("skarbiec.vault.json")
            .to_string_lossy()
            .into_owned(),
        value => value,
    };
    let skarbiec = match env_trimmed("TLT_SKARBIEC_BIN") {
        value if value.is_empty() => home
            .join(".stado")
            .join("bin")
            .join("skarbiec")
            .to_string_lossy()
            .into_owned(),
        value => value,
    };
    if Path::new(&skarbiec).is_file() && Path::new(&vault).is_file() {
        let env = [("SKARBIEC_VAULT_FILE", vault)];
        if let Some((true, stdout)) = run_capture(&skarbiec, &["get", item], &env, None) {
            let value = serde_json::from_str::<serde_json::Value>(&stdout)
                .ok()
                .and_then(|parsed| parsed.get("fields").cloned())
                .and_then(|fields| fields.get(field).cloned())
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_default();
            let value = value.trim();
            if !value.is_empty() {
                return value.to_string();
            }
        }
    }
    let stado = match env_trimmed("TLT_STADO_BIN") {
        value if value.is_empty() => DEFAULT_STADO_BIN.to_string(),
        value => value,
    };
    let consumer = match env_trimmed("TLT_SKARBIEC_CONSUMER") {
        value if value.is_empty() => "local-operator".to_string(),
        value => value,
    };
    let token_file = match env_trimmed("TLT_SKARBIEC_TOKEN_FILE") {
        value if value.is_empty() => home
            .join(".stado")
            .join("local-operator-skarbiec-token")
            .to_string_lossy()
            .into_owned(),
        value => value,
    };
    let env = [
        ("WC_SKARBIEC_CONSUMER", consumer),
        ("WC_SKARBIEC_TOKEN_FILE", token_file),
    ];
    match run_capture(
        &stado,
        &["credentials", "get", "--field", field, item],
        &env,
        None,
    ) {
        Some((true, stdout)) => stdout.trim().to_string(),
        _ => String::new(),
    }
}

#[derive(Clone)]
pub struct BramaClient {
    pub url: String,
    pub agent_id: String,
    secret: String,
    token: Option<String>,
    http: reqwest::blocking::Client,
}

impl BramaClient {
    pub fn new(url: &str, agent_id: String, secret: String, token: Option<String>) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|error| Error(format!("could not build the Brama HTTP client: {error}")))?;
        Ok(BramaClient {
            url: url.trim_end_matches('/').to_string(),
            agent_id,
            secret,
            token: token.filter(|value| !value.is_empty()),
            http,
        })
    }

    pub fn from_env() -> Result<Self> {
        let agent_id = match env_trimmed("WISENT_APP_AGENT_ID") {
            value if value.is_empty() => "wisent-app".to_string(),
            value => value,
        };
        let mut secret = env_trimmed("WISENT_APP_AGENT_AUTH_SECRET");
        if secret.is_empty() {
            let item = match env_trimmed("TLT_BRAMA_AGENT_ITEM") {
                value if value.is_empty() => DEFAULT_AGENT_ITEM.to_string(),
                value => value,
            };
            secret = skarbiec_read(&item, "value");
        }
        if secret.is_empty() {
            bail!(
                "no Brama signing secret: set WISENT_APP_AGENT_AUTH_SECRET, or make \
                 Skarbiec item '{DEFAULT_AGENT_ITEM}' readable (vault-first via \
                 ~/.stado/bin/skarbiec, fallback 'stado credentials get')"
            )
        }
        let mut token = env_trimmed("BRAMA_TOKEN");
        if token.is_empty() {
            let item = match env_trimmed("TLT_BRAMA_TOKEN_ITEM") {
                value if value.is_empty() => DEFAULT_TOKEN_ITEM.to_string(),
                value => value,
            };
            token = skarbiec_read(&item, "token");
        }
        let url = resolve_url()?;
        Self::new(&url, agent_id, secret, Some(token))
    }

    /// The identity headers jeden signs with: the canonical string is
    /// `"{agent_id}:{timestamp}:{body_hash}"`, keyed by the shared secret.
    fn auth_headers(&self, body: &str) -> Vec<(&'static str, String)> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0)
            .to_string();
        let body_hash = if body.is_empty() {
            String::new()
        } else {
            hex::encode(Sha256::digest(body.as_bytes()))
        };
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(self.secret.as_bytes())
            .expect("HMAC accepts a key of any length");
        mac.update(format!("{}:{timestamp}:{body_hash}", self.agent_id).as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let mut headers = vec![
            ("content-type", "application/json".to_string()),
            ("x-agent-id", self.agent_id.clone()),
            ("x-agent-timestamp", timestamp),
            ("x-agent-body-sha256", body_hash),
            ("x-agent-signature", signature),
        ];
        if let Some(token) = &self.token {
            headers.push(("authorization", format!("Bearer {token}")));
        }
        headers
    }

    /// One non-streaming chat completion; returns the message content.
    pub fn chat(&self, model: &str, messages: &[Message]) -> Result<String> {
        let body = serde_json::to_string(&serde_json::json!({
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": ANSWER_MAX_TOKENS,
        }))?;
        let mut request = self.http.post(format!("{}/v1/chat/completions", self.url));
        for (name, value) in self.auth_headers(&body) {
            request = request.header(name, value);
        }
        let response = match request.body(body).send() {
            Ok(response) => response,
            Err(error) => bail!("Brama unreachable at {}: {error}", self.url),
        };
        let status = response.status();
        let text = response.text().unwrap_or_default();
        if status.as_u16() != 200 {
            let detail = truncate_chars(&text, 300);
            let detail = detail.trim();
            let detail = if detail.is_empty() {
                "(empty body)"
            } else {
                detail
            };
            bail!("Brama answered HTTP {}: {detail}", status.as_u16())
        }
        let content = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|payload| {
                payload
                    .get("choices")?
                    .get(0)?
                    .get("message")?
                    .get("content")?
                    .as_str()
                    .map(str::to_string)
            });
        match content {
            Some(content) => Ok(content.trim().to_string()),
            None => bail!(
                "Brama response was not an OpenAI chat completion: {}",
                truncate_chars(&text, 300)
            ),
        }
    }
}

/// The first `limit` characters, the way Python's `text[:limit]` slices.
pub fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}

/// Compact single-label classification prompt.
pub fn build_prompt(aspect: &str, values: &[String], text: &str) -> Vec<Message> {
    let allowed = values.join(", ");
    vec![
        Message::new(
            "system",
            "You classify coding-agent session transcripts. Answer with \
             exactly one of the allowed values and nothing else."
                .to_string(),
        ),
        Message::new(
            "user",
            format!(
                "Aspect: {aspect}\n\
                 Allowed values: {allowed}\n\n\
                 Classify this session transcript into one allowed value.\n\n\
                 Transcript:\n{text}"
            ),
        ),
    ]
}

/// Map a raw model answer to an allowed value.
///
/// `Some((value, exact))` where `exact == false` means the value was recovered
/// from a longer answer. `None` is a parse failure: apply nothing.
pub fn parse_answer<S: AsRef<str>>(answer: &str, values: &[S]) -> Option<(String, bool)> {
    let cleaned = answer
        .trim()
        .trim_matches(|c| c == '"' || c == '`' || c == '\'' || c == ' ' || c == '.');
    for value in values {
        let value = value.as_ref();
        if cleaned.eq_ignore_ascii_case(value) || cleaned.to_lowercase() == value.to_lowercase() {
            return Some((value.to_string(), true));
        }
    }
    let haystack = answer.to_lowercase();
    let mut found = values
        .iter()
        .map(AsRef::as_ref)
        .filter(|value| haystack.contains(&value.to_lowercase()));
    match (found.next(), found.next()) {
        (Some(only), None) => Some((only.to_string(), false)),
        _ => None,
    }
}
