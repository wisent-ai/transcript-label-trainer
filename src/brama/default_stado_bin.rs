use super::*;

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
pub const BEST_MODEL: &str = "best";

pub(crate) const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) const STADO_TIMEOUT: Duration = Duration::from_secs(20);

pub(crate) const ANSWER_MAX_TOKENS: u32 = 64;

/// One OpenAI chat message. Field order is `role` then `content`, matching the
/// dicts the Python client built and hashed.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub(crate) fn new(role: &str, content: String) -> Self {
        Message {
            role: role.to_string(),
            content,
        }
    }
}

/// Run a child process, capture stdout, drain stderr, and optionally give up
/// after `timeout` the way `subprocess.run(..., timeout=)` does. `None` means
/// the process could not be started, could not be waited on, or timed out.
pub(crate) fn run_capture(
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

pub(crate) fn env_trimmed(key: &str) -> String {
    std::env::var(key).unwrap_or_default().trim().to_string()
}

/// Where Stado says Brama is actually placed, or an empty string.
///
/// `stado service directory connect brama` derives the address from the
/// placement rather than from a name someone wrote down, and verifies that
/// something answers there before reporting it.
pub(crate) fn stado_url() -> String {
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

pub(crate) fn resolve_url() -> Result<String> {
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
pub(crate) fn skarbiec_read(item: &str, field: &str) -> String {
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
    pub(crate) secret: String,
    pub(crate) token: Option<String>,
    pub(crate) http: reqwest::blocking::Client,
}
