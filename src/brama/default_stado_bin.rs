use super::*;

pub const DEFAULT_STADO_BIN: &str = "stado";

pub const DEFAULT_AGENT_ITEM: &str = "agent:wisent-app";

pub const DEFAULT_TOKEN_ITEM: &str = "jeden-model-router";

// Catalogue membership is not proof that the configured agent can use a route.
// Label work uses an operator subscription through Brama, not an unrelated
// product route or a local fallback. Override the route with --brama-model.
pub const DEFAULT_MODEL: &str = "codex/gpt-5.6-sol";

/// Strongest active operator subscription route exposed by Brama.
pub const BEST_MODEL: &str = "best";

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

/// Run a child process, capture stdout and drain stderr. It ends when the
/// child exits; `None` means it could not be started or waited on.
pub(crate) fn run_capture(
    program: &str,
    args: &[&str],
    extra_env: &[(&str, String)],
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
    let status = child.wait().ok()?;
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
        if let Some((true, stdout)) = run_capture(&skarbiec, &["get", item], &env) {
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
