use super::*;

pub const DEFAULT_STADO_BIN: &str = "stado";

/// Brama's delegating alias: the strongest operator subscription the caller's
/// signed identity may use. Which model answers is Brama's route table's
/// business, never a provider id written here.
pub const BEST_MODEL: &str = "best";

/// The default teacher. Label work is paid by an operator subscription, so it
/// asks the same alias the reviewer does; override it with --brama-model or
/// --teacher-model, or `judge.model` in a job file.
pub const DEFAULT_MODEL: &str = BEST_MODEL;

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

/// The vault role and field an environment variable declares as `ROLE#FIELD`.
/// A role, not an item: the item playing it can be replaced or renamed
/// without anything here changing. `holds` says what the value is for, so the
/// refusal tells the operator what to declare.
pub(crate) fn role_reference(variable: &str, holds: &str) -> Result<(String, String)> {
    let reference = env_trimmed(variable);
    match reference.split_once('#') {
        Some((role, field)) if !role.is_empty() && !field.is_empty() => {
            Ok((role.to_string(), field.to_string()))
        }
        _ => bail!(
            "{variable} must declare the vault role holding {holds} and its field as ROLE#FIELD \
             (it is {reference:?}); the value is read with `stado credentials get --role ROLE \
             --field FIELD`"
        ),
    }
}

/// Read the secret behind a `ROLE#FIELD` environment variable through Stado,
/// which selects the one item playing that role, as Stado's own identity.
/// Secret values are held in memory only. A scoped consumer named only by
/// environment could not resolve a role (it cannot list the vault), so no
/// consumer override is taken here.
pub(crate) fn read_role(variable: &str, holds: &str) -> Result<String> {
    let (role, field) = role_reference(variable, holds)?;
    let stado = match env_trimmed("TLT_STADO_BIN") {
        value if value.is_empty() => DEFAULT_STADO_BIN.to_string(),
        value => value,
    };
    let command = format!("{stado} credentials get --role {role} --field {field}");
    match run_capture(
        &stado,
        &["credentials", "get", "--role", &role, "--field", &field],
        &[],
    ) {
        Some((true, stdout)) if !stdout.trim().is_empty() => Ok(stdout.trim().to_string()),
        Some((true, _)) => bail!("{variable}: `{command}` answered an empty value"),
        Some((false, _)) => bail!(
            "{variable}: `{command}` was refused; run it to read Stado's reason (no item plays \
             the role, several do, or this consumer is not granted it)"
        ),
        None => bail!(
            "{variable}: {stado} could not be started; set TLT_STADO_BIN to the stado executable"
        ),
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
