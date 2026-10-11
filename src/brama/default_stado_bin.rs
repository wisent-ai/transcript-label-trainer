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

/// What a finished child process answered: whether it succeeded, and both
/// streams, so a refusal can be reported with the child's own reason.
pub(crate) struct Captured {
    pub(crate) success: bool,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Run a child process to its end and capture both streams, saying on
/// stderr what is waited for. A child that cannot be run is an error naming
/// the program and the cause.
pub(crate) fn run_capture(
    program: &str,
    args: &[&str],
    extra_env: &[(&str, String)],
) -> Result<Captured> {
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null());
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = stado_wait::output(&mut command)
        .map_err(|cause| Error(format!("{program} could not be run: {cause}")))?;
    Ok(Captured {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

pub(crate) fn env_trimmed(key: &str) -> String {
    std::env::var(key).unwrap_or_default().trim().to_string()
}

/// Where Stado says Brama is actually placed.
///
/// `stado service directory connect brama` derives the address from the
/// placement rather than from a name someone wrote down, and verifies that
/// something answers there before reporting it. Every way the answer fails
/// is an error with its cause: stado not run, Stado's refusal with its
/// reason, a reply that is not JSON, a reply without a url.
pub(crate) fn stado_url() -> Result<String> {
    let stado = match env_trimmed("TLT_STADO_BIN") {
        value if value.is_empty() => DEFAULT_STADO_BIN.to_string(),
        value => value,
    };
    let command = format!("{stado} service directory connect --json brama");
    let captured = run_capture(
        &stado,
        &["service", "directory", "connect", "--json", "brama"],
        &[],
    )?;
    if !captured.success {
        bail!("`{command}` was refused: {}", captured.stderr.trim())
    }
    let reply = captured.stdout.trim();
    let parsed: serde_json::Value = serde_json::from_str(reply).map_err(|cause| {
        Error(format!(
            "`{command}` answered a reply that is not JSON: {cause}; reply: {reply}"
        ))
    })?;
    match parsed
        .get("url")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
    {
        Some(url) if !url.is_empty() => Ok(url.to_string()),
        _ => bail!("`{command}` answered no url: {reply}"),
    }
}

pub(crate) fn resolve_url() -> Result<String> {
    for var in ["BRAMA_URL", "JEDEN_BRAMA_URL"] {
        let value = env_trimmed(var);
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let jeden_env = home_dir().join(".jeden").join(".env");
    match std::fs::read_to_string(&jeden_env) {
        Ok(contents) => {
            for line in contents.lines() {
                if let Some(value) = line.strip_prefix("BRAMA_URL=") {
                    let value = value.trim();
                    if !value.is_empty() {
                        return Ok(value.to_string());
                    }
                }
            }
        }
        Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => {}
        Err(cause) => bail!(
            "{} cannot be read for BRAMA_URL: {cause}",
            jeden_env.display()
        ),
    }
    // There used to be a public default here, `https://brama.wisent.com`. That
    // hostname resolves to the company website, so every call that reached this
    // line got a 404 page from Vercel and a parse error blaming Brama. The
    // fleet keeps the real placement in Stado's service directory, which is
    // also the one answer that follows the gateway when it moves.
    match stado_url() {
        Ok(url) => Ok(url),
        Err(cause) => {
            bail!("no Brama endpoint: BRAMA_URL is unset and Stado could not place Brama: {cause}")
        }
    }
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
    let captured = run_capture(
        &stado,
        &["credentials", "get", "--role", &role, "--field", &field],
        &[],
    )
    .map_err(|cause| {
        Error(format!(
            "{variable}: {cause}; TLT_STADO_BIN names the stado executable"
        ))
    })?;
    if !captured.success {
        bail!(
            "{variable}: `{command}` was refused: {}",
            captured.stderr.trim()
        )
    }
    let value = captured.stdout.trim();
    if value.is_empty() {
        bail!("{variable}: `{command}` answered an empty value")
    }
    Ok(value.to_string())
}

#[derive(Clone)]
pub struct BramaClient {
    pub url: String,
    pub agent_id: String,
    pub(crate) secret: String,
    pub(crate) token: Option<String>,
    pub(crate) http: reqwest::blocking::Client,
}
