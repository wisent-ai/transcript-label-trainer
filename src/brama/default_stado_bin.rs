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

/// The value an environment variable holds, trimmed; `None` only when it is
/// unset. A set but empty value, or one that is not UTF-8, is refused by
/// name: read as unset, it would send the command to another source.
pub(crate) fn env_value(key: &str) -> Result<Option<String>> {
    match std::env::var(key) {
        Ok(value) if value.trim().is_empty() => bail!("{key} is set but empty; set it or unset it"),
        Ok(value) => Ok(Some(value.trim().to_string())),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("{key} is set but is not UTF-8"),
    }
}

/// The stado executable: `TLT_STADO_BIN` when it names one, otherwise the
/// program named stado, found on PATH when it runs.
fn stado_bin() -> Result<String> {
    Ok(match env_value("TLT_STADO_BIN")? {
        Some(stado) => stado,
        None => DEFAULT_STADO_BIN.to_string(),
    })
}

/// Where Stado says Brama is actually placed.
///
/// `stado service directory connect brama` derives the address from the
/// placement rather than from a name someone wrote down, and verifies that
/// something answers there before reporting it. Every way the answer fails
/// is an error with its cause: stado not run, Stado's refusal with its
/// reason, a reply that is not JSON, a reply without a url.
pub(crate) fn stado_url() -> Result<String> {
    let stado = stado_bin()?;
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

/// Where Brama is: `BRAMA_URL` when the caller names it, otherwise where
/// Stado's service directory places the gateway, the one record that follows
/// it when it moves.
pub(crate) fn resolve_url() -> Result<String> {
    if let Some(url) = env_value("BRAMA_URL")? {
        return Ok(url);
    }
    stado_url().map_err(|cause| {
        Error(format!("no Brama endpoint: BRAMA_URL is unset and Stado could not place Brama: {cause}"))
    })
}

/// The vault role and field an environment variable declares as `ROLE#FIELD`.
/// A role, not an item: the item playing it can be replaced or renamed
/// without anything here changing. `holds` says what the value is for, so the
/// refusal tells the operator what to declare.
pub(crate) fn role_reference(variable: &str, holds: &str) -> Result<(String, String)> {
    let reference = env_value(variable)?.ok_or_else(|| {
        Error(format!(
            "{variable} is not set: declare the vault role holding {holds} and its field as ROLE#FIELD"
        ))
    })?;
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
    let stado = stado_bin()?;
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
