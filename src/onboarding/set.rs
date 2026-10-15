use super::*;

pub(crate) fn set(state: &mut Value, key: &str, value: Value) -> Result<()> {
    state
        .as_object_mut()
        .ok_or_else(|| Error("onboarding state is not an object".into()))?
        .insert(key.to_string(), value);
    Ok(())
}

pub(crate) fn string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Recorded progress for this machine, or a fresh attempt. `reset` discards
/// what was recorded rather than resuming it, which is what replay means here:
/// the walk starts again at the journey's entry screen.
pub(crate) fn load_or_start_state(
    definition: &Value,
    revision: &str,
    reset: bool,
) -> Result<Value> {
    if !reset {
        if let Some(existing) = read_state(definition)? {
            return Ok(existing);
        }
    }
    let subject = subject_hash()?;
    let attempt = attempt_id(&subject);
    let state = json!({
        "schema": STATE_SCHEMA,
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": definition.get("journey_version"),
        "source_revision": definition.get("source_revision"),
        "subject_hash": subject,
        "attempt_id": attempt,
        "started_at": now_iso(),
        "current_screen_id": definition.get("entry_screen_id"),
        "status": "in_progress",
        "revision": revision,
    });
    save_state(&state)?;
    Ok(state)
}

/// What is on disk for this journey, or nothing usable.
pub(crate) fn read_state(definition: &Value) -> Result<Option<Value>> {
    let path = state_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(|error| {
        Error(format!(
            "cannot read onboarding progress {}: {error}",
            path.display()
        ))
    })?;
    let existing: Value = serde_json::from_str(&text).map_err(|error| {
        Error(format!(
            "onboarding progress {} is not JSON ({error}); use --reset to replace it",
            path.display()
        ))
    })?;
    if existing.get("schema").and_then(Value::as_str) != Some(STATE_SCHEMA)
        || existing.get("product_id").and_then(Value::as_str) != Some(PRODUCT_ID)
        || existing.get("journey_id").and_then(Value::as_str) != Some(JOURNEY_ID)
    {
        return Err(Error(
            "stored onboarding state identity mismatch; use --reset to replace it".into(),
        ));
    }
    // A journey republished with new screens invalidates a screen id that no
    // longer exists; resuming into it would show nothing at all.
    let current = string_field(&existing, "current_screen_id")
        .ok_or_else(|| Error("stored onboarding state has no current screen".into()))?;
    if string_field(&existing, "journey_version") != string_field(definition, "journey_version")
        || screen_by_id(definition, &current).is_err()
    {
        return Ok(None);
    }
    Ok(Some(existing))
}

/// The state as it stands on disk, which is where `infer` wrote the
/// first-success fact.
pub(crate) fn recorded_state(definition: &Value) -> Result<Value> {
    read_state(definition)?
        .ok_or_else(|| Error("onboarding progress is no longer readable on disk".into()))
}

pub(crate) fn save_state(state: &Value) -> Result<()> {
    crate::util::write_whole(
        &state_path()?,
        format!("{}\n", serde_json::to_string(state)?).as_bytes(),
    )
}

/// Progress belongs to this operator on this machine, not to a training root:
/// a second `--training-root` is still the same first use. The operator is the
/// account `USER` names; without it the progress has no owner and is refused.
pub(crate) fn subject_hash() -> Result<String> {
    let user = match std::env::var("USER") {
        Ok(user) if !user.trim().is_empty() => user,
        Ok(_) => return Err(Error("USER is set but empty; onboarding progress belongs to an account and needs its name".into())),
        Err(error) => return Err(Error(format!("USER cannot be read ({error}); onboarding progress belongs to an account and needs its name"))),
    };
    Ok(format!(
        "{:x}",
        Sha256::digest(
            format!(
                "transcript-label-trainer-onboarding\0{user}\0{}",
                machine_name()?
            )
            .as_bytes()
        )
    ))
}

/// This attempt, distinguishable from a later replay of the same journey: the
/// whole SHA-256 of the subject, the start instant and the process.
pub(crate) fn attempt_id(subject: &str) -> String {
    let digest =
        Sha256::digest(format!("{subject}\0{}\0{}", now_iso(), std::process::id()).as_bytes());
    format!("{digest:x}")
}

/// Local host name. The lake and Stado both identify this machine by it, and
/// there is no crate here that answers the question, so `hostname` is asked
/// and its refusal is reported in its own words.
pub(crate) fn machine_name() -> Result<String> {
    let answer = stado_wait::output(&mut Command::new("hostname"))
        .map_err(|error| Error(format!("could not run hostname: {error}")))?;
    if !answer.status.success() {
        return Err(Error(format!(
            "hostname failed ({}): {}",
            answer.status,
            String::from_utf8_lossy(&answer.stderr).trim()
        )));
    }
    let name = String::from_utf8_lossy(&answer.stdout).trim().to_string();
    if name.is_empty() {
        return Err(Error("hostname answered with no name".into()));
    }
    Ok(name)
}

/// Where this machine's onboarding progress is kept: under the XDG state
/// directory, which is `$XDG_STATE_HOME`, or `$HOME/.local/state` when that is
/// unset or empty, as the XDG Base Directory specification defines
/// (https://specifications.freedesktop.org/basedir-spec/latest/).
pub(crate) fn state_path() -> Result<PathBuf> {
    let state_home = match std::env::var_os("XDG_STATE_HOME") {
        Some(path) if !path.is_empty() => PathBuf::from(path),
        _ => match std::env::var_os("HOME") {
            Some(home) if !home.is_empty() => PathBuf::from(home).join(".local").join("state"),
            _ => {
                return Err(Error(
                    "neither XDG_STATE_HOME nor HOME names a directory for onboarding progress"
                        .into(),
                ))
            }
        },
    };
    Ok(state_home.join(PRODUCT_ID).join("onboarding.json"))
}

pub(crate) fn wait_for_enter(unattended: bool, prompt: &str) -> Result<()> {
    if unattended {
        return Ok(());
    }
    print!("{prompt}");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    Ok(())
}
