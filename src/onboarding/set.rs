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
pub(crate) fn load_or_start_state(definition: &Value, revision: &str, reset: bool) -> Result<Value> {
    if !reset {
        if let Some(existing) = read_state(definition)? {
            return Ok(existing);
        }
    }
    let state = json!({
        "schema": STATE_SCHEMA,
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": definition.get("journey_version"),
        "source_revision": definition.get("source_revision"),
        "subject_hash": subject_hash(),
        "attempt_id": attempt_id(),
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
    let path = state_path();
    if !path.exists() {
        return Ok(None);
    }
    let existing: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
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
    let path = state_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = PathBuf::from(format!("{}.tmp-{}", path.display(), std::process::id()));
    fs::write(&temporary, format!("{}\n", serde_json::to_string(state)?))?;
    fs::rename(&temporary, &path)?;
    Ok(())
}

/// Progress belongs to this operator on this machine, not to a training root:
/// a second `--training-root` is still the same first use.
pub(crate) fn subject_hash() -> String {
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown-user".to_string());
    format!(
        "{:x}",
        Sha256::digest(
            format!(
                "transcript-label-trainer-onboarding\0{user}\0{}",
                machine_name()
            )
            .as_bytes()
        )
    )
}

/// This attempt, distinguishable from a later replay of the same journey.
pub(crate) fn attempt_id() -> String {
    let digest = Sha256::digest(
        format!(
            "{}\0{}\0{}",
            subject_hash(),
            now_iso(),
            std::process::id()
        )
        .as_bytes(),
    );
    format!("{digest:x}")[..32].to_string()
}

/// Local host name. The lake and Stado both identify this machine by it, and
/// there is no crate here that answers the question.
pub(crate) fn machine_name() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

pub(crate) fn state_path() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| home_dir().join(".local").join("state"))
        .join(PRODUCT_ID)
        .join("onboarding.json")
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
