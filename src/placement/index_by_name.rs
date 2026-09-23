use super::*;

/// Targets keyed by their declared name, in registry order — the shape the
/// Python built with a dict comprehension, duplicates and all: a later target
/// carrying a name already seen replaces the value at the first position.
pub(crate) fn index_by_name(targets: &[Value]) -> Vec<(Option<String>, &Value)> {
    let mut by_name: Vec<(Option<String>, &Value)> = Vec::with_capacity(targets.len());
    for target in targets {
        if !target.is_object() {
            continue;
        }
        let name = target
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string);
        match by_name.iter_mut().find(|(seen, _)| *seen == name) {
            Some(slot) => slot.1 = target,
            None => by_name.push((name, target)),
        }
    }
    by_name
}

pub(crate) fn declared_lake_root(target: Option<&Value>) -> Option<PathBuf> {
    let block = target?.get(LAKE_KEY)?;
    if !block.is_object() {
        return None;
    }
    let root = block.get(LAKE_ROOT_KEY)?;
    if !json_truthy(root) {
        return None;
    }
    Some(expanduser(&json_text(root)))
}

/// The one host Stado places label-model training on.
pub(crate) fn declared_training(by_name: &[(Option<String>, &Value)]) -> (Option<String>, Option<PathBuf>) {
    for (name, target) in by_name {
        let Some(block) = target.get(TRAINING_KEY) else {
            continue;
        };
        if !block.is_object() {
            continue;
        }
        if !block.get(TRAINING_ENABLED_KEY).is_some_and(json_truthy) {
            continue;
        }
        if let Some(Value::Array(kinds)) = block.get(TRAINING_KINDS_KEY) {
            if !kinds
                .iter()
                .any(|kind| kind.as_str() == Some(TRAINING_KIND))
            {
                continue;
            }
        }
        let root = block
            .get(TRAINING_ROOT_KEY)
            .filter(|root| json_truthy(root))
            .map(|root| expanduser(&json_text(root)));
        return (name.clone(), root);
    }
    (None, None)
}

/// Which registry target this machine is, per `stado registry self`.
pub(crate) fn self_target() -> (Option<String>, String) {
    let (out, failure) = run_stado(&["registry", "self"]);
    let Some(out) = out else {
        return (None, failure);
    };
    let name = if out.trim().is_empty() {
        String::new()
    } else {
        out.split('\t').next().unwrap_or("").trim().to_string()
    };
    if name.is_empty() {
        return (
            None,
            "'stado registry self' did not name this machine".to_string(),
        );
    }
    (Some(name), String::new())
}

/// Run a stado subcommand. Returns `(Some(stdout), "")` or `(None, why not)`.
pub(crate) fn run_stado(args: &[&str]) -> (Option<String>, String) {
    let printable = format!("{STADO_BIN} {}", args.join(" "));
    let spawned = Command::new(STADO_BIN)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return (None, format!("the '{STADO_BIN}' CLI is not on PATH"));
        }
        Err(error) => return (None, format!("'{printable}' could not run: {error}")),
    };

    // Drain both pipes from their own threads: a registry document is larger
    // than a pipe buffer, and a writer blocked on a full pipe would never
    // reach the deadline below.
    let (Some(mut out_pipe), Some(mut err_pipe)) = (child.stdout.take(), child.stderr.take())
    else {
        let _ = child.kill();
        let _ = child.wait();
        return (None, format!("'{printable}' could not run: no pipes"));
    };
    let out_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = out_pipe.read_to_end(&mut buffer);
        buffer
    });
    let err_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = err_pipe.read_to_end(&mut buffer);
        buffer
    });

    let deadline = Instant::now() + Duration::from_secs(STADO_TIMEOUT_SECONDS);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (
                        None,
                        format!("'{printable}' timed out after {STADO_TIMEOUT_SECONDS}s"),
                    );
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return (None, format!("'{printable}' could not run: {error}")),
        }
    };

    let stdout = String::from_utf8_lossy(&out_reader.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
    if !status.success() {
        return (
            None,
            format!("'{printable}' failed: {}", first_line(&stderr, &stdout)),
        );
    }
    (Some(stdout), String::new())
}

pub(crate) fn run_stado_json(args: &[&str]) -> (Option<Value>, String) {
    let (out, failure) = run_stado(args);
    let Some(out) = out else {
        return (None, failure);
    };
    let printable = format!("{STADO_BIN} {}", args.join(" "));
    match serde_json::from_str::<Value>(&out) {
        Err(error) => (
            None,
            format!("'{printable}' returned unparseable JSON: {error}"),
        ),
        Ok(parsed) if parsed.is_object() => (Some(parsed), String::new()),
        Ok(other) => (
            None,
            format!(
                "'{printable}' returned a {}, not an object",
                type_name(&other)
            ),
        ),
    }
}

pub(crate) fn first_line(stderr: &str, stdout: &str) -> String {
    let stderr = stderr.trim();
    let stdout = stdout.trim();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "no output"
    };
    detail.lines().next().unwrap_or(detail).to_string()
}

pub(crate) fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

pub(crate) fn display_or_none(path: Option<&PathBuf>) -> String {
    match path {
        Some(path) => path.display().to_string(),
        None => "None".to_string(),
    }
}

pub(crate) fn expanduser(value: &str) -> PathBuf {
    if value == "~" {
        return home_dir();
    }
    match value.strip_prefix("~/") {
        Some(rest) => home_dir().join(rest),
        None => PathBuf::from(value),
    }
}
