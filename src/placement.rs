//! Where training runs and where lake data lives — resolved from Stado.
//!
//! Stado owns the canonical compute-target registry, and that registry is the
//! authority on placement:
//!
//! - `targets[<this machine>].transcript_lake.root` — the lake data root, i.e.
//!   the storage root this trainer reads labels and session text out of;
//! - `targets[<host>].training` — the host that trains label models, with
//!   `models_dir` as the artifact root on that host.
//!
//! Resolution order, strongest first:
//!
//! 1. an explicit CLI flag (`--training-root` / `--storage-root`),
//! 2. the environment (`TLT_HOME` / `LAKE_DATA`),
//! 3. the Stado registry declarations above,
//! 4. a local fallback under `~`.
//!
//! The fallback is an exception, not a default. Anything that stops Stado from
//! answering — the binary absent, the registry unreachable, no declaration for
//! this machine, training placed on a host that is not this one — degrades to
//! the local path and writes the reason into [`Placement::detail`], which
//! `info` prints. Resolution never fails: a broken control plane must not stop
//! a local run, it must only stop being invisible.
//!
//! [`Placement::source`] reports the *weakest* layer any root depended on, so
//! one silent local fallback cannot hide behind a root that did resolve.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::util::{home_dir, json_text, json_truthy};

// Registry keys read here, grounded in `stado registry pull` output. Neither
// block is modelled by stado's own Rust structs: both ride in the per-target
// `extra` map, which the registry loader round-trips verbatim.
const TARGETS_KEY: &str = "targets";
const LAKE_KEY: &str = "transcript_lake";
const LAKE_ROOT_KEY: &str = "root";
const TRAINING_KEY: &str = "training";
const TRAINING_ENABLED_KEY: &str = "enabled";
const TRAINING_KINDS_KEY: &str = "kinds";
const TRAINING_ROOT_KEY: &str = "models_dir";

// The training kind this trainer claims; a host may be declared for others.
const TRAINING_KIND: &str = "label-model";

// Historical defaults, kept only as the local exception path.
const LOCAL_TRAINING_DIR: &str = ".transcript-label-trainer";
const LOCAL_STORAGE_DIR: &str = ".transcript-lake";

const STADO_BIN: &str = "stado";
const STADO_TIMEOUT_SECONDS: u64 = 20;

// Weakest to strongest. `Placement.source` is the weakest one in play.
const SOURCE_ORDER: [&str; 4] = ["local-fallback", "stado", "env", "flag"];

/// The resolved answer to "where does this run, and out of what data".
#[derive(Clone, Debug)]
pub struct Placement {
    pub training_host: Option<String>,
    pub training_root: PathBuf,
    pub storage_root: PathBuf,
    pub source: &'static str,
    pub detail: String,
}

#[derive(Default)]
struct Overrides {
    training_root: Option<PathBuf>,
    storage_root: Option<PathBuf>,
}

static OVERRIDES: LazyLock<Mutex<Overrides>> = LazyLock::new(|| Mutex::new(Overrides::default()));
static CACHE: LazyLock<Mutex<Option<Placement>>> = LazyLock::new(|| Mutex::new(None));

/// Record explicit CLI roots — the strongest layer — and drop the cache.
pub fn set_override(training_root: Option<&str>, storage_root: Option<&str>) {
    {
        let mut overrides = OVERRIDES
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(value) = training_root.filter(|value| !value.is_empty()) {
            overrides.training_root = Some(expanduser(value));
        }
        if let Some(value) = storage_root.filter(|value| !value.is_empty()) {
            overrides.storage_root = Some(expanduser(value));
        }
    }
    *CACHE.lock().unwrap_or_else(|poison| poison.into_inner()) = None;
}

/// Resolve placement once per process, per set of overrides.
pub fn resolve_placement() -> Placement {
    let mut cache = CACHE.lock().unwrap_or_else(|poison| poison.into_inner());
    if cache.is_none() {
        *cache = Some(resolve());
    }
    cache.clone().unwrap_or_else(local_placement)
}

/// The resolved placement, shaped for `info --json`.
pub fn as_dict() -> Value {
    let resolved = resolve_placement();
    let mut map = serde_json::Map::new();
    map.insert("source".to_string(), Value::from(resolved.source));
    map.insert(
        "training_host".to_string(),
        match resolved.training_host {
            Some(host) => Value::from(host),
            None => Value::Null,
        },
    );
    map.insert(
        "training_root".to_string(),
        Value::from(resolved.training_root.to_string_lossy().into_owned()),
    );
    map.insert(
        "storage_root".to_string(),
        Value::from(resolved.storage_root.to_string_lossy().into_owned()),
    );
    map.insert("detail".to_string(), Value::from(resolved.detail));
    Value::Object(map)
}

#[derive(Default)]
struct Declared {
    training_host: Option<String>,
    training_root: Option<PathBuf>,
    storage_root: Option<PathBuf>,
}

fn resolve() -> Placement {
    let (declared, why_not) = stado_declaration();

    let (training_root, training_source, training_note) = pick(
        overridden(|overrides| overrides.training_root.clone()),
        "TLT_HOME",
        declared.training_root,
        home_dir().join(LOCAL_TRAINING_DIR),
        &why_not,
    );
    let (storage_root, storage_source, storage_note) = pick(
        overridden(|overrides| overrides.storage_root.clone()),
        "LAKE_DATA",
        declared.storage_root,
        home_dir().join(LOCAL_STORAGE_DIR),
        &why_not,
    );
    Placement {
        training_host: declared.training_host,
        training_root,
        storage_root,
        source: weakest(training_source, storage_source),
        detail: format!("training root {training_note}; storage root {storage_note}"),
    }
}

/// The placement a process falls back to when the cache itself is unusable.
fn local_placement() -> Placement {
    let training_root = home_dir().join(LOCAL_TRAINING_DIR);
    let storage_root = home_dir().join(LOCAL_STORAGE_DIR);
    let detail = format!(
        "training root {} — local fallback because placement could not be resolved; \
         storage root {} — local fallback because placement could not be resolved",
        training_root.display(),
        storage_root.display()
    );
    Placement {
        training_host: None,
        training_root,
        storage_root,
        source: "local-fallback",
        detail,
    }
}

fn overridden(field: impl FnOnce(&Overrides) -> Option<PathBuf>) -> Option<PathBuf> {
    let overrides = OVERRIDES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    field(&overrides)
}

/// One root through the four layers, plus the reason it landed there.
fn pick(
    flagged: Option<PathBuf>,
    env_var: &str,
    declared: Option<PathBuf>,
    local: PathBuf,
    why_not: &str,
) -> (PathBuf, &'static str, String) {
    if let Some(flagged) = flagged {
        let note = format!("{} from the command line", flagged.display());
        return (flagged, "flag", note);
    }
    let from_env = std::env::var(env_var).unwrap_or_default();
    let from_env = from_env.trim();
    if !from_env.is_empty() {
        let path = expanduser(from_env);
        let note = format!("{} from ${env_var}", path.display());
        return (path, "env", note);
    }
    if let Some(declared) = declared {
        let note = format!("{} declared in the Stado registry", declared.display());
        return (declared, "stado", note);
    }
    let reason = if why_not.is_empty() {
        "the Stado registry declares no root for it"
    } else {
        why_not
    };
    let note = format!("{} — local fallback because {reason}", local.display());
    (local, "local-fallback", note)
}

fn weakest(left: &'static str, right: &'static str) -> &'static str {
    let rank = |source: &str| {
        SOURCE_ORDER
            .iter()
            .position(|entry| *entry == source)
            .unwrap_or(0)
    };
    if rank(left) <= rank(right) {
        left
    } else {
        right
    }
}

/// The registry's placement declarations, or why there are none.
///
/// Returns `(declared, why_not)`. `declared` carries whichever of
/// `training_host` / `training_root` / `storage_root` Stado answered for;
/// `why_not` states why the rest are absent. Never fails.
fn stado_declaration() -> (Declared, String) {
    let (registry, failure) = run_stado_json(&["registry", "pull"]);
    let Some(registry) = registry else {
        return (Declared::default(), failure);
    };
    let (self_name, failure) = self_target();
    let Some(self_name) = self_name else {
        return (Declared::default(), failure);
    };

    let Some(Value::Array(targets)) = registry.get(TARGETS_KEY) else {
        return (
            Declared::default(),
            format!("the Stado registry carries no '{TARGETS_KEY}' list"),
        );
    };
    let by_name = index_by_name(targets);

    let mut declared = Declared::default();
    let mut reasons: Vec<String> = Vec::new();

    let own = by_name
        .iter()
        .find(|(name, _)| name.as_deref() == Some(self_name.as_str()))
        .map(|(_, target)| *target);
    match declared_lake_root(own) {
        None => reasons.push(format!(
            "Stado target '{self_name}' declares no {LAKE_KEY}.{LAKE_ROOT_KEY}"
        )),
        Some(root) => declared.storage_root = Some(root),
    }

    let (training_host, training_root) = declared_training(&by_name);
    match training_host {
        None => reasons.push(format!(
            "no Stado target declares {TRAINING_KEY}.{TRAINING_KINDS_KEY} \
             containing '{TRAINING_KIND}'"
        )),
        Some(training_host) => {
            if training_host != self_name {
                reasons.push(format!(
                    "Stado places {TRAINING_KIND} training on {training_host} at {}, \
                     and this machine is {self_name}",
                    display_or_none(training_root.as_ref())
                ));
            } else if training_root.is_none() {
                reasons.push(format!(
                    "Stado target '{training_host}' declares {TRAINING_KEY} without \
                     {TRAINING_ROOT_KEY}"
                ));
            } else {
                declared.training_root = training_root;
            }
            declared.training_host = Some(training_host);
        }
    }

    (declared, reasons.join("; "))
}

/// Targets keyed by their declared name, in registry order — the shape the
/// Python built with a dict comprehension, duplicates and all: a later target
/// carrying a name already seen replaces the value at the first position.
fn index_by_name(targets: &[Value]) -> Vec<(Option<String>, &Value)> {
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

fn declared_lake_root(target: Option<&Value>) -> Option<PathBuf> {
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
fn declared_training(by_name: &[(Option<String>, &Value)]) -> (Option<String>, Option<PathBuf>) {
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
fn self_target() -> (Option<String>, String) {
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
fn run_stado(args: &[&str]) -> (Option<String>, String) {
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

fn run_stado_json(args: &[&str]) -> (Option<Value>, String) {
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

fn first_line(stderr: &str, stdout: &str) -> String {
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

fn type_name(value: &Value) -> &'static str {
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

fn display_or_none(path: Option<&PathBuf>) -> String {
    match path {
        Some(path) => path.display().to_string(),
        None => "None".to_string(),
    }
}

fn expanduser(value: &str) -> PathBuf {
    if value == "~" {
        return home_dir();
    }
    match value.strip_prefix("~/") {
        Some(rest) => home_dir().join(rest),
        None => PathBuf::from(value),
    }
}
