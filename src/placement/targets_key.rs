use super::*;

// Registry keys read here, grounded in `stado registry pull` output. Neither
// block is modelled by stado's own Rust structs: both ride in the per-target
// `extra` map, which the registry loader round-trips verbatim.
pub(crate) const TARGETS_KEY: &str = "targets";

pub(crate) const LAKE_KEY: &str = "transcript_lake";

pub(crate) const LAKE_ROOT_KEY: &str = "root";

pub(crate) const TRAINING_KEY: &str = "training";

pub(crate) const TRAINING_ENABLED_KEY: &str = "enabled";

pub(crate) const TRAINING_KINDS_KEY: &str = "kinds";

pub(crate) const TRAINING_ROOT_KEY: &str = "models_dir";

// The training kind this trainer claims; a host may be declared for others.
pub(crate) const TRAINING_KIND: &str = "label-model";

// Historical defaults, kept only as the local exception path.
pub(crate) const LOCAL_TRAINING_DIR: &str = ".transcript-label-trainer";

pub(crate) const LOCAL_STORAGE_DIR: &str = ".transcript-lake";

pub(crate) const STADO_BIN: &str = "stado";

pub(crate) const STADO_TIMEOUT_SECONDS: u64 = 20;

// Weakest to strongest. `Placement.source` is the weakest one in play.
pub(crate) const SOURCE_ORDER: [&str; 4] = ["local-fallback", "stado", "env", "flag"];

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
pub(crate) struct Overrides {
    pub(crate) training_root: Option<PathBuf>,
    pub(crate) storage_root: Option<PathBuf>,
}

pub(crate) static OVERRIDES: LazyLock<Mutex<Overrides>> = LazyLock::new(|| Mutex::new(Overrides::default()));

pub(crate) static CACHE: LazyLock<Mutex<Option<Placement>>> = LazyLock::new(|| Mutex::new(None));

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
pub(crate) struct Declared {
    pub(crate) training_host: Option<String>,
    pub(crate) training_root: Option<PathBuf>,
    pub(crate) storage_root: Option<PathBuf>,
}

pub(crate) fn resolve() -> Placement {
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
pub(crate) fn local_placement() -> Placement {
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

pub(crate) fn overridden(field: impl FnOnce(&Overrides) -> Option<PathBuf>) -> Option<PathBuf> {
    let overrides = OVERRIDES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    field(&overrides)
}

/// One root through the four layers, plus the reason it landed there.
pub(crate) fn pick(
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

pub(crate) fn weakest(left: &'static str, right: &'static str) -> &'static str {
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
pub(crate) fn stado_declaration() -> (Declared, String) {
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
