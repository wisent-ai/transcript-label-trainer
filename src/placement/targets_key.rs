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

// The local layer's directories under `$HOME`, for a machine Stado places nothing on.
pub(crate) const LOCAL_TRAINING_DIR: &str = ".transcript-label-trainer";

pub(crate) const LOCAL_STORAGE_DIR: &str = ".transcript-lake";

/// The layer a root was resolved from, weakest first: the derived order is
/// the strength order, so the weaker of two is their minimum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    LocalFallback,
    Stado,
    Env,
    Flag,
}

impl Source {
    /// The layer as `info` and its JSON name it.
    pub fn label(self) -> &'static str {
        match self {
            Source::LocalFallback => "local-fallback",
            Source::Stado => "stado",
            Source::Env => "env",
            Source::Flag => "flag",
        }
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

/// The resolved answer to "where does this run, and out of what data".
#[derive(Clone, Debug)]
pub struct Placement {
    pub training_host: Option<String>,
    pub training_root: PathBuf,
    pub storage_root: PathBuf,
    pub source: Source,
    pub detail: String,
}

#[derive(Default)]
pub(crate) struct Overrides {
    pub(crate) training_root: Option<PathBuf>,
    pub(crate) storage_root: Option<PathBuf>,
}

pub(crate) static OVERRIDES: LazyLock<Mutex<Overrides>> =
    LazyLock::new(|| Mutex::new(Overrides::default()));

pub(crate) static CACHE: LazyLock<Mutex<Option<Placement>>> = LazyLock::new(|| Mutex::new(None));

/// The value behind one of this module's locks. Both hold plain paths that a
/// panicking holder cannot leave half-written, so a poisoned lock still holds
/// a whole value and is read as it stands.
fn held<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Record explicit CLI roots — the strongest layer — and drop the cache.
pub fn set_override(training_root: Option<&str>, storage_root: Option<&str>) {
    {
        let mut overrides = held(&OVERRIDES);
        if let Some(value) = training_root.filter(|value| !value.is_empty()) {
            overrides.training_root = Some(expanduser(value));
        }
        if let Some(value) = storage_root.filter(|value| !value.is_empty()) {
            overrides.storage_root = Some(expanduser(value));
        }
    }
    *held(&CACHE) = None;
}

/// Resolve placement once per process, per set of overrides.
pub fn resolve_placement() -> Placement {
    let mut cache = held(&CACHE);
    match cache.as_ref() {
        Some(resolved) => resolved.clone(),
        None => {
            let resolved = resolve();
            *cache = Some(resolved.clone());
            resolved
        }
    }
}

/// The resolved placement, shaped for `info --json`.
pub fn as_dict() -> Value {
    let resolved = resolve_placement();
    let mut map = serde_json::Map::new();
    map.insert("source".to_string(), Value::from(resolved.source.label()));
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
        source: training_source.min(storage_source),
        detail: format!("training root {training_note}; storage root {storage_note}"),
    }
}

pub(crate) fn overridden(field: impl FnOnce(&Overrides) -> Option<PathBuf>) -> Option<PathBuf> {
    field(&held(&OVERRIDES))
}

/// A root named by `env_var`, when it names one. The variable is read as the
/// operating system holds it, so a root that is not UTF-8 is still a root;
/// only one that is unset or blank is absent.
fn from_environment(env_var: &str) -> Option<PathBuf> {
    let value = std::env::var_os(env_var)?;
    match value.to_str() {
        Some(text) if text.trim().is_empty() => None,
        Some(text) => Some(expanduser(text.trim())),
        None => Some(PathBuf::from(value)),
    }
}

/// One root through the four layers, plus the reason it landed there.
pub(crate) fn pick(
    flagged: Option<PathBuf>,
    env_var: &str,
    declared: Option<PathBuf>,
    local: PathBuf,
    why_not: &str,
) -> (PathBuf, Source, String) {
    if let Some(flagged) = flagged {
        let note = format!("{} from the command line", flagged.display());
        return (flagged, Source::Flag, note);
    }
    if let Some(path) = from_environment(env_var) {
        let note = format!("{} from ${env_var}", path.display());
        return (path, Source::Env, note);
    }
    if let Some(declared) = declared {
        let note = format!("{} declared in the Stado registry", declared.display());
        return (declared, Source::Stado, note);
    }
    let reason = if why_not.is_empty() {
        "the Stado registry declares no root for it"
    } else {
        why_not
    };
    let note = format!("{} — local fallback because {reason}", local.display());
    (local, Source::LocalFallback, note)
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
