use super::*;

pub(crate) fn humanizer_specs() -> Vec<Spec> {
    let teacher = brama::DEFAULT_MODEL;
    let best = brama::BEST_MODEL;
    let prepare = Spec {
        name: "humanizer-prepare",
        help: "build the humanizer's inverse style-transfer splits through Brama".to_string(),
        description: Some(
            "Have a Brama teacher rewrite every masked authored target as generic AI prose, \
             refuse rewrites that drop a URL, e-mail address or number, keep only pairs an \
             independent Brama review calls usable, and write session-separated train.jsonl, \
             test.jsonl and preparation.json; a session lands in test below scikit-learn's \
             documented test share. Nothing is written unless both splits hold a row."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "targets",
            help: "JSONL of {id, session_id, runtime, target} written by humanizer-model"
                .to_string(),
        }],
        opts: vec![
            required(
                "--output-dir",
                "DIR",
                Kind::Text,
                "directory that receives the splits and preparation.json".to_string(),
            ),
            option(
                "--teacher-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama route that writes the generic source (default: {teacher})"),
            ),
            option(
                "--review-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama route that reviews every pair (default: {best})"),
            ),
        ],
    };
    let audit = Spec {
        name: "humanizer-audit",
        help: "judge base and trained humanizer outputs through Brama".to_string(),
        description: Some(
            "Judge every held-out case for semantic fidelity, voice match and AI boilerplate \
             for both the base model and the trained adapter, write the complete record, and \
             exit 1 unless the adapter beats the base: higher voice match, semantic fidelity \
             and pass rate no lower, AI boilerplate rate no higher. Each case is asked once; \
             any case the judge cannot answer fails the audit."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "predictions",
            help: "JSONL of {id, source, target, base, student}".to_string(),
        }],
        opts: vec![
            required(
                "--output",
                "PATH",
                Kind::Text,
                "write the complete audit record here".to_string(),
            ),
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama route of the independent judge (default: {best})"),
            ),
        ],
    };
    let publish = Spec {
        name: "humanizer-publish",
        help: "publish a qualified adapter to an explicitly named private Hugging Face repository".to_string(),
        description: Some(
            "Require the humanizer metrics contract and a passed audit, refuse public destinations, \
             upload through the supported hf CLI on an isolated publication branch, and read back \
             its private immutable revision and file inventory. No account default or visibility change."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "model",
            help: "qualified adapter directory".to_string(),
        }],
        opts: vec![
            required("--repo", "OWNER/REPOSITORY", Kind::Text, "explicit private destination".to_string()),
            required("--metrics", "PATH", Kind::Text, "metrics from the qualified training run".to_string()),
            required("--audit", "PATH", Kind::Text, "passed independent audit".to_string()),
            required("--preparation", "PATH", Kind::Text, "preparation evidence from the same run".to_string()),
            required("--output", "PATH", Kind::Text, "JSON publication or refusal receipt".to_string()),
        ],
    };
    vec![prepare, audit, publish]
}

/// The Brama alias a command option names, or the one it documents as its
/// default when the option is left out.
fn named_or<'a>(args: &'a Parsed, flag: &str, documented: &'a str) -> &'a str {
    match args.text(flag) {
        Some(model) => model,
        None => documented,
    }
}

/// The status of a command that succeeds only when `report[field]` is true.
fn status_of(report: &Value, field: &str) -> i32 {
    match report[field] == Value::Bool(true) {
        true => EXIT_SUCCESS,
        false => EXIT_FAILURE,
    }
}

pub(crate) fn cmd_humanizer_prepare(args: &Parsed) -> Result<i32> {
    let report = crate::humanizer::prepare_dataset(
        std::path::Path::new(args.positional("targets")),
        std::path::Path::new(args.required("--output-dir")),
        named_or(args, "--teacher-model", brama::DEFAULT_MODEL),
        named_or(args, "--review-model", brama::BEST_MODEL),
    )?;
    outln!("{}", dumps(&report));
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_humanizer_audit(args: &Parsed) -> Result<i32> {
    let summary = crate::humanizer::audit_outputs(
        std::path::Path::new(args.positional("predictions")),
        std::path::Path::new(args.required("--output")),
        named_or(args, "--brama-model", brama::BEST_MODEL),
    )?;
    outln!("{}", dumps(&summary));
    Ok(status_of(&summary, "passed"))
}

pub(crate) fn cmd_humanizer_publish(args: &Parsed) -> Result<i32> {
    use std::io::Write;
    use std::path::Path;

    let path = Path::new(args.required("--output"));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            Error(format!(
                "cannot create publication report {}: {error}",
                path.display()
            ))
        })?;
    // A refused publication is written as the refusal receipt the option
    // documents, with the refusal's own words.
    let report = match crate::humanizer::publish_adapter(crate::humanizer::Publication {
        model: Path::new(args.positional("model")),
        metrics: Path::new(args.required("--metrics")),
        audit: Path::new(args.required("--audit")),
        preparation: Path::new(args.required("--preparation")),
        repository: args.required("--repo"),
    }) {
        Ok(report) => report,
        Err(error) => serde_json::json!({"qualified": false, "error": error.to_string()}),
    };
    let mut encoded = dumps(&report);
    encoded.push('\n');
    output.write_all(encoded.as_bytes())?;
    outln!("{}", encoded.trim_end_matches('\n'));
    Ok(status_of(&report, "qualified"))
}
