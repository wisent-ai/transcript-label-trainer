use super::*;

/// Parallel Brama calls while preparing pairs: two per row, short answers.
const PREPARE_WORKERS: usize = 16;

/// Parallel Brama judges during the audit: long answers on a strong route.
const AUDIT_WORKERS: usize = 8;

fn workers(args: &Parsed, standard: usize) -> Result<usize> {
    match args.int("--workers") {
        None => Ok(standard),
        Some(value) if value >= 1 => Ok(value as usize),
        Some(value) => Err(Error(format!("--workers must be at least 1, not {value}"))),
    }
}

fn workers_option(standard: usize) -> Opt {
    option(
        "--workers",
        "N",
        Kind::Int,
        format!("parallel Brama calls (default: {standard})"),
    )
}

pub(crate) fn humanizer_specs() -> Vec<Spec> {
    let teacher = brama::DEFAULT_MODEL;
    let best = brama::BEST_MODEL;
    let prepare = Spec {
        name: "humanizer-prepare",
        help: "build the humanizer's inverse style-transfer splits through Brama".to_string(),
        description: Some(
            "Have a Brama teacher rewrite every masked authored target as generic AI prose, \
             refuse rewrites that drop a URL, e-mail address or number or change the length \
             too much, keep only pairs an independent Brama review calls usable, and write \
             session-separated train.jsonl, validation.jsonl, test.jsonl and preparation.json. \
             Nothing is written unless each split reaches its minimum."
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
            workers_option(PREPARE_WORKERS),
        ],
    };
    let audit = Spec {
        name: "humanizer-audit",
        help: "judge base and trained humanizer outputs through Brama".to_string(),
        description: Some(
            "Judge every held-out case for semantic fidelity, voice match and AI boilerplate \
             for both the base model and the trained adapter, write the complete record, and \
             exit 1 when the adapter misses the quality gate. Any case the judge cannot answer \
             fails the audit."
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
            workers_option(AUDIT_WORKERS),
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

pub(crate) fn cmd_humanizer_prepare(args: &Parsed) -> Result<i32> {
    let teacher = args.text("--teacher-model").unwrap_or(brama::DEFAULT_MODEL);
    let reviewer = args.text("--review-model").unwrap_or(brama::BEST_MODEL);
    let report = crate::humanizer::prepare_dataset(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output-dir").unwrap_or_default()),
        teacher,
        reviewer,
        workers(args, PREPARE_WORKERS)?,
    )?;
    outln!("{}", dumps(&report));
    Ok(0)
}

pub(crate) fn cmd_humanizer_audit(args: &Parsed) -> Result<i32> {
    let judge = args.text("--brama-model").unwrap_or(brama::BEST_MODEL);
    let summary = crate::humanizer::audit_outputs(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output").unwrap_or_default()),
        judge,
        workers(args, AUDIT_WORKERS)?,
    )?;
    outln!("{}", dumps(&summary));
    Ok(i32::from(
        summary.get("passed").and_then(Value::as_bool) != Some(true),
    ))
}

pub(crate) fn cmd_humanizer_publish(args: &Parsed) -> Result<i32> {
    use std::io::Write;
    use std::path::Path;

    let path = Path::new(args.text("--output").unwrap_or_default());
    let mut output = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|error| Error(format!("cannot create publication report {}: {error}", path.display())))?;
    let report = match crate::humanizer::publish_adapter(crate::humanizer::Publication {
        model: Path::new(args.positional(0)),
        metrics: Path::new(args.text("--metrics").unwrap_or_default()),
        audit: Path::new(args.text("--audit").unwrap_or_default()),
        preparation: Path::new(args.text("--preparation").unwrap_or_default()),
        repository: args.text("--repo").unwrap_or_default(),
    }) {
        Ok(report) => report,
        Err(error) => serde_json::json!({"qualified": false, "error": error.to_string()}),
    };
    let failed = report.get("qualified").and_then(Value::as_bool) != Some(true);
    let mut encoded = dumps(&report);
    encoded.push('\n');
    output.write_all(encoded.as_bytes())?;
    outln!("{}", encoded.trim_end_matches('\n'));
    Ok(i32::from(failed))
}
