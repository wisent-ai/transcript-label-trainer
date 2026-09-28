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
    vec![prepare, audit]
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
