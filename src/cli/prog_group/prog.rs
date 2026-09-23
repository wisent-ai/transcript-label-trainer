use super::*;

pub(crate) const PROG: &str = "transcript-label-trainer";

/// argparse's `HelpFormatter._width`: the terminal width less two, and the
/// terminal falls back to 80 columns whenever stdout is not a terminal.
pub(crate) const WIDTH: usize = 78;

/// argparse's `max_help_position`, the column help text starts in.
pub(crate) const MAX_HELP_POSITION: usize = 24;

pub(crate) const TOP_DESCRIPTION: &str = "Train small local classifiers that predict Transcript Lake aspect \
     labels, and emit label suggestions. Never writes to the lake.";

pub(crate) const TRAINING_ROOT_HELP: &str = "override where model artifacts live; beats $TLT_HOME and the \
     Stado registry declaration";

pub(crate) const STORAGE_ROOT_HELP: &str = "override the lake data root; beats $LAKE_DATA and the Stado \
     registry declaration";

pub(crate) const HELP_HELP: &str = "show this help message and exit";

/// The crate version, printed bare by `--version`, exactly as the lake's CLI
/// prints its own (`transcript-lake --version` -> `0.2.0`). A catalogued
/// product built from a checkout has to be able to state its version, or
/// `wisent-products status` can only answer by hashing bytes.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) const VERSION_HELP: &str = "show program's version number and exit";

/// The entry point `main` calls. The returned integer is the process status:
/// 0 on success, 1 on a failed command, 2 on a usage error or on a run the
/// lake does not hold enough labeled sessions for.
pub fn run(args: Vec<String>) -> Result<i32> {
    let specs = build_specs();

    let mut training_root: Option<String> = None;
    let mut storage_root: Option<String> = None;
    let mut command: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg == "-h" || arg == "--help" {
            print!("{}", top_help(&specs));
            return Ok(0);
        }
        if arg == "-V" || arg == "--version" {
            println!("{VERSION}");
            return Ok(0);
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg, None),
        };
        if name == "--training-root" || name == "--storage-root" {
            let value = match inline {
                Some(value) => {
                    index += 1;
                    value
                }
                None => match args.get(index + 1) {
                    Some(next) if !looks_like_option(next) => {
                        index += 2;
                        next.clone()
                    }
                    _ => {
                        return Ok(top_error(&format!(
                            "argument {name}: expected one argument"
                        )));
                    }
                },
            };
            if name == "--training-root" {
                training_root = Some(value);
            } else {
                storage_root = Some(value);
            }
            continue;
        }
        if looks_like_option(arg) {
            return Ok(top_error(&format!("unrecognized arguments: {arg}")));
        }
        command = Some(arg.to_string());
        index += 1;
        break;
    }

    let Some(command) = command else {
        return Ok(top_error("the following arguments are required: command"));
    };
    let Some(spec) = specs.iter().find(|spec| spec.name == command) else {
        let choices = specs
            .iter()
            .map(|spec| format!("'{}'", spec.name))
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(top_error(&format!(
            "argument command: invalid choice: '{command}' (choose from {choices})"
        )));
    };

    let parsed = match parse_sub(spec, &args[index..]) {
        Outcome::Help => {
            print!("{}", sub_help(spec));
            return Ok(0);
        }
        Outcome::Error(message) => return Ok(sub_error(spec, &message)),
        Outcome::Parsed(parsed) => parsed,
    };

    placement::set_override(training_root.as_deref(), storage_root.as_deref());

    match spec.name {
        "train" => cmd_train(&parsed),
        "run" => cmd_run(&parsed),
        "evaluate" => cmd_evaluate(&parsed),
        "infer" => cmd_infer(&parsed),
        "autolabel" => cmd_autolabel(&parsed),
        "corpus-adopt" => cmd_corpus_adopt(&parsed),
        "corpus-status" => cmd_corpus_status(&parsed),
        "gui" => cmd_gui(&parsed),
        "aspect-discover" => cmd_aspect_discover(&parsed),
        "info" => cmd_info(&parsed),
        "onboarding" => cmd_onboarding(&parsed),
        "goal-model" => cmd_goal_model(&parsed),
        "lifecycle-review" => cmd_lifecycle_review(&parsed),
        "lifecycle-model" => cmd_lifecycle_model(&parsed),
        "humanizer-model" => cmd_humanizer_model(&parsed),
        "lifecycle-audit" => cmd_lifecycle_audit(&parsed),
        "goal-audit" => cmd_goal_audit(&parsed),
        other => Err(Error(format!("unknown command '{other}'"))),
    }
}

// ---------------------------------------------------------------- commands

pub(crate) fn cmd_train(args: &Parsed) -> Result<i32> {
    let eval_split = if args.flag("--no-eval-split") {
        serde_json::json!({"enabled": false, "fraction": Value::Null, "seed": Value::Null})
    } else {
        serde_json::json!({
            "enabled": true,
            "fraction": args.float("--eval-split-fraction").unwrap_or(jobs::DEFAULT_EVAL_FRACTION),
            "seed": args.int("--eval-split-seed").unwrap_or(jobs::DEFAULT_EVAL_SEED),
        })
    };
    let metrics = match model::train(
        args.text("--aspect").unwrap_or_default(),
        args.text("--model"),
        args.float("--epochs").unwrap_or(3.0),
        count(args.int("--batch-size"), 8),
        args.float("--lr").unwrap_or(2e-5),
        count(args.int("--max-length"), 512),
        &eval_split,
    ) {
        Ok(metrics) => metrics,
        Err(failure) => return Ok(report("train", failure)),
    };
    outln!("{}", dumps(&metrics));
    Ok(0)
}

pub(crate) fn cmd_run(args: &Parsed) -> Result<i32> {
    let job = match jobs::load(args.positional(0)) {
        Ok(job) => job,
        Err(error) => {
            eprintln!("run: {error}");
            return Ok(1);
        }
    };
    if let Some(target) = args.text("--compute-target") {
        return stado::execute(args.positional(0), &job, target);
    }
    let resolved = model::resolve_job(&job)?;
    outln!("{}", dumps(&model::job_summary(&job, &resolved)));

    let plan = match model::prepare_job(&job, &resolved) {
        Ok(plan) => plan,
        Err(failure) => return Ok(report("run", failure)),
    };
    outln!(
        "{}",
        dumps(&serde_json::json!({"eval_split": model::split_summary(&plan)}))
    );

    let metrics = match model::run_job(&job, &plan) {
        Ok(metrics) => metrics,
        Err(failure) => return Ok(report("run", failure)),
    };
    outln!("{}", dumps(&metrics));
    Ok(0)
}

pub(crate) fn cmd_evaluate(args: &Parsed) -> Result<i32> {
    let judge = if args.flag("--no-judge") {
        Some(false)
    } else {
        None
    };
    let verdict = match evaluate::evaluate(
        args.positional(0),
        judge,
        args.text("--brama-model"),
        args.flag("--best"),
    ) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("evaluate: {error}");
            return Ok(1);
        }
    };
    let status = if args.flag("--best")
        && verdict
            .pointer("/best_review/sensible")
            .and_then(Value::as_bool)
            != Some(true)
    {
        1
    } else {
        0
    };
    if args.flag("--json") {
        outln!("{}", dumps(&verdict));
    } else {
        print_verdict(&verdict);
    }
    Ok(status)
}

pub(crate) fn cmd_infer(args: &Parsed) -> Result<i32> {
    let suggestions = match model::infer(
        args.text("--aspect").unwrap_or_default(),
        args.text("--session"),
        args.int("--limit"),
    ) {
        Ok(suggestions) => suggestions,
        Err(error) => {
            eprintln!("infer: {error}");
            return Ok(1);
        }
    };
    outln!("{}", dumps(&suggestions));
    Ok(0)
}
