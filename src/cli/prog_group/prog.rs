use super::*;

pub(crate) const PROG: &str = "transcript-label-trainer";

/// argparse's `HelpFormatter._width` when stdout is not a terminal: the
/// columns Python then reports, less the two argparse keeps free.
pub(crate) const WIDTH: usize = 78; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py HelpFormatter.__init__: width = shutil.get_terminal_size().columns; width -= 2 — and off a terminal https://docs.python.org/3/library/shutil.html#shutil.get_terminal_size fallback=(80, 24)

/// argparse's `max_help_position`, the column help text starts in.
pub(crate) const MAX_HELP_POSITION: usize = 24; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py HelpFormatter.__init__(max_help_position=24)

pub(crate) const TOP_DESCRIPTION: &str =
    "Train small local classifiers that predict Transcript Lake aspect \
     labels, and emit label suggestions. Never writes to the lake.";

pub(crate) const TRAINING_ROOT_HELP: &str =
    "override where model artifacts live; beats $TLT_HOME and the \
     Stado registry declaration";

pub(crate) const STORAGE_ROOT_HELP: &str =
    "override the lake data root; beats $LAKE_DATA and the Stado \
     registry declaration";

pub(crate) const HELP_HELP: &str = "show this help message and exit";

/// The crate version, printed bare by `--version`, exactly as the lake's CLI
/// prints its own (`transcript-lake --version` -> `0.2.0`). A catalogued
/// product built from a checkout has to be able to state its version, or
/// `stado product status` can only answer by hashing bytes.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) const VERSION_HELP: &str = "show program's version number and exit";

/// The entry point `main` calls. The returned integer is the process status:
/// `EXIT_SUCCESS`, `EXIT_FAILURE` for a failed command, and the usage-error
/// status argparse uses, which a run the lake does not hold enough labeled
/// sessions for shares.
pub fn run(args: Vec<String>) -> Result<i32> {
    let specs = build_specs();

    let mut training_root: Option<String> = None;
    let mut storage_root: Option<String> = None;
    let mut rest = args.as_slice();
    let command = loop {
        let Some((arg, after)) = rest.split_first() else {
            return Ok(top_error("the following arguments are required: command"));
        };
        rest = after;
        if arg == "-h" || arg == "--help" {
            print!("{}", top_help(&specs));
            return Ok(EXIT_SUCCESS);
        }
        if arg == "-V" || arg == "--version" {
            println!("{VERSION}");
            return Ok(EXIT_SUCCESS);
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg.as_str(), None),
        };
        if name == "--training-root" || name == "--storage-root" {
            let value = match (inline, rest.split_first()) {
                (Some(value), _) => value,
                (None, Some((next, after))) if !looks_like_option(next) => {
                    rest = after;
                    next.clone()
                }
                (None, _) => {
                    return Ok(top_error(&format!(
                        "argument {name}: expected one argument"
                    )));
                }
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
        break arg.clone();
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

    let parsed = match parse_sub(spec, rest) {
        Outcome::Help => {
            print!("{}", sub_help(spec));
            return Ok(EXIT_SUCCESS);
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
        "corpus-select" => cmd_corpus_change(&parsed, false),
        "corpus-remove" => cmd_corpus_change(&parsed, true),
        "gui" => cmd_gui(&parsed),
        "aspect-discover" => cmd_aspect_discover(&parsed),
        "info" => cmd_info(&parsed),
        "onboarding" => cmd_onboarding(&parsed),
        "goal-model" => cmd_goal_model(&parsed),
        "lifecycle-review" => cmd_lifecycle_review(&parsed),
        "lifecycle-model" => cmd_lifecycle_model(&parsed),
        "humanizer-model" => cmd_humanizer_model(&parsed),
        "humanizer-prepare" => cmd_humanizer_prepare(&parsed),
        "humanizer-audit" => cmd_humanizer_audit(&parsed),
        "humanizer-publish" => cmd_humanizer_publish(&parsed),
        "humanizer-examples" => cmd_humanizer_examples(&parsed),
        "humanizer-evaluate-gguf" => cmd_humanizer_evaluate_gguf(&parsed),
        "lifecycle-audit" => cmd_lifecycle_audit(&parsed),
        "lifecycle-evaluate-gguf" => cmd_lifecycle_evaluate_gguf(&parsed),
        "lifecycle-assemble-curriculum" => cmd_lifecycle_assemble_curriculum(&parsed),
        "lifecycle-generate-curriculum" => cmd_lifecycle_generate_curriculum(&parsed),
        "lifecycle-split-by-day" => cmd_lifecycle_split_by_day(&parsed),
        "lifecycle-decisions" => cmd_lifecycle_decisions(&parsed),
        "lifecycle-examples" => cmd_lifecycle_examples(&parsed),
        "goal-audit" => cmd_goal_audit(&parsed),
        "goal-examples" => cmd_goal_examples(&parsed),
        "goal-evaluate-gguf" => cmd_goal_evaluate_gguf(&parsed),
        "release-publish" => cmd_release_publish(&parsed),
        "model-manifest" => cmd_model_manifest(&parsed),
        other => Err(Error(format!("unknown command '{other}'"))),
    }
}

// ---------------------------------------------------------------- commands

/// The `eval_split` section the train flags state, written the way a job
/// spec writes it, so the job validator decides it exactly as it decides a
/// spec's.
fn stated_eval_split(args: &Parsed) -> Result<serde_yaml::Value> {
    use serde_yaml::Value as Yaml;
    let fraction = args.float("--eval-split-fraction");
    let seed = args.int("--eval-split-seed");
    match (args.flag("--no-eval-split"), args.text("--eval-split"), fraction, seed) {
        (true, _, _, _) => Ok(Yaml::Bool(false)),
        (false, Some(preset), None, None) => Ok(Yaml::String(preset.to_string())),
        (false, Some(preset), _, _) => Err(Error(format!(
            "--eval-split {preset} states the fraction and seed; --eval-split-fraction and --eval-split-seed cannot be given beside it"
        ))),
        (false, None, Some(fraction), Some(seed)) => {
            let mut stated = serde_yaml::Mapping::new();
            stated.insert(Yaml::from("fraction"), Yaml::from(fraction));
            stated.insert(Yaml::from("seed"), Yaml::from(seed));
            Ok(Yaml::Mapping(stated))
        }
        (false, None, _, _) => Err(Error(
            "train needs --eval-split-fraction F and --eval-split-seed N, --eval-split \
             scikit-learn, or --no-eval-split; transcript-label-trainer chooses no holdout"
                .to_string(),
        )),
    }
}

pub(crate) fn cmd_train(args: &Parsed) -> Result<i32> {
    let aspect = args.required("--aspect");
    let mut spec = serde_yaml::Mapping::new();
    spec.insert(
        serde_yaml::Value::from("eval_split"),
        stated_eval_split(args)?,
    );
    let eval_split = serde_json::to_value(jobs::eval_split(&spec, aspect)?)?;
    let model_id = args.text("--model");
    let backend = match model_id {
        Some(model) => model,
        None => jobs::SKLEARN_MODEL,
    };
    let training = jobs::training_from_flags(backend, |flag| args.text(flag))?;
    let metrics = match model::train(aspect, model_id, &training, &eval_split) {
        Ok(metrics) => metrics,
        Err(failure) => return Ok(report("train", failure)),
    };
    outln!("{}", dumps(&metrics));
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_run(args: &Parsed) -> Result<i32> {
    let job_file = args.positional("job_file");
    let job = match jobs::load(job_file) {
        Ok(job) => job,
        Err(error) => {
            eprintln!("run: {error}");
            return Ok(EXIT_FAILURE);
        }
    };
    if let Some(target) = args.text("--compute-target") {
        return stado::execute(job_file, &job, target);
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
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_evaluate(args: &Parsed) -> Result<i32> {
    let judge = if args.flag("--no-judge") {
        Some(false)
    } else {
        None
    };
    let verdict = match evaluate::evaluate(
        args.positional("name"),
        judge,
        args.text("--brama-model"),
        args.flag("--best"),
    ) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("evaluate: {error}");
            return Ok(EXIT_FAILURE);
        }
    };
    if args.flag("--json") {
        outln!("{}", dumps(&verdict));
    } else {
        print_verdict(&verdict);
    }
    Ok(reviewed_status(args, &verdict))
}

pub(crate) fn cmd_infer(args: &Parsed) -> Result<i32> {
    let suggestions = match model::infer(
        args.required("--aspect"),
        args.text("--session"),
        args.int("--limit"),
    ) {
        Ok(suggestions) => suggestions,
        Err(error) => {
            eprintln!("infer: {error}");
            return Ok(EXIT_FAILURE);
        }
    };
    outln!("{}", dumps(&suggestions));
    Ok(EXIT_SUCCESS)
}
