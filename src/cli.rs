//! Command-line interface: train, run, infer, evaluate, info, autolabel.
//!
//! The parser is hand-rolled rather than pulled from a crate, because the
//! surface it has to reproduce is not "some CLI" but exactly the one argparse
//! produced: the same flags, the same defaults, the same `--help` bodies, the
//! same `error:` sentences and the same exit statuses. Those strings are
//! quoted by the README and parsed by callers, so they are the contract.

use std::fmt::Write as _;

use serde_json::Value;

use crate::util::{float_repr, json_text, json_truthy, Error, Result, TrainFailure};
use crate::{autolabel, brama, evaluate, goal, jobs, model, placement, stado};

/// `println!` that does not panic when the reader has closed the pipe.
///
/// `transcript-label-trainer info | head` is a normal thing to type, and a
/// Rust panic is not an answer to it; a CLI that has nowhere left to write
/// simply stops writing.
macro_rules! outln {
    () => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stdout());
    }};
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stdout(), $($arg)*);
    }};
}

const PROG: &str = "transcript-label-trainer";
/// argparse's `HelpFormatter._width`: the terminal width less two, and the
/// terminal falls back to 80 columns whenever stdout is not a terminal.
const WIDTH: usize = 78;
/// argparse's `max_help_position`, the column help text starts in.
const MAX_HELP_POSITION: usize = 24;

const TOP_DESCRIPTION: &str = "Train small local classifiers that predict Transcript Lake aspect \
     labels, and emit label suggestions. Never writes to the lake.";
const TRAINING_ROOT_HELP: &str = "override where model artifacts live; beats $TLT_HOME and the \
     Stado registry declaration";
const STORAGE_ROOT_HELP: &str = "override the lake data root; beats $LAKE_DATA and the Stado \
     registry declaration";
const HELP_HELP: &str = "show this help message and exit";

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
                        return Ok(top_error(&format!("argument {name}: expected one argument")));
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
        "info" => cmd_info(&parsed),
        "goal-model" => cmd_goal_model(&parsed),
        "goal-audit" => cmd_goal_audit(&parsed),
        other => Err(Error(format!("unknown command '{other}'"))),
    }
}

// ---------------------------------------------------------------- commands

fn cmd_train(args: &Parsed) -> Result<i32> {
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

fn cmd_run(args: &Parsed) -> Result<i32> {
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
    outln!("{}",
    dumps(&serde_json::json!({"eval_split": model::split_summary(&plan)})));

    let metrics = match model::run_job(&job, &plan) {
        Ok(metrics) => metrics,
        Err(failure) => return Ok(report("run", failure)),
    };
    outln!("{}", dumps(&metrics));
    Ok(0)
}

fn cmd_evaluate(args: &Parsed) -> Result<i32> {
    let judge = if args.flag("--no-judge") { Some(false) } else { None };
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

fn cmd_infer(args: &Parsed) -> Result<i32> {
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

fn cmd_autolabel(args: &Parsed) -> Result<i32> {
    let values: Vec<String> = args
        .text("--values")
        .unwrap_or_default()
        .split(',')
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();
    if values.is_empty() {
        eprintln!("autolabel: --values must name at least one allowed value");
        return Ok(1);
    }
    let summary = match autolabel::autolabel(
        args.text("--aspect").unwrap_or_default(),
        &values,
        args.text("--brama-model"),
        args.flag("--best"),
        args.int("--limit"),
        args.text("--runtime"),
    ) {
        Ok(summary) => summary,
        Err(error) => {
            eprintln!("autolabel: {error}");
            return Ok(1);
        }
    };
    outln!("{}", dumps(&summary));
    if args.flag("--best")
        && summary
            .pointer("/best_review/sensible")
            .and_then(Value::as_bool)
            != Some(true)
    {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn cmd_goal_model(args: &Parsed) -> Result<i32> {
    let root = placement::resolve_placement()
        .training_root
        .join("goal-model")
        .join("datasets");
    std::fs::create_dir_all(&root)?;
    let stamp = crate::util::now_iso().replace([':', '-'], "");
    let dataset = root.join(format!("reviewed-goals-{stamp}.jsonl"));
    let summary = goal::build_dataset(
        &dataset,
        count(args.int("--limit"), 1_500),
        args.text("--teacher-model"),
    )?;
    outln!("{}", dumps(&summary));
    let job = stado::execute_goal_model(
        &dataset,
        args.text("--compute-target").unwrap_or_default(),
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

fn cmd_goal_audit(args: &Parsed) -> Result<i32> {
    let review_model = match (args.flag("--best"), args.text("--brama-model")) {
        (true, None) => crate::brama::BEST_MODEL,
        (false, Some(model)) => model,
        (true, Some(_)) => return Err(Error("use either --best or --brama-model".to_string())),
        (false, None) => {
            return Err(Error(
                "goal-audit requires --best or --brama-model".to_string(),
            ))
        }
    };
    let result = goal::audit_predictions(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output").unwrap_or_default()),
        review_model,
    )?;
    outln!("{}", dumps(&result));
    Ok(i32::from(
        result.get("passed").and_then(Value::as_bool) != Some(true),
    ))
}

fn cmd_info(args: &Parsed) -> Result<i32> {
    let entries = model::info()?;
    if args.flag("--json") {
        outln!("{}",
        dumps(&serde_json::json!({
            "placement": placement::as_dict(),
            "aspects": entries,
        })));
        return Ok(0);
    }
    print_placement();
    print_info(&entries);
    Ok(0)
}

/// Print a training failure the way the Python did, and pick its exit status:
/// 2 when the lake simply has too little labeled data, 1 otherwise.
fn report(command: &str, failure: TrainFailure) -> i32 {
    match failure {
        TrainFailure::NotEnoughData(message) => {
            eprintln!("{command}: {message}");
            2
        }
        TrainFailure::Failed(error) => {
            eprintln!("{command}: {error}");
            1
        }
    }
}


fn count(value: Option<i64>, fallback: usize) -> usize {
    match value {
        Some(value) if value >= 0 => value as usize,
        Some(_) => 0,
        None => fallback,
    }
}

// ---------------------------------------------------------------- printers

static NULL: Value = Value::Null;

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&NULL)
}

fn text(value: &Value) -> String {
    json_text(value)
}

fn print_info(entries: &[Value]) {
    if entries.is_empty() {
        outln!("no trained aspects under {}", model::models_dir().display());
        return;
    }
    for entry in entries {
        let artifacts: &[Value] = match field(entry, "artifacts") {
            Value::Array(artifacts) => artifacts.as_slice(),
            _ => &[],
        };
        if artifacts.is_empty() {
            outln!("{}: no trained artifacts in {}",
            text(field(entry, "aspect")),
            text(field(entry, "dir")));
            continue;
        }
        let active = text(field(entry, "active"));
        outln!("{} (active backend: {active}):", text(field(entry, "aspect")));
        for artifact in artifacts {
            let metrics = field(artifact, "metrics");
            let backend = text(field(artifact, "backend"));
            let marker = if backend == active { "*" } else { " " };
            let job = field(metrics, "job");
            if json_truthy(job) {
                outln!("    job:     {} — {} (evaluator: {})",
                text(field(job, "name")),
                text(field(job, "task")),
                text(field(job, "evaluator")));
            }
            let quality = if backend == "sklearn" {
                let cv = field(metrics, "cv_accuracy");
                if cv.is_null() {
                    "cv_accuracy=n/a".to_string()
                } else {
                    format!(
                        "cv_accuracy={} ({}-fold)",
                        text(cv),
                        text(field(metrics, "cv_folds"))
                    )
                }
            } else {
                let hyperparameters = field(metrics, "hyperparameters");
                let accuracy = field(field(metrics, "in_training_eval"), "accuracy");
                let mut quality = if accuracy.is_null() {
                    "in_training_accuracy=n/a".to_string()
                } else {
                    format!("in_training_accuracy={}", text(accuracy))
                };
                let _ = write!(
                    quality,
                    " ({}, epochs={}, lr={}, device={})",
                    text(field(metrics, "base_model")),
                    text(field(hyperparameters, "epochs")),
                    text(field(hyperparameters, "lr")),
                    text(field(metrics, "device"))
                );
                quality
            };
            outln!(" {marker} {backend}: {} sessions trained on, classes={}, {quality}\n\
             \x20   model:   {}\n\
             \x20   trained: {}",
            text(field(metrics, "n_sessions")),
            repr(field(metrics, "classes")),
            text(field(metrics, "model_path")),
            text(field(metrics, "trained_at")));
            let holdout = field(metrics, "holdout_evaluation");
            let split = field(metrics, "eval_split");
            if json_truthy(holdout) {
                outln!("    frozen holdout: accuracy={} on {} session(s), seed={}, fraction={}",
                text(field(holdout, "accuracy")),
                text(field(holdout, "n_sessions")),
                text(field(split, "seed")),
                text(field(split, "fraction")));
            } else if json_truthy(split)
                && !split.get("enabled").map(json_truthy).unwrap_or(true)
            {
                outln!("    frozen holdout: disabled by eval_split: false");
            }
        }
    }
}

/// Where Stado puts this run — and, when it could not, why it is local.
fn print_placement() {
    let resolved = placement::resolve_placement();
    outln!("placement:");
    outln!("    source:        {}", resolved.source);
    outln!("    training host: {}",
    resolved
        .training_host
        .as_deref()
        .filter(|host| !host.is_empty())
        .unwrap_or("undeclared"));
    outln!("    training root: {}", resolved.training_root.display());
    outln!("    storage root:  {}", resolved.storage_root.display());
    if resolved.source == "local-fallback" {
        outln!("    fallback:      {}", resolved.detail);
    }
    outln!();
}

/// The evaluate report: frozen-holdout scores, then the teacher's verdict.
fn print_verdict(verdict: &Value) {
    let split = field(verdict, "eval_split");
    let holdout = field(verdict, "holdout_evaluation");
    outln!("{} (aspect: {}, backend: {}):",
    text(field(verdict, "name")),
    text(field(verdict, "aspect")),
    text(field(verdict, "backend")));
    outln!("    frozen split:  {} session(s), fraction={}, seed={}, created {}\n\
     \x20   split file:    {}",
    text(field(split, "frozen_sessions")),
    text(field(split, "fraction")),
    text(field(split, "seed")),
    text(field(split, "created_at")),
    text(field(split, "path")));
    if json_truthy(field(split, "missing_ground_truth")) {
        outln!("    unlabeled now: {} (excluded)",
        text(field(split, "missing_ground_truth")));
    }
    if json_truthy(field(split, "skipped_no_text")) {
        outln!("    without text:  {} (excluded)",
        text(field(split, "skipped_no_text")));
    }
    outln!("    holdout:       accuracy={} on {} session(s)",
    text(field(holdout, "accuracy")),
    text(field(holdout, "n_sessions")));
    let correct = field(holdout, "correct");
    if let Value::Object(counts) = field(holdout, "counts") {
        let mut values: Vec<&String> = counts.keys().collect();
        values.sort_unstable();
        for value in values {
            let scored = correct.get(value.as_str()).unwrap_or(&Value::Null);
            let scored = if scored.is_null() { "0".to_string() } else { text(scored) };
            outln!("        {value}: {scored}/{} correct",
            text(counts.get(value).unwrap_or(&NULL)));
        }
    }
    if let Value::Array(pairs) = field(holdout, "confusion") {
        for pair in pairs {
            outln!("        confused {} -> {} ({}x)",
            text(field(pair, "gold")),
            text(field(pair, "predicted")),
            text(field(pair, "n")));
        }
    }
    let judge = field(verdict, "judge");
    if !json_truthy(field(judge, "enabled")) {
        outln!("    judge:         skipped (--no-judge)");
        return;
    }
    outln!("    judge:         {} calls {}/{} prediction(s) acceptable \
     (agreement_rate={}, failed={})",
    text(field(judge, "model")),
    text(field(judge, "acceptable")),
    text(field(judge, "judged")),
    text(field(judge, "agreement_rate")),
    text(field(judge, "failed")));
    let best = field(verdict, "best_review");
    if json_truthy(field(best, "enabled")) {
        outln!(
            "    best review:   {} reviewed={}, labels nonsensical={}, \
             judge opinions nonsensical={}, failed={}, sensible={}",
            text(field(best, "model")),
            text(field(best, "reviewed")),
            text(field(best, "label_nonsensical")),
            text(field(best, "judge_nonsensical")),
            text(field(best, "failed")),
            text(field(best, "sensible"))
        );
    }
    if let Value::Array(records) = field(verdict, "sessions") {
        for record in records {
            let mark = if text(field(record, "verdict")) == evaluate::JUDGE_VALUES[0] {
                "ok "
            } else {
                "bad"
            };
            outln!("        {mark} {}: gold={} predicted={} ({})",
            text(field(record, "session_id")),
            text(field(record, "gold")),
            text(field(record, "prediction")),
            text(field(record, "confidence")));
            if json_truthy(field(record, "best_review")) {
                outln!("             final review={}", text(field(record, "best_review")));
            }
        }
    }
    if let Value::Array(failures) = field(verdict, "failures") {
        for failure in failures {
            outln!("        err {}: {}",
            text(field(failure, "session_id")),
            text(field(failure, "error")));
        }
    }
    outln!("    verdict file:  {}", text(field(verdict, "judge_path")));
}

// -------------------------------------------------------------- json output

/// `json.dumps(value, indent=2)`, including `ensure_ascii`.
///
/// Rust would print `ą` and `2e-5` where Python printed `\u0105` and `2e-05`,
/// and these bytes land in files the lake and the operator already have.
fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_json(&mut out, value, 0);
    out
}

fn write_json(out: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => out.push_str(&float_repr(float)),
            _ => out.push_str(&number.to_string()),
        },
        Value::String(string) => write_json_string(out, string),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                indent(out, depth + 1);
                write_json(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                indent(out, depth + 1);
                write_json_string(out, key);
                out.push_str(": ");
                write_json(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push('}');
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth * 2 {
        out.push(' ');
    }
}

fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character if (character as u32) < 0x7f => out.push(character),
            character => {
                let code = character as u32;
                if code > 0xFFFF {
                    let value = code - 0x10000;
                    let high = 0xD800 + (value >> 10);
                    let low = 0xDC00 + (value & 0x3FF);
                    let _ = write!(out, "\\u{high:04x}\\u{low:04x}");
                } else {
                    let _ = write!(out, "\\u{code:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Python's `repr()` of a value. `info` interpolates the class list, and
/// `['agent', 'data']` is what the README shows.
fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => float_repr(float),
            _ => number.to_string(),
        },
        Value::String(string) => repr_str(string),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, item)| format!("{}: {}", repr_str(key), repr(item)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn repr_str(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character == quote => {
                out.push('\\');
                out.push(character);
            }
            character => out.push(character),
        }
    }
    out.push(quote);
    out
}

// ------------------------------------------------------------ the arguments

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Flag,
    Text,
    Int,
    Float,
}

struct Opt {
    flag: &'static str,
    metavar: &'static str,
    kind: Kind,
    required: bool,
    /// Non-zero puts this option in a mutually exclusive group with every
    /// other option carrying the same number.
    group: u8,
    help: String,
}

struct Positional {
    name: &'static str,
    help: String,
}

struct Spec {
    name: &'static str,
    help: String,
    description: Option<String>,
    positionals: Vec<Positional>,
    opts: Vec<Opt>,
}

fn option(flag: &'static str, metavar: &'static str, kind: Kind, help: String) -> Opt {
    Opt { flag, metavar, kind, required: false, group: 0, help }
}

fn required(flag: &'static str, metavar: &'static str, kind: Kind, help: String) -> Opt {
    Opt { flag, metavar, kind, required: true, group: 0, help }
}

fn build_specs() -> Vec<Spec> {
    let fraction = float_repr(jobs::DEFAULT_EVAL_FRACTION);
    let seed = jobs::DEFAULT_EVAL_SEED;
    let teacher = brama::DEFAULT_MODEL;

    let train = Spec {
        name: "train",
        help: "train a classifier for one aspect".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required("--aspect", "ASPECT", Kind::Text, "aspect name, e.g. reviewed".to_string()),
            option(
                "--model",
                "HF_MODEL_ID",
                Kind::Text,
                "fine-tune this HuggingFace model instead of the default TF-IDF + \
                 logistic regression (requires the 'hf' extra); multilingual models \
                 such as distilbert-base-multilingual-cased fit the mixed \
                 Polish/English transcripts"
                    .to_string(),
            ),
            option("--epochs", "EPOCHS", Kind::Float, "HF training epochs (default: 3)".to_string()),
            option("--batch-size", "BATCH_SIZE", Kind::Int, "HF batch size (default: 8)".to_string()),
            option("--lr", "LR", Kind::Float, "HF learning rate (default: 2e-5)".to_string()),
            option(
                "--max-length",
                "MAX_LENGTH",
                Kind::Int,
                "HF tokenizer max tokens per session (default: 512)".to_string(),
            ),
            option(
                "--eval-split-fraction",
                "F",
                Kind::Float,
                format!(
                    "share of labeled sessions frozen out of training the first time \
                     this aspect is trained (default: {fraction}); later runs reuse \
                     the frozen eval-split.json unchanged"
                ),
            ),
            option(
                "--eval-split-seed",
                "N",
                Kind::Int,
                format!("seed that picks the frozen holdout (default: {seed})"),
            ),
            option(
                "--no-eval-split",
                "",
                Kind::Flag,
                "train on every labeled session, with no frozen holdout to evaluate on".to_string(),
            ),
        ],
    };

    let run = Spec {
        name: "run",
        help: "execute a declarative training job (YAML spec)".to_string(),
        description: Some(format!(
            "Execute a declarative training job. Two spec sections are on unless the \
             spec turns them off. 'eval_split' (fraction: {fraction}, seed: {seed}) \
             freezes a holdout of labeled sessions into \
             <training root>/models/<name>/eval-split.json the first time the job \
             runs; every later run reuses that file unchanged, trains on nothing in \
             it, and reports it under 'holdout_evaluation' in metrics.json. \
             'eval_split: false' trains on every labeled session. 'judge' (model: \
             {teacher}) names the Brama-routed teacher that 'evaluate' asks for a \
             verdict; 'judge: false' skips it. run prints the resolved job, then the \
             resolved split, then the metrics."
        )),
        positionals: vec![Positional {
            name: "job_file",
            help: "path to the job spec YAML".to_string(),
        }],
        opts: vec![option(
            "--compute-target",
            "COMPUTE_TARGET",
            Kind::Text,
            "export the selected lake rows, submit this run to that canonical \
             Stado compute target, follow it to completion, and run evaluate \
             --best after training"
                .to_string(),
        )],
    };

    let evaluate = Spec {
        name: "evaluate",
        help: "score a trained model on its frozen holdout and have a Brama teacher judge \
               whether the predictions are acceptable"
            .to_string(),
        description: Some(
            "Score the trained model on the frozen holdout in \
             <training root>/models/<name>/eval-split.json — the sessions training \
             never saw — and then send each of them to a Brama-routed teacher with \
             the model's prediction and the ground-truth label, asking whether the \
             prediction is acceptable. The verdict (agreement rate plus one record \
             per session) is written to <training root>/models/<name>/judge.json. A \
             Brama error fails only its own session and is counted; if not one \
             session could be judged, the gateway's own error is reported and the \
             exit status is nonzero — no verdict is invented and there is no local \
             fallback."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "name",
            help: "job name or aspect — the directory under <training root>/models/".to_string(),
        }],
        opts: vec![
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!(
                    "Brama-routed judge model (default: the job spec's judge.model, \
                     else {teacher})"
                ),
            ),
            option(
                "--no-judge",
                "",
                Kind::Flag,
                "report the frozen-holdout scores only, without asking the teacher".to_string(),
            ),
            option("--json", "", Kind::Flag, "print machine-readable JSON".to_string()),
            option(
                "--best",
                "",
                Kind::Flag,
                "after the configured judge, use Brama's -best route to audit \
                 whether every ground-truth label and judge opinion is sensible; \
                 exit nonzero when the audit finds an issue"
                    .to_string(),
            ),
        ],
    };

    let infer = Spec {
        name: "infer",
        help: "emit label suggestions for unlabeled sessions".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required("--aspect", "ASPECT", Kind::Text, "aspect name, e.g. reviewed".to_string()),
            Opt {
                group: 1,
                ..option(
                    "--session",
                    "SESSION",
                    Kind::Text,
                    "predict for one session id, labeled or not".to_string(),
                )
            },
            Opt {
                group: 1,
                ..option(
                    "--limit",
                    "LIMIT",
                    Kind::Int,
                    "cap the number of unlabeled sessions".to_string(),
                )
            },
        ],
    };

    let info = Spec {
        name: "info",
        help: "list trained aspects, artifacts, and metrics".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![option(
            "--json",
            "",
            Kind::Flag,
            "print machine-readable JSON".to_string(),
        )],
    };

    let autolabel = Spec {
        name: "autolabel",
        help: "label every unlabeled session for an aspect via a Brama teacher (zero-touch)"
            .to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required("--aspect", "ASPECT", Kind::Text, "aspect name, e.g. tasktype".to_string()),
            required(
                "--values",
                "VALUES",
                Kind::Text,
                "comma-separated allowed label values, e.g. bugfix,feature,chore,question"
                    .to_string(),
            ),
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama-routed teacher model (default: {teacher})"),
            ),
            option(
                "--best",
                "",
                Kind::Flag,
                "have Brama's -best route audit every proposed label before it \
                 reaches Transcript Lake; reject nonsensical labels and exit \
                 nonzero when the audit finds an issue"
                    .to_string(),
            ),
            option("--limit", "LIMIT", Kind::Int, "cap the number of sessions labeled".to_string()),
            option("--runtime", "RUNTIME", Kind::Text, "only sessions of this runtime".to_string()),
        ],
    };

    let goal_model = Spec {
        name: "goal-model",
        help: "build and train the reviewed Jeden goal model on Stado".to_string(),
        description: Some(
            "Read only privacy-masked Transcript Lake events, use a Brama teacher \
             to label task goals, require an independent Brama -best review, then \
             train on the named exclusive Stado GPU target. The held-out gold \
             predictions must all pass a second -best audit before GGUF artifacts \
             are published."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            required(
                "--compute-target",
                "COMPUTE_TARGET",
                Kind::Text,
                "canonical Stado GPU target that trains and exports the model".to_string(),
            ),
            option(
                "--limit",
                "LIMIT",
                Kind::Int,
                "maximum teacher-labeled candidates (default: 1500)".to_string(),
            ),
            option(
                "--teacher-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama-routed goal teacher (default: {teacher})"),
            ),
        ],
    };

    let goal_audit = Spec {
        name: "goal-audit",
        help: "apply the final Brama audit to held-out goal predictions".to_string(),
        description: None,
        positionals: vec![Positional {
            name: "predictions",
            help: "JSONL containing message, reference goal, and student output".to_string(),
        }],
        opts: vec![
            required(
                "--output",
                "PATH",
                Kind::Text,
                "write the complete independent audit record here".to_string(),
            ),
            option(
                "--best",
                "",
                Kind::Flag,
                "require Brama's strongest operator-approved subscription route".to_string(),
            ),
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                "use an explicit Brama-routed model when the best subscription is unavailable"
                    .to_string(),
            ),
        ],
    };

    vec![
        train,
        run,
        evaluate,
        infer,
        info,
        autolabel,
        goal_model,
        goal_audit,
    ]
}

#[derive(Default)]
struct Parsed {
    flags: Vec<&'static str>,
    texts: Vec<(&'static str, String)>,
    ints: Vec<(&'static str, i64)>,
    floats: Vec<(&'static str, f64)>,
    positionals: Vec<String>,
}

impl Parsed {
    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|seen| *seen == name)
    }

    fn text(&self, name: &str) -> Option<&str> {
        self.texts
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.as_str())
    }

    fn int(&self, name: &str) -> Option<i64> {
        self.ints.iter().find(|(key, _)| *key == name).map(|(_, value)| *value)
    }

    fn float(&self, name: &str) -> Option<f64> {
        self.floats.iter().find(|(key, _)| *key == name).map(|(_, value)| *value)
    }

    fn positional(&self, index: usize) -> &str {
        self.positionals.get(index).map(String::as_str).unwrap_or("")
    }

    fn has(&self, opt: &Opt) -> bool {
        match opt.kind {
            Kind::Flag => self.flag(opt.flag),
            Kind::Text => self.text(opt.flag).is_some(),
            Kind::Int => self.int(opt.flag).is_some(),
            Kind::Float => self.float(opt.flag).is_some(),
        }
    }
}

enum Outcome {
    Parsed(Parsed),
    Help,
    Error(String),
}

fn parse_sub(spec: &Spec, argv: &[String]) -> Outcome {
    let mut parsed = Parsed::default();
    let mut extras: Vec<String> = Vec::new();
    let mut in_group: Vec<(u8, &'static str)> = Vec::new();
    let mut index = 0;

    while index < argv.len() {
        let arg = argv[index].as_str();
        if arg == "-h" || arg == "--help" {
            return Outcome::Help;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg, None),
        };
        let Some(opt) = spec.opts.iter().find(|opt| opt.flag == name) else {
            if looks_like_option(arg) {
                extras.push(arg.to_string());
            } else {
                parsed.positionals.push(arg.to_string());
            }
            index += 1;
            continue;
        };

        if opt.kind == Kind::Flag {
            if let Some(value) = inline {
                return Outcome::Error(format!(
                    "argument {}: ignored explicit argument '{value}'",
                    opt.flag
                ));
            }
            parsed.flags.push(opt.flag);
            index += 1;
        } else {
            let value = match inline {
                Some(value) => {
                    index += 1;
                    value
                }
                None => match argv.get(index + 1) {
                    Some(next) if !looks_like_option(next) => {
                        index += 2;
                        next.clone()
                    }
                    _ => {
                        return Outcome::Error(format!(
                            "argument {}: expected one argument",
                            opt.flag
                        ));
                    }
                },
            };
            match opt.kind {
                Kind::Text => parsed.texts.push((opt.flag, value)),
                Kind::Int => match value.trim().parse::<i64>() {
                    Ok(number) => parsed.ints.push((opt.flag, number)),
                    Err(_) => {
                        return Outcome::Error(format!(
                            "argument {}: invalid int value: '{value}'",
                            opt.flag
                        ));
                    }
                },
                Kind::Float => match value.trim().parse::<f64>() {
                    Ok(number) => parsed.floats.push((opt.flag, number)),
                    Err(_) => {
                        return Outcome::Error(format!(
                            "argument {}: invalid float value: '{value}'",
                            opt.flag
                        ));
                    }
                },
                Kind::Flag => {}
            }
        }

        if opt.group != 0 {
            if let Some((_, other)) = in_group
                .iter()
                .find(|(group, other)| *group == opt.group && *other != opt.flag)
            {
                return Outcome::Error(format!(
                    "argument {}: not allowed with argument {other}",
                    opt.flag
                ));
            }
            in_group.push((opt.group, opt.flag));
        }
    }

    if parsed.positionals.len() > spec.positionals.len() {
        extras.extend(parsed.positionals.split_off(spec.positionals.len()));
    }
    if !extras.is_empty() {
        return Outcome::Error(format!("unrecognized arguments: {}", extras.join(" ")));
    }

    let mut missing: Vec<&str> = Vec::new();
    for (index, positional) in spec.positionals.iter().enumerate() {
        if parsed.positionals.len() <= index {
            missing.push(positional.name);
        }
    }
    for opt in &spec.opts {
        if opt.required && !parsed.has(opt) {
            missing.push(opt.flag);
        }
    }
    if !missing.is_empty() {
        return Outcome::Error(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    Outcome::Parsed(parsed)
}

/// A token argparse would read as an option rather than as a value: it starts
/// with a dash, is longer than a bare `-`, and is not a negative number.
fn looks_like_option(arg: &str) -> bool {
    if !arg.starts_with('-') || arg.len() < 2 {
        return false;
    }
    arg[1..].parse::<f64>().is_err()
}

// ----------------------------------------------------------------- the help

struct Action {
    invocation: String,
    help: Option<String>,
    indent: usize,
}

fn top_usage_parts(specs: &[Spec]) -> (Vec<String>, Vec<String>) {
    let optionals = vec![
        "[-h]".to_string(),
        "[--training-root PATH]".to_string(),
        "[--storage-root PATH]".to_string(),
    ];
    (optionals, vec![format!("{} ...", choices_metavar(specs))])
}

fn choices_metavar(specs: &[Spec]) -> String {
    format!(
        "{{{}}}",
        specs.iter().map(|spec| spec.name).collect::<Vec<_>>().join(",")
    )
}

fn top_help(specs: &[Spec]) -> String {
    let (optionals, positionals) = top_usage_parts(specs);
    let mut choices = vec![Action {
        invocation: choices_metavar(specs),
        help: None,
        indent: 2,
    }];
    for spec in specs {
        choices.push(Action {
            invocation: spec.name.to_string(),
            help: Some(spec.help.clone()),
            indent: 4,
        });
    }
    let options = vec![
        Action { invocation: "-h, --help".to_string(), help: Some(HELP_HELP.to_string()), indent: 2 },
        Action {
            invocation: "--training-root PATH".to_string(),
            help: Some(TRAINING_ROOT_HELP.to_string()),
            indent: 2,
        },
        Action {
            invocation: "--storage-root PATH".to_string(),
            help: Some(STORAGE_ROOT_HELP.to_string()),
            indent: 2,
        },
    ];
    assemble(
        PROG,
        &optionals,
        &positionals,
        Some(TOP_DESCRIPTION),
        &choices,
        &options,
    )
}

fn sub_usage_parts(spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut optionals = vec!["[-h]".to_string()];
    let mut grouped: Vec<u8> = Vec::new();
    for opt in &spec.opts {
        if opt.group != 0 {
            if grouped.contains(&opt.group) {
                continue;
            }
            grouped.push(opt.group);
            let members: Vec<String> = spec
                .opts
                .iter()
                .filter(|other| other.group == opt.group)
                .map(bare_usage)
                .collect();
            optionals.push(format!("[{}]", members.join(" | ")));
            continue;
        }
        let bare = bare_usage(opt);
        optionals.push(if opt.required { bare } else { format!("[{bare}]") });
    }
    let positionals = spec
        .positionals
        .iter()
        .map(|positional| positional.name.to_string())
        .collect();
    (optionals, positionals)
}

fn bare_usage(opt: &Opt) -> String {
    if opt.kind == Kind::Flag {
        opt.flag.to_string()
    } else {
        format!("{} {}", opt.flag, opt.metavar)
    }
}

fn sub_help(spec: &Spec) -> String {
    let prog = format!("{PROG} {}", spec.name);
    let (optionals, positional_parts) = sub_usage_parts(spec);
    let positionals: Vec<Action> = spec
        .positionals
        .iter()
        .map(|positional| Action {
            invocation: positional.name.to_string(),
            help: Some(positional.help.clone()),
            indent: 2,
        })
        .collect();
    let mut options = vec![Action {
        invocation: "-h, --help".to_string(),
        help: Some(HELP_HELP.to_string()),
        indent: 2,
    }];
    for opt in &spec.opts {
        options.push(Action {
            invocation: bare_usage(opt),
            help: Some(opt.help.clone()),
            indent: 2,
        });
    }
    assemble(
        &prog,
        &optionals,
        &positional_parts,
        spec.description.as_deref(),
        &positionals,
        &options,
    )
}

fn assemble(
    prog: &str,
    optionals: &[String],
    positional_parts: &[String],
    description: Option<&str>,
    positionals: &[Action],
    options: &[Action],
) -> String {
    let mut out = format_usage(prog, optionals, positional_parts);
    let action_max_length = positionals
        .iter()
        .chain(options.iter())
        .map(|action| action.invocation.chars().count() + action.indent)
        .max()
        .unwrap_or(0);
    let help_position = (action_max_length + 2).min(MAX_HELP_POSITION);

    if let Some(description) = description {
        out.push('\n');
        for line in wrap(description, WIDTH) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    if !positionals.is_empty() {
        out.push('\n');
        out.push_str("positional arguments:\n");
        for action in positionals {
            format_action(action, help_position, &mut out);
        }
    }
    if !options.is_empty() {
        out.push('\n');
        out.push_str("options:\n");
        for action in options {
            format_action(action, help_position, &mut out);
        }
    }
    out
}

fn format_action(action: &Action, help_position: usize, out: &mut String) {
    let header_length = action.invocation.chars().count();
    let Some(help) = action.help.as_deref().filter(|help| !help.trim().is_empty()) else {
        push_spaces(out, action.indent);
        out.push_str(&action.invocation);
        out.push('\n');
        return;
    };
    let action_width = help_position.saturating_sub(action.indent + 2);
    let help_width = WIDTH.saturating_sub(help_position).max(11);
    let lines = wrap(help, help_width);

    push_spaces(out, action.indent);
    out.push_str(&action.invocation);
    if header_length <= action_width {
        push_spaces(out, action_width - header_length + 2);
    } else {
        out.push('\n');
        push_spaces(out, help_position);
    }
    let mut lines = lines.into_iter();
    if let Some(first) = lines.next() {
        out.push_str(&first);
    }
    out.push('\n');
    for line in lines {
        push_spaces(out, help_position);
        out.push_str(&line);
        out.push('\n');
    }
}

fn push_spaces(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push(' ');
    }
}

/// argparse's usage line, wrapping included: one line while it fits, then the
/// option parts folded under the program name.
fn format_usage(prog: &str, optionals: &[String], positionals: &[String]) -> String {
    let prefix = "usage: ";
    let mut single = prog.to_string();
    for part in optionals.iter().chain(positionals.iter()) {
        single.push(' ');
        single.push_str(part);
    }
    if prefix.len() + single.chars().count() <= WIDTH {
        return format!("{prefix}{single}\n");
    }

    let prog_length = prog.chars().count();
    let lines: Vec<String> = if prefix.len() + prog_length <= WIDTH * 3 / 4 {
        let indent = " ".repeat(prefix.len() + prog_length + 1);
        if !optionals.is_empty() {
            let mut head = vec![prog.to_string()];
            head.extend(optionals.iter().cloned());
            let mut lines = usage_lines(&head, &indent, Some(prefix));
            lines.extend(usage_lines(positionals, &indent, None));
            lines
        } else if !positionals.is_empty() {
            let mut head = vec![prog.to_string()];
            head.extend(positionals.iter().cloned());
            usage_lines(&head, &indent, Some(prefix))
        } else {
            vec![prog.to_string()]
        }
    } else {
        let indent = " ".repeat(prefix.len());
        let mut parts = optionals.to_vec();
        parts.extend(positionals.iter().cloned());
        let mut lines = usage_lines(&parts, &indent, None);
        if lines.len() > 1 {
            lines = usage_lines(optionals, &indent, None);
            lines.extend(usage_lines(positionals, &indent, None));
        }
        let mut folded = vec![prog.to_string()];
        folded.extend(lines);
        folded
    };
    format!("{prefix}{}\n", lines.join("\n"))
}

fn usage_lines(parts: &[String], indent: &str, prefix: Option<&str>) -> Vec<String> {
    let indent_length = indent.chars().count();
    let mut lines: Vec<String> = Vec::new();
    let mut line: Vec<&str> = Vec::new();
    let mut line_length = match prefix {
        Some(prefix) => prefix.chars().count().saturating_sub(1),
        None => indent_length.saturating_sub(1),
    };
    for part in parts {
        let part_length = part.chars().count();
        if line_length + 1 + part_length > WIDTH && !line.is_empty() {
            lines.push(format!("{indent}{}", line.join(" ")));
            line.clear();
            line_length = indent_length.saturating_sub(1);
        }
        line.push(part);
        line_length += 1 + part_length;
    }
    if !line.is_empty() {
        lines.push(format!("{indent}{}", line.join(" ")));
    }
    if prefix.is_some() {
        if let Some(first) = lines.first_mut() {
            *first = first.chars().skip(indent_length).collect();
        }
    }
    lines
}

/// `textwrap.wrap`: greedy fill over whitespace- and hyphen-separated chunks,
/// dropping the whitespace a line breaks on.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut chunks: Vec<String> = Vec::new();
    for (index, word) in text.split_whitespace().enumerate() {
        if index > 0 {
            chunks.push(" ".to_string());
        }
        chunks.extend(hyphen_chunks(word));
    }

    let mut lines: Vec<String> = Vec::new();
    let mut position = 0;
    while position < chunks.len() {
        if !lines.is_empty() && chunks[position] == " " {
            position += 1;
            continue;
        }
        let mut current: Vec<String> = Vec::new();
        let mut length = 0;
        while position < chunks.len() {
            let chunk_length = chunks[position].chars().count();
            if length + chunk_length > width {
                break;
            }
            length += chunk_length;
            current.push(chunks[position].clone());
            position += 1;
        }
        // One chunk wider than the whole line: cut it and keep the remainder.
        if position < chunks.len() && chunks[position].chars().count() > width {
            let room = width.saturating_sub(length).max(1);
            let chunk = chunks[position].clone();
            current.push(chunk.chars().take(room).collect());
            chunks[position] = chunk.chars().skip(room).collect();
        }
        while current.last().is_some_and(|chunk| chunk == " ") {
            current.pop();
        }
        if current.is_empty() {
            position += 1;
            continue;
        }
        lines.push(current.concat());
    }
    lines
}

/// A word split at the hyphens `textwrap` is willing to break on, the hyphen
/// staying with the chunk before it.
fn hyphen_chunks(word: &str) -> Vec<String> {
    let characters: Vec<char> = word.chars().collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0;
    for index in 0..characters.len() {
        if characters[index] != '-' || !breakable_hyphen(&characters, index) {
            continue;
        }
        chunks.push(characters[start..=index].iter().collect());
        start = index + 1;
    }
    if start < characters.len() {
        chunks.push(characters[start..].iter().collect());
    }
    if chunks.is_empty() {
        chunks.push(word.to_string());
    }
    chunks
}

fn breakable_hyphen(characters: &[char], index: usize) -> bool {
    let letter = |at: usize| {
        characters
            .get(at)
            .is_some_and(|character| character.is_alphabetic() || *character == '_')
    };
    let behind = (index >= 2 && letter(index - 1) && letter(index - 2))
        || (index >= 3
            && letter(index - 1)
            && characters.get(index - 2) == Some(&'-')
            && letter(index - 3));
    if !behind || !letter(index + 1) {
        return false;
    }
    letter(index + 2) || (characters.get(index + 2) == Some(&'-') && letter(index + 3))
}

fn top_error(message: &str) -> i32 {
    let specs = build_specs();
    let (optionals, positionals) = top_usage_parts(&specs);
    eprint!("{}", format_usage(PROG, &optionals, &positionals));
    eprintln!("{PROG}: error: {message}");
    2
}

fn sub_error(spec: &Spec, message: &str) -> i32 {
    let prog = format!("{PROG} {}", spec.name);
    let (optionals, positionals) = sub_usage_parts(spec);
    eprint!("{}", format_usage(&prog, &optionals, &positionals));
    eprintln!("{prog}: error: {message}");
    2
}
