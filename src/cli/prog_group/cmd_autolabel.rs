use super::*;

/// The port `--port` means when it is left out: the operating system picks a
/// free one when asked for this port.
const SYSTEM_CHOSEN_PORT: i64 = 0; // https://doc.rust-lang.org/std/net/struct.TcpListener.html#method.bind "Binding with a port number of 0 will request that the OS assigns a port to this listener"

/// The status a reviewed command exits with: success only when `--best` was
/// not asked for, or its audit called every result sensible.
pub(crate) fn reviewed_status(args: &Parsed, summary: &Value) -> i32 {
    match !args.flag("--best")
        || summary.pointer("/best_review/sensible") == Some(&Value::Bool(true))
    {
        true => EXIT_SUCCESS,
        false => EXIT_FAILURE,
    }
}

/// The status an audit exits with: success only when the audit passed.
fn audited_status(result: &Value) -> i32 {
    match result["passed"] == Value::Bool(true) {
        true => EXIT_SUCCESS,
        false => EXIT_FAILURE,
    }
}

/// The reviewer an audit command names: `--best` or `--brama-model`, exactly one.
fn audit_reviewer<'a>(args: &'a Parsed, command: &str) -> Result<&'a str> {
    match (args.flag("--best"), args.text("--brama-model")) {
        (true, None) => Ok(crate::brama::BEST_MODEL),
        (false, Some(model)) => Ok(model),
        (true, Some(_)) => Err(Error("use either --best or --brama-model".to_string())),
        (false, None) => Err(Error(format!("{command} requires --best or --brama-model"))),
    }
}

pub(crate) fn cmd_autolabel(args: &Parsed) -> Result<i32> {
    let values: Vec<String> = args
        .required("--values")
        .split(',')
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect();
    if values.is_empty() {
        eprintln!("autolabel: --values must name at least one allowed value");
        return Ok(EXIT_FAILURE);
    }
    let summary = match autolabel::autolabel(
        args.required("--aspect"),
        &values,
        args.text("--brama-model"),
        args.flag("--best"),
        args.int("--limit"),
        args.text("--runtime"),
    ) {
        Ok(summary) => summary,
        Err(error) => {
            eprintln!("autolabel: {error}");
            return Ok(EXIT_FAILURE);
        }
    };
    outln!("{}", dumps(&summary));
    Ok(reviewed_status(args, &summary))
}

pub(crate) fn cmd_aspect_discover(args: &Parsed) -> Result<i32> {
    let summary = match discover::discover(
        args.int("--limit"),
        args.text("--brama-model"),
        args.flag("--best"),
        args.int("--max-aspects"),
        args.text("--runtime"),
    ) {
        Ok(summary) => summary,
        Err(error) => {
            eprintln!("aspect-discover: {error}");
            return Ok(EXIT_FAILURE);
        }
    };
    outln!("{}", dumps(&summary));
    Ok(reviewed_status(args, &summary))
}

/// Where a dataset curated now is written under the training root.
fn stamped_dataset(model: &str, prefix: &str) -> Result<std::path::PathBuf> {
    let root = placement::resolve_placement()
        .training_root
        .join(model)
        .join("datasets");
    std::fs::create_dir_all(&root)?;
    let stamp = crate::util::now_iso().replace([':', '-'], "");
    Ok(root.join(format!("{prefix}-{stamp}.jsonl")))
}

pub(crate) fn cmd_goal_model(args: &Parsed) -> Result<i32> {
    // The job's own settings are checked before curation spends any Brama
    // call: a run that would be refused at submission is refused first.
    let ster_options = stado::stated_ster_options(args.required("--ster-options"))?;
    let dataset = stamped_dataset("goal-model", "reviewed-goals")?;
    let summary = goal::build_dataset(&dataset, args.text("--teacher-model"))?;
    outln!("{}", dumps(&summary));
    let job =
        stado::execute_goal_model(&dataset, args.required("--compute-target"), &ster_options)?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_goal_audit(args: &Parsed) -> Result<i32> {
    let result = goal::audit_predictions(
        std::path::Path::new(args.positional("predictions")),
        std::path::Path::new(args.required("--output")),
        audit_reviewer(args, "goal-audit")?,
    )?;
    outln!("{}", dumps(&result));
    Ok(audited_status(&result))
}

pub(crate) fn cmd_lifecycle_review(args: &Parsed) -> Result<i32> {
    // Labels are ground truth for every model trained from them, so the
    // reviewer is the strongest operator-approved route unless one is named.
    let model = match args.text("--brama-model") {
        Some(model) => model,
        None => crate::brama::BEST_MODEL,
    };
    let limit = match args.int("--limit") {
        Some(limit) => Some(usize::try_from(limit).map_err(|error| {
            Error(format!(
                "--limit must be a count of rows, not {limit}: {error}"
            ))
        })?),
        None => None,
    };
    let result = crate::lifecycle::review_dataset(
        std::path::Path::new(args.positional("input")),
        std::path::Path::new(args.required("--output")),
        args.required("--split"),
        model,
        limit,
    )?;
    outln!("{}", dumps(&result));
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_lifecycle_model(args: &Parsed) -> Result<i32> {
    let job = stado::execute_lifecycle_model(
        std::path::Path::new(args.positional("train")),
        std::path::Path::new(args.positional("eval")),
        args.required("--compute-target"),
        args.required("--brama-url"),
        args.required("--ster-options"),
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_humanizer_model(args: &Parsed) -> Result<i32> {
    let hf_repo = std::env::var("HUMANIZER_HF_REPO")
        .map_err(|error| Error(format!("cannot read required HUMANIZER_HF_REPO: {error}")))?;
    if hf_repo.trim().is_empty() {
        return Err(Error(
            "HUMANIZER_HF_REPO must explicitly name a private destination".into(),
        ));
    }
    // The job's training settings are checked before the corpus is exported:
    // a run that would be refused at submission is refused first.
    let training = stado::HumanizerTraining {
        ster_options: stado::stated_ster_options(args.required("--ster-options"))?.to_string(),
    };
    let targets = stamped_dataset("humanizer-model", "lukasz-targets")?;
    let summary = crate::humanizer::export_targets(&targets)?;
    outln!("{}", dumps(&summary));
    let job = stado::execute_humanizer_model(
        &targets,
        args.required("--compute-target"),
        &hf_repo,
        &training,
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_lifecycle_audit(args: &Parsed) -> Result<i32> {
    let result = crate::lifecycle::audit_predictions(
        std::path::Path::new(args.positional("predictions")),
        std::path::Path::new(args.required("--output")),
        audit_reviewer(args, "lifecycle-audit")?,
    )?;
    outln!("{}", dumps(&result));
    Ok(audited_status(&result))
}

pub(crate) fn cmd_info(args: &Parsed) -> Result<i32> {
    let entries = model::info()?;
    if args.flag("--json") {
        outln!(
            "{}",
            dumps(&serde_json::json!({
                "placement": placement::as_dict(),
                "aspects": entries,
            }))
        );
        return Ok(EXIT_SUCCESS);
    }
    print_placement();
    print_info(&entries);
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_corpus_adopt(args: &Parsed) -> Result<i32> {
    let report = corpus::adopt(std::path::Path::new(args.positional("bundle")))?;
    if args.flag("--json") {
        outln!("{}", dumps(&report));
        return Ok(EXIT_SUCCESS);
    }
    outln!(
        "selected corpus {} for aspect '{}' ({} imported, {} unchanged, {} conflicting, {} rejected)",
        json_text(&report["corpusId"]),
        json_text(&report["aspect"]),
        json_text(&report["imported"]),
        json_text(&report["unchanged"]),
        json_text(&report["conflicting"]),
        json_text(&report["rejected"]),
    );
    outln!("retained bundle: {}", json_text(&report["bundlePath"]));
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_corpus_status(args: &Parsed) -> Result<i32> {
    let report = corpus::status()?;
    if args.flag("--json") {
        outln!("{}", dumps(&report));
        return Ok(EXIT_SUCCESS);
    }
    match report.get("selected").filter(|value| !value.is_null()) {
        Some(selected) => outln!(
            "selected corpus {} for aspect '{}' ({} records)\nretained bundle: {}",
            json_text(&selected["id"]),
            json_text(&selected["aspect"]),
            json_text(&selected["records"]),
            json_text(&selected["bundlePath"]),
        ),
        None => outln!("no adopted corpus is selected"),
    }
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_gui(args: &Parsed) -> Result<i32> {
    let bind = match args.text("--bind") {
        Some(bind) => bind.to_string(),
        None => std::net::Ipv4Addr::LOCALHOST.to_string(),
    };
    let port = match args.int("--port") {
        Some(port) => port,
        None => SYSTEM_CHOSEN_PORT,
    };
    gui::serve(&bind, port)
}

pub(crate) fn cmd_onboarding(args: &Parsed) -> Result<i32> {
    onboarding::onboarding(
        args.flag("--reset"),
        args.flag("--yes"),
        args.flag("--json"),
        args.text("--corpus"),
        args.flag("--skip-corpus"),
    )
}

/// Print a training failure the way the Python did, and pick its exit status:
/// a missing prerequisite when the lake simply has too little labeled data,
/// a failure otherwise.
pub(crate) fn report(command: &str, failure: TrainFailure) -> i32 {
    match failure {
        TrainFailure::NotEnoughData(message) => {
            eprintln!("{command}: {message}");
            EXIT_PREREQUISITE_MISSING
        }
        TrainFailure::Failed(error) => {
            eprintln!("{command}: {error}");
            EXIT_FAILURE
        }
    }
}
