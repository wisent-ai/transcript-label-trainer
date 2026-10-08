use super::*;

pub(crate) fn cmd_autolabel(args: &Parsed) -> Result<i32> {
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

pub(crate) fn cmd_goal_model(args: &Parsed) -> Result<i32> {
    // The job's own settings are checked before curation spends any Brama
    // call: a run that would be refused at submission is refused first.
    let ster_options = stado::stated_ster_options(args.text("--ster-options").unwrap_or_default())?;
    let root = placement::resolve_placement()
        .training_root
        .join("goal-model")
        .join("datasets");
    std::fs::create_dir_all(&root)?;
    let stamp = crate::util::now_iso().replace([':', '-'], "");
    let dataset = root.join(format!("reviewed-goals-{stamp}.jsonl"));
    let summary = goal::build_dataset(
        &dataset,
        stated_count(args, "--limit")?,
        args.text("--teacher-model"),
        stated_count(args, "--workers")?,
    )?;
    outln!("{}", dumps(&summary));
    let job = stado::execute_goal_model(
        &dataset,
        args.text("--compute-target").unwrap_or_default(),
        stated_count(args, "--audit-workers")?,
        &ster_options,
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_goal_audit(args: &Parsed) -> Result<i32> {
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
        stated_count(args, "--workers")?,
    )?;
    outln!("{}", dumps(&result));
    Ok(i32::from(
        result.get("passed").and_then(Value::as_bool) != Some(true),
    ))
}

pub(crate) fn cmd_lifecycle_review(args: &Parsed) -> Result<i32> {
    // Labels are ground truth for every model trained from them, so the
    // reviewer must use the strongest operator-approved route.
    let model = args.text("--brama-model").unwrap_or(crate::brama::BEST_MODEL);
    let result = crate::lifecycle::review_dataset(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output").unwrap_or_default()),
        args.text("--split").unwrap_or_default(),
        model,
        args.int("--limit").map(|value| value as usize),
        stated_count(args, "--workers")?,
    )?;
    outln!("{}", dumps(&result));
    Ok(0)
}

pub(crate) fn cmd_lifecycle_model(args: &Parsed) -> Result<i32> {
    let job = stado::execute_lifecycle_model(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.positional(1)),
        args.text("--compute-target").unwrap_or_default(), args.text("--brama-url").unwrap_or_default(),
        args.text("--ster-options").unwrap_or_default(),
        &stado::LifecycleAudit {
            workers: stated_count(args, "--audit-workers")?,
            max_wrong_share: stated_share(args, "--audit-max-wrong-share")?,
        },
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_humanizer_model(args: &Parsed) -> Result<i32> {
    let hf_repo = std::env::var("HUMANIZER_HF_REPO")
        .map_err(|error| Error(format!("cannot read required HUMANIZER_HF_REPO: {error}")))?;
    if hf_repo.trim().is_empty() {
        return Err(Error("HUMANIZER_HF_REPO must explicitly name a private destination".into()));
    }
    // The job's training settings are checked before the corpus is exported:
    // a run that would be refused at submission is refused first.
    let training = stado::HumanizerTraining {
        ster_options: stado::stated_ster_options(args.text("--ster-options").unwrap_or_default())?.to_string(),
    };
    let root = placement::resolve_placement()
        .training_root
        .join("humanizer-model")
        .join("datasets");
    std::fs::create_dir_all(&root)?;
    let stamp = crate::util::now_iso().replace([':', '-'], "");
    let targets = root.join(format!("lukasz-targets-{stamp}.jsonl"));
    let bounds = crate::humanizer::CorpusBounds {
        limit: stated_count(args, "--limit")?,
        minimum: stated_count(args, "--min-targets")?,
        max_per_session: stated_count(args, "--max-per-session")?,
        min_chars: stated_count(args, "--min-target-chars")?,
        max_chars: stated_count(args, "--max-target-chars")?,
        max_lines: stated_count(args, "--max-target-lines")?,
        min_words: stated_count(args, "--min-target-words")?,
        min_meaningful_share: crate::cli::specs_humanizer::stated_share(args, "--min-meaningful-share")?,
    };
    let summary = crate::humanizer::export_targets(&targets, &bounds)?;
    outln!("{}", dumps(&summary));
    let job = stado::execute_humanizer_model(
        &targets,
        args.text("--compute-target").unwrap_or_default(),
        &hf_repo,
        stated_count(args, "--workers")?,
        stated_count(args, "--attempts")?,
        &crate::cli::specs_humanizer::stated_gate(args)?,
        &training,
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

/// A count the caller states on the command line, at least one. There is no
/// default: a missing count is refused by its flag's name.
pub(crate) fn stated_count(args: &Parsed, flag: &str) -> Result<usize> {
    match args.int(flag) {
        Some(value) if value >= 1 => Ok(value as usize),
        Some(value) => Err(Error(format!("{flag} must be at least 1, not {value}"))),
        None => Err(Error(format!(
            "{flag} N is required on the command line; this command has no default for it"
        ))),
    }
}

/// A share the caller states, refused by its flag's name when missing.
pub(crate) fn stated_share(args: &Parsed, flag: &str) -> Result<f64> {
    args.float(flag)
        .ok_or_else(|| Error(format!("{flag} F is required on the command line; this command has no default for it")))
}

pub(crate) fn cmd_lifecycle_audit(args: &Parsed) -> Result<i32> {
    let review_model = match (args.flag("--best"), args.text("--brama-model")) {
        (true, None) => crate::brama::BEST_MODEL,
        (false, Some(model)) => model,
        (true, Some(_)) => return Err(Error("use either --best or --brama-model".to_string())),
        (false, None) => {
            return Err(Error(
                "lifecycle-audit requires --best or --brama-model".to_string(),
            ))
        }
    };
    let result = crate::lifecycle::audit_predictions(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output").unwrap_or_default()),
        review_model,
        std::num::NonZeroUsize::new(stated_count(args, "--workers")?)
            .ok_or_else(|| Error("--workers must be positive".to_string()))?,
        stated_share(args, "--max-wrong-share")?,
    )?;
    outln!("{}", dumps(&result));
    Ok(i32::from(
        result.get("passed").and_then(Value::as_bool) != Some(true),
    ))
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
        return Ok(0);
    }
    print_placement();
    print_info(&entries);
    Ok(0)
}

pub(crate) fn cmd_corpus_adopt(args: &Parsed) -> Result<i32> {
    let report = corpus::adopt(std::path::Path::new(args.positional(0)))?;
    if args.flag("--json") {
        outln!("{}", dumps(&report));
        return Ok(0);
    }
    outln!(
        "selected corpus {} for aspect '{}' ({} imported, {} unchanged, {} conflicting, {} rejected)",
        report.get("corpusId").and_then(Value::as_str).unwrap_or(""),
        report.get("aspect").and_then(Value::as_str).unwrap_or(""),
        report.get("imported").and_then(Value::as_u64).unwrap_or(0),
        report.get("unchanged").and_then(Value::as_u64).unwrap_or(0),
        report.get("conflicting").and_then(Value::as_u64).unwrap_or(0),
        report.get("rejected").and_then(Value::as_u64).unwrap_or(0),
    );
    outln!(
        "retained bundle: {}",
        report
            .get("bundlePath")
            .and_then(Value::as_str)
            .unwrap_or("")
    );
    Ok(0)
}

pub(crate) fn cmd_corpus_status(args: &Parsed) -> Result<i32> {
    let report = corpus::status()?;
    if args.flag("--json") {
        outln!("{}", dumps(&report));
        return Ok(0);
    }
    match report.get("selected").filter(|value| !value.is_null()) {
        Some(selected) => outln!(
            "selected corpus {} for aspect '{}' ({} records)\nretained bundle: {}",
            selected.get("id").and_then(Value::as_str).unwrap_or(""),
            selected.get("aspect").and_then(Value::as_str).unwrap_or(""),
            selected.get("records").and_then(Value::as_u64).unwrap_or(0),
            selected
                .get("bundlePath")
                .and_then(Value::as_str)
                .unwrap_or(""),
        ),
        None => outln!("no adopted corpus is selected"),
    }
    Ok(0)
}

pub(crate) fn cmd_gui(args: &Parsed) -> Result<i32> {
    gui::serve(
        args.text("--bind").unwrap_or("127.0.0.1"),
        args.int("--port").unwrap_or(0),
    )
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
/// 2 when the lake simply has too little labeled data, 1 otherwise.
pub(crate) fn report(command: &str, failure: TrainFailure) -> i32 {
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
