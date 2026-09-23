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
    let job =
        stado::execute_goal_model(&dataset, args.text("--compute-target").unwrap_or_default())?;
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
    )?;
    outln!("{}", dumps(&result));
    Ok(i32::from(
        result.get("passed").and_then(Value::as_bool) != Some(true),
    ))
}

pub(crate) fn cmd_lifecycle_review(args: &Parsed) -> Result<i32> {
    // Labels are ground truth for every model trained from them, so the
    // reviewer must be the strongest operator-approved route. This defaulted to
    // `wisent-backend/chat/primary` until 2026-08-18, which is an unrelated
    // product model: it labelled 1,617 curriculum rows, put a subagent
    // completion report in the `ignore` class where the human-reviewed held-out
    // split says `continueCurrent`, and its rows went into the candidate that
    // then failed the quality gate on false completions.
    let model = args.text("--brama-model").unwrap_or(crate::brama::BEST_MODEL);
    let result = crate::lifecycle::review_dataset(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.text("--output").unwrap_or_default()),
        args.text("--split").unwrap_or_default(),
        model,
        args.int("--limit").map(|value| value as usize),
    )?;
    outln!("{}", dumps(&result));
    Ok(0)
}

pub(crate) fn cmd_lifecycle_model(args: &Parsed) -> Result<i32> {
    let job = stado::execute_lifecycle_model(
        std::path::Path::new(args.positional(0)),
        std::path::Path::new(args.positional(1)),
        args.text("--compute-target").unwrap_or_default(),
        args.text("--brama-url").unwrap_or_default(),
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
}

pub(crate) fn cmd_humanizer_model(args: &Parsed) -> Result<i32> {
    let root = placement::resolve_placement()
        .training_root
        .join("humanizer-model")
        .join("datasets");
    std::fs::create_dir_all(&root)?;
    let stamp = crate::util::now_iso().replace([':', '-'], "");
    let targets = root.join(format!("lukasz-targets-{stamp}.jsonl"));
    let summary = crate::humanizer::export_targets(&targets, count(args.int("--limit"), 1_500))?;
    outln!("{}", dumps(&summary));
    let job = stado::execute_humanizer_model(
        &targets,
        args.text("--compute-target").unwrap_or_default(),
    )?;
    outln!("Stado job: {}", job.job_id);
    outln!("model artifact: {}", job.output_uri);
    Ok(job.status)
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
