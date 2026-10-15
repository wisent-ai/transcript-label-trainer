use super::*;

/// The file a required path option names.
fn path_of(args: &Parsed, flag: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(args.required(flag))
}

/// A count the caller states on the command line, at least one. There is no
/// default: a missing count is refused by its flag's name.
pub(crate) fn stated_count(args: &Parsed, flag: &str) -> Result<usize> {
    let Some(value) = args.int(flag) else {
        return Err(Error(format!(
            "{flag} N is required on the command line; this command has no default for it"
        )));
    };
    let count = usize::try_from(value).map_err(|error| {
        Error(format!(
            "{flag} must be a count of at least one, not {value}: {error}"
        ))
    })?;
    match std::num::NonZeroUsize::new(count) {
        Some(count) => Ok(count.get()),
        None => Err(Error(format!(
            "{flag} must be a count of at least one, not {value}"
        ))),
    }
}

/// Print `report` as the command's answer and end it as having done what it
/// promised.
pub(crate) fn printed(report: &Value) -> Result<i32> {
    outln!("{}", dumps(report));
    Ok(EXIT_SUCCESS)
}

pub(crate) fn cmd_model_manifest(args: &Parsed) -> Result<i32> {
    let model = stado::ManifestModel::named(args.required("--model"))?;
    let output = path_of(args, "--output-dir");
    let artifact = args.text("--artifact").map(std::path::PathBuf::from);
    printed(&stado::write_manifest(model, &output, artifact.as_deref())?)
}

pub(crate) fn cmd_lifecycle_decisions(args: &Parsed) -> Result<i32> {
    let (rows, output) = (path_of(args, "--rows"), path_of(args, "--output"));
    printed(&crate::lifecycle::export_decisions(
        &crate::lifecycle::DecisionExport {
            rows: &rows,
            output: &output,
        },
    )?)
}

pub(crate) fn cmd_lifecycle_examples(args: &Parsed) -> Result<i32> {
    let (rows, output) = (path_of(args, "--rows"), path_of(args, "--output"));
    printed(&crate::lifecycle::export_examples(
        &crate::lifecycle::DecisionExport {
            rows: &rows,
            output: &output,
        },
    )?)
}

pub(crate) fn cmd_lifecycle_split_by_day(args: &Parsed) -> Result<i32> {
    let (train, eval) = (path_of(args, "--train"), path_of(args, "--eval"));
    let (output_train, output_eval) = (
        path_of(args, "--output-train"),
        path_of(args, "--output-eval"),
    );
    printed(&crate::lifecycle::split_by_day(
        &crate::lifecycle::DaySplit {
            train: &train,
            eval: &eval,
            eval_day: args.required("--eval-day"),
            output_train: &output_train,
            output_eval: &output_eval,
        },
    )?)
}

pub(crate) fn cmd_lifecycle_generate_curriculum(args: &Parsed) -> Result<i32> {
    let (source, output) = (path_of(args, "--source"), path_of(args, "--output"));
    let seed = args.int("--seed").ok_or_else(|| {
        Error("--seed N is required; this command has no default for it".to_string())
    })?;
    let seed = u64::try_from(seed).map_err(|error| {
        Error(format!(
            "--seed must be a whole number of zero or more, not {seed}: {error}"
        ))
    })?;
    printed(&crate::lifecycle::generate_curriculum(
        &crate::lifecycle::CurriculumGeneration {
            source: &source,
            output: &output,
            per_family: stated_count(args, "--per-family")?,
            seed,
        },
    )?)
}

pub(crate) fn cmd_lifecycle_assemble_curriculum(args: &Parsed) -> Result<i32> {
    let (base_train, base_eval) = (path_of(args, "--base-train"), path_of(args, "--base-eval"));
    let (reviewed_train, reviewed_eval) = (
        path_of(args, "--curriculum-train"),
        path_of(args, "--curriculum-eval"),
    );
    let (output_train, output_eval) = (
        path_of(args, "--output-train"),
        path_of(args, "--output-eval"),
    );
    printed(&crate::lifecycle::assemble_curriculum(
        &crate::lifecycle::CurriculumAssembly {
            base_train: &base_train,
            base_eval: &base_eval,
            reviewed_train: &reviewed_train,
            reviewed_eval: &reviewed_eval,
            output_train: &output_train,
            output_eval: &output_eval,
            minimum_eval_per_action: stated_count(args, "--minimum-eval-per-action")?,
        },
    )?)
}

pub(crate) fn cmd_release_publish(args: &Parsed) -> Result<i32> {
    let name = args.required("--model");
    let Some(model) = stado::RELEASE_MODELS
        .iter()
        .find(|model| model.name == name)
    else {
        return Err(Error(format!(
            "--model must be one of {}, not {name:?}",
            stado::RELEASE_MODELS
                .iter()
                .map(|model| model.name)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    printed(&stado::publish_release(
        model,
        args.positional("source"),
        args.text("--repo"),
    )?)
}

pub(crate) fn cmd_lifecycle_evaluate_gguf(args: &Parsed) -> Result<i32> {
    printed(&crate::lifecycle::evaluate_gguf(
        &crate::lifecycle::GgufEvaluation {
            model: path_of(args, "--model"),
            dataset: path_of(args, "--dataset"),
            predictions: path_of(args, "--predictions"),
            metrics: path_of(args, "--metrics"),
            server: path_of(args, "--server"),
            server_log: path_of(args, "--server-log"),
            output_schema: path_of(args, "--output-schema"),
        },
    )?)
}
