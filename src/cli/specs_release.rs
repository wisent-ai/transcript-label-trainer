use super::*;

pub(crate) fn release_specs() -> Vec<Spec> {
    let models = stado::RELEASE_MODELS
        .iter()
        .map(|model| model.name)
        .collect::<Vec<_>>()
        .join("|");
    vec![Spec {
        name: "release-publish",
        help: "publish a qualified fine-tune from its Stado job output".to_string(),
        description: Some(
            "Fetch the job output's model-manifest.json, refuse unless it is qualified by \
             final-judge.json and the judge passed, verify every evidence file and model part \
             against the manifest's SHA-256, check that the ordered parts rebuild the qualified \
             artifact, and publish them under the artifact's digest. Objects already present at \
             that coordinate are left untouched, so a rerun resumes."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "source",
            help: "Stado job output URI holding model-manifest.json".to_string(),
        }],
        opts: vec![required(
            "--model",
            "MODEL",
            Kind::Text,
            format!("which fine-tune the output holds ({models})"),
        )],
    }]
}

pub(crate) fn cmd_release_publish(args: &Parsed) -> Result<i32> {
    let name = args.text("--model").unwrap_or_default();
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
    let published = stado::publish_release(model, args.positional(0))?;
    outln!("{}", dumps(&published));
    Ok(0)
}
