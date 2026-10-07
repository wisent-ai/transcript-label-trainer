use super::*;

pub(crate) fn release_specs() -> Vec<Spec> {
    let models = stado::RELEASE_MODELS
        .iter()
        .map(|model| model.name)
        .collect::<Vec<_>>()
        .join("|");
    let path = |flag: &'static str, help: &str| required(flag, "PATH", Kind::Text, help.to_string());
    let evaluate_gguf = Spec {
        name: "lifecycle-evaluate-gguf",
        help: "measure the quantized lifecycle model the way production serves it".to_string(),
        description: Some(
            "Start llama-server on the quantized model on a loopback port the system assigns, wait \
             until its health answer is ready (reading its log between probes, failing when it exits \
             first), ask it for every evaluation row's decision with decoding constrained to the \
             output schema narrowed to the row's candidate references, and write every prediction \
             and the measured rates. Each request waits for its answer; a failed request fails the \
             run naming its row. A rate with nothing to measure is null."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--model", "the quantized GGUF model"),
            path("--dataset", "JSONL of reviewed evaluation rows"),
            path("--predictions", "JSONL every prediction is written to"),
            path("--metrics", "JSON the measured rates are written to"),
            path("--server", "the llama-server executable"),
            path("--server-log", "where the server's log is written"),
            path("--output-schema", "the checked-in decision schema"),
            required("--parallel", "N", Kind::Int, "server slots and concurrent requests".to_string()),
            required("--slot-context", "N", Kind::Int, "context tokens each slot holds".to_string()),
            required("--gpu-layers", "N", Kind::Text, "layers llama-server offloads to the GPU, as llama-server takes it".to_string()),
        ],
    };
    vec![evaluate_gguf, Spec {
        name: "release-publish",
        help: "publish a qualified fine-tune from its Stado job output".to_string(),
        description: Some(
            "Fetch the job output's model-manifest.json, refuse unless it is qualified by \
             final-judge.json and the judge passed, verify every evidence file and model part \
             against the manifest's SHA-256, check that the ordered parts rebuild the qualified \
             artifact, and publish it where the model runs. A model that ships inside a desktop \
             application goes to the release channel under the artifact's digest; objects already \
             present there are left untouched, so a rerun resumes. A model served from a GPU host \
             goes to the private Hugging Face repository --repo names, as one new branch whose \
             immutable revision is read back and checked file by file; the release store receives \
             nothing."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "source",
            help: "Stado job output URI holding model-manifest.json".to_string(),
        }],
        opts: vec![
            required(
                "--model",
                "MODEL",
                Kind::Text,
                format!("which fine-tune the output holds ({models})"),
            ),
            option(
                "--repo",
                "OWNER/REPOSITORY",
                Kind::Text,
                "private Hugging Face repository of a model served from a GPU host (lifecycle); refused for a model that ships through the release channel (goal)".to_string(),
            ),
        ],
    }, assemble_curriculum_spec(), generate_curriculum_spec()]
}

fn generate_curriculum_spec() -> Spec {
    let path = |flag: &'static str, help: &str| required(flag, "PATH", Kind::Text, help.to_string());
    Spec {
        name: "lifecycle-generate-curriculum",
        help: "write deterministic hard-case lifecycle turns for Brama review".to_string(),
        description: Some(
            "For every family in training/lifecycle-model/curriculum/templates.json, write the stated \
             number of rows built from real reviewed envelopes with an active candidate: one of the \
             family's sentences filled with that candidate's title (a switch also names a candidate \
             the envelope does not hold), the family and its intended action recorded for the \
             assembler, and turn indexes after every source turn. The envelope keeps only what Oko \
             sends. The same seed gives the same rows."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--source", "JSONL of reviewed lifecycle rows the turns are built from"),
            path("--output", "JSONL the generated rows are written to"),
            required("--per-family", "N", Kind::Int, "rows written for each family".to_string()),
            required("--seed", "N", Kind::Int, "the seed every choice is drawn from".to_string()),
        ],
    }
}

pub(crate) fn cmd_lifecycle_generate_curriculum(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    let (source, output) = (path("--source"), path("--output"));
    let seed = args
        .int("--seed")
        .ok_or_else(|| Error("--seed N is required; this command has no default for it".to_string()))?;
    let run = crate::lifecycle::CurriculumGeneration {
        source: &source,
        output: &output,
        per_family: stated_count(args, "--per-family")?,
        seed: seed as u64,
    };
    let report = crate::lifecycle::generate_curriculum(&run)?;
    outln!("{}", dumps(&report));
    Ok(0)
}

fn assemble_curriculum_spec() -> Spec {
    let path = |flag: &'static str, help: &str| required(flag, "PATH", Kind::Text, help.to_string());
    Spec {
        name: "lifecycle-assemble-curriculum",
        help: "assemble the lifecycle splits from reviewed rows and the agreed curriculum".to_string(),
        description: Some(
            "Read the reviewed training and evaluation rows and the reviewed generated curriculum, \
             check every row's one decision against the decision contract and clear its title, keep \
             a curriculum row only when its review chose the action the curriculum intended (each \
             disagreement counted as intended->reviewed), require the stated number of accepted \
             evaluation rows for every action the output schema declares, refuse duplicate ids in a \
             split and any id shared between the splits, write both splits whole and print the \
             counts."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--base-train", "JSONL of reviewed real training rows"),
            path("--base-eval", "JSONL of reviewed real evaluation rows"),
            path("--curriculum-train", "JSONL of the reviewed training curriculum"),
            path("--curriculum-eval", "JSONL of the reviewed evaluation curriculum"),
            path("--output-train", "JSONL the assembled training split is written to"),
            path("--output-eval", "JSONL the assembled evaluation split is written to"),
            required(
                "--minimum-eval-per-action",
                "N",
                Kind::Int,
                "accepted evaluation curriculum rows every action needs".to_string(),
            ),
        ],
    }
}

pub(crate) fn cmd_lifecycle_assemble_curriculum(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    let (base_train, base_eval) = (path("--base-train"), path("--base-eval"));
    let (reviewed_train, reviewed_eval) = (path("--curriculum-train"), path("--curriculum-eval"));
    let (output_train, output_eval) = (path("--output-train"), path("--output-eval"));
    let run = crate::lifecycle::CurriculumAssembly {
        base_train: &base_train,
        base_eval: &base_eval,
        reviewed_train: &reviewed_train,
        reviewed_eval: &reviewed_eval,
        output_train: &output_train,
        output_eval: &output_eval,
        minimum_eval_per_action: stated_count(args, "--minimum-eval-per-action")?,
    };
    let report = crate::lifecycle::assemble_curriculum(&run)?;
    outln!("{}", dumps(&report));
    Ok(0)
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
    let published = stado::publish_release(model, args.positional(0), args.text("--repo"))?;
    outln!("{}", dumps(&published));
    Ok(0)
}

pub(crate) fn cmd_lifecycle_evaluate_gguf(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    let run = crate::lifecycle::GgufEvaluation {
        model: path("--model"),
        dataset: path("--dataset"),
        predictions: path("--predictions"),
        metrics: path("--metrics"),
        server: path("--server"),
        server_log: path("--server-log"),
        output_schema: path("--output-schema"),
        parallel: stated_count(args, "--parallel")?,
        slot_context: stated_count(args, "--slot-context")?,
        gpu_layers: args.text("--gpu-layers").unwrap_or_default().to_string(),
    };
    let report = crate::lifecycle::evaluate_gguf(&run)?;
    outln!("{}", dumps(&report));
    Ok(0)
}

/// The serving settings the lifecycle model's quantized evaluation is stated to run with.
pub(crate) fn lifecycle_serving(args: &Parsed) -> Result<stado::LifecycleServing> {
    let gpu_layers = args.text("--eval-gpu-layers").unwrap_or_default().trim().to_string();
    if gpu_layers.is_empty() {
        return Err(Error("--eval-gpu-layers N is required on the command line; this command has no default for it".to_string()));
    }
    Ok(stado::LifecycleServing {
        parallel: stated_count(args, "--eval-parallel")?,
        slot_context: stated_count(args, "--eval-slot-context")?,
        gpu_layers,
    })
}
