use super::*;

/// A required option naming a file.
fn path(flag: &'static str, help: &str) -> Opt {
    required(flag, "PATH", Kind::Text, help.to_string())
}

pub(crate) fn release_specs() -> Vec<Spec> {
    let models = stado::RELEASE_MODELS
        .iter()
        .map(|model| model.name)
        .collect::<Vec<_>>()
        .join("|");
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
    }, assemble_curriculum_spec(), generate_curriculum_spec(), split_by_day_spec(), decisions_spec(), examples_spec(), manifest_spec()]
}

fn manifest_spec() -> Spec {
    Spec {
        name: "model-manifest",
        help: "write a training job's model-manifest.json from its output directory".to_string(),
        description: Some(
            "Read the output directory a training job staged, record every evidence file's size and \
             SHA-256, the ordered <artifact>.part-* parts and the assembled artifact they rebuild, and \
             qualified from the model's independent gate (final-judge.json passed for the GGUF models; \
             publication.json qualified and audit.json passed for the humanizer), write \
             model-manifest.json there and print it. release-publish reads this manifest. A missing or \
             unreadable input file is refused by name."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            required(
                "--model",
                "MODEL",
                Kind::Text,
                format!("which fine-tune the output holds ({})", stado::ManifestModel::names()),
            ),
            required("--output-dir", "PATH", Kind::Text, "the job's staged output directory".to_string()),
            option(
                "--artifact",
                "PATH",
                Kind::Text,
                "the assembled GGUF whose parts the output holds (goal, lifecycle); refused for the humanizer".to_string(),
            ),
        ],
    }
}

fn decisions_spec() -> Spec {
    Spec {
        name: "lifecycle-decisions",
        help: "write reviewed lifecycle rows as Ster labelled decisions".to_string(),
        description: Some(
            "Check every reviewed row's decision against the decision contract and write the rows as \
             the labelled-decision document ster tune decide trains on: the masked input envelope as \
             the state, the questions training/lifecycle-model/decision/questions.json declares (goal_ref \
             offers the row's own candidates by title), the reviewed action, goal_ref and \
             lifecycle_evidence as the answers."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--rows", "JSONL of reviewed lifecycle rows"),
            path("--output", "JSON the labelled decisions are written to"),
        ],
    }
}

fn examples_spec() -> Spec {
    Spec {
        name: "lifecycle-examples",
        help: "write reviewed lifecycle rows as the Ster examples the served model trains on".to_string(),
        description: Some(
            "Check every reviewed row's decision against the decision contract and write the rows as \
             the supervised example set ster tune sft trains on, each one the chat Oko serves: the \
             lifecycle system prompt as the system turn, the row's user envelope as the prompt and \
             the reviewed decision, its title blanked, as the JSON completion."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--rows", "JSONL of reviewed lifecycle rows"),
            path("--output", "JSON the supervised examples are written to"),
        ],
    }
}

fn split_by_day_spec() -> Spec {
    Spec {
        name: "lifecycle-split-by-day",
        help: "re-split reviewed lifecycle rows, holding one day out for evaluation".to_string(),
        description: Some(
            "Merge the reviewed rows of an earlier training and evaluation split by id (one id with \
             two different rows is refused), put every row whose split_day is --eval-day into \
             evaluation and every other row into training, ordered by id, require every action the \
             output schema declares in both splits, write both and print the counts."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--train", "JSONL of the earlier reviewed training rows"),
            path("--eval", "JSONL of the earlier reviewed evaluation rows"),
            required("--eval-day", "DAY", Kind::Text, "the split_day held out for evaluation".to_string()),
            path("--output-train", "JSONL the new training split is written to"),
            path("--output-eval", "JSONL the new evaluation split is written to"),
        ],
    }
}

fn generate_curriculum_spec() -> Spec {
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

fn assemble_curriculum_spec() -> Spec {
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
