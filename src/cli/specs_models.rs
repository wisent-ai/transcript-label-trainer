use super::*;

#[allow(unused_variables)]
pub(crate) fn model_specs() -> Vec<Spec> {
    let fraction = float_repr(jobs::DEFAULT_EVAL_FRACTION);
    let seed = jobs::DEFAULT_EVAL_SEED;
    let teacher = brama::DEFAULT_MODEL;

    let aspect_discover = Spec {
        name: "aspect-discover",
        help: "propose new aspect dimensions from recent sessions via a Brama \
               teacher (writes nothing)"
            .to_string(),
        description: Some(
            "Read recent privacy-masked sessions, have a Brama teacher propose \
             aspect dimensions grounded in what the user asked for and how the \
             agent answered, merge the proposals across batches, and print them \
             with the exact autolabel and train commands that would turn each \
             one into a model. Writes nothing: the lake's labeler owns the \
             aspect vocabulary."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            option(
                "--limit",
                "LIMIT",
                Kind::Int,
                "newest sessions sampled (default: 60)".to_string(),
            ),
            option(
                "--max-aspects",
                "N",
                Kind::Int,
                "keep at most this many merged proposals (default: 8)".to_string(),
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
                "have Brama's best route audit every kept proposal; reject \
                 nonsensical ones and exit nonzero when the audit finds an issue"
                    .to_string(),
            ),
            option(
                "--runtime",
                "RUNTIME",
                Kind::Text,
                "only sessions of this runtime".to_string(),
            ),
        ],
    };

    let goal_model = Spec {
        name: "goal-model",
        help: "build and train the reviewed Jeden goal model on Stado".to_string(),
        description: Some(
            "Read only privacy-masked Transcript Lake events, use a Brama teacher \
             to label task goals, require an independent Brama best review, then \
             train on the named exclusive Stado GPU target. The held-out gold \
             predictions must all pass a second best audit before GGUF artifacts \
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

    let lifecycle_review = Spec {
        name: "lifecycle-review",
        help: "replace Oko lifecycle silver answers with Brama-reviewed decisions".to_string(),
        description: Some(
            "Read masked Oko training envelopes, classify each one through a named Brama route, \
             enforce the oko-goal-lifecycle-v1 contract, and write ordered JSONL with reviewer \
             provenance."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "input",
            help: "JSONL of Oko lifecycle training envelopes".to_string(),
        }],
        opts: vec![
            required(
                "--output",
                "PATH",
                Kind::Text,
                "write reviewed lifecycle JSONL here".to_string(),
            ),
            required(
                "--split",
                "train|eval",
                Kind::Text,
                "record the immutable dataset split".to_string(),
            ),
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!(
                    "Brama route used to review every decision (default: {})",
                    crate::brama::BEST_MODEL
                ),
            ),
            option(
                "--limit",
                "LIMIT",
                Kind::Int,
                "cap reviewed rows".to_string(),
            ),
        ],
    };

    let lifecycle_model = Spec {
        name: "lifecycle-model",
        help: "train and qualify Oko's reviewed lifecycle model on Stado".to_string(),
        description: Some(
            "Upload immutable reviewed train and held-out datasets, fine-tune on the named \
             exclusive Stado GPU target, audit every held-out decision through Brama -best, \
             and publish the complete candidate only when the lifecycle quality gate passes."
                .to_string(),
        ),
        positionals: vec![
            Positional {
                name: "train",
                help: "reviewed lifecycle training JSONL".to_string(),
            },
            Positional {
                name: "eval",
                help: "reviewed held-out lifecycle JSONL".to_string(),
            },
        ],
        opts: vec![
            required(
                "--compute-target",
                "COMPUTE_TARGET",
                Kind::Text,
                "canonical Stado GPU target that trains, audits, and exports the model".to_string(),
            ),
            required(
                "--brama-url",
                "URL",
                Kind::Text,
                "Brama endpoint reachable from the compute target".to_string(),
            ),
        ],
    };

    let humanizer_model = Spec {
        name: "humanizer-model",
        help: "train and qualify Echo's personal-voice humanizer on Stado".to_string(),
        description: Some(
            "Export only privacy-masked likely-authored user turns from Transcript Lake, \
             derive inverse style-transfer inputs through Brama, freeze session-separated \
             train, validation, and test splits, train a LoRA adapter on the pinned Cydonia-24B \
             deployment base on the named exclusive Stado GPU target, compare it with the base \
             model, require an independent Brama audit, and publish only a qualified private adapter revision."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            required(
                "--compute-target",
                "COMPUTE_TARGET",
                Kind::Text,
                "canonical Stado GPU target that curates, trains, audits, and publishes the model"
                    .to_string(),
            ),
            option(
                "--limit",
                "LIMIT",
                Kind::Int,
                "maximum clean authored targets (default: 1500; minimum: 1000)".to_string(),
            ),
        ],
    };

    let lifecycle_audit = Spec {
        name: "lifecycle-audit",
        help: "apply the final Brama semantic audit to lifecycle predictions".to_string(),
        description: Some(
            "Judge every held-out student decision independently, reject inferred completion, \
             retain the full verdict record, and fail the lifecycle quality gate when more than \
             two percent are semantically wrong."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "predictions",
            help: "JSONL containing masked input, reference, and student decision".to_string(),
        }],
        opts: vec![
            required(
                "--output",
                "PATH",
                Kind::Text,
                "write the complete lifecycle audit record here".to_string(),
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
                "use an explicit Brama-routed independent judge".to_string(),
            ),
        ],
    };
    vec![
        aspect_discover,
        goal_model,
        goal_audit,
        lifecycle_review,
        lifecycle_model,
        humanizer_model,
        lifecycle_audit,
    ]
}
