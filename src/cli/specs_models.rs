use super::*;

#[allow(unused_variables)]
pub(crate) fn model_specs() -> Vec<Spec> {
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
                "newest sessions sampled (default: every session)".to_string(),
            ),
            option(
                "--max-aspects",
                "N",
                Kind::Int,
                "keep at most this many merged proposals (default: every proposal)".to_string(),
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
            required(
                "--limit",
                "LIMIT",
                Kind::Int,
                "most teacher-labeled candidates".to_string(),
            ),
            option(
                "--teacher-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama-routed goal teacher (default: {teacher})"),
            ),
            required(
                "--workers",
                "N",
                Kind::Int,
                "parallel Brama teacher calls while curating, at least 1".to_string(),
            ),
            required(
                "--audit-workers",
                "N",
                Kind::Int,
                "parallel Brama calls in the job's final goal-audit, at least 1".to_string(),
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
            required(
                "--workers",
                "N",
                Kind::Int,
                "parallel Brama audit calls, at least 1".to_string(),
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
            required(
                "--workers",
                "N",
                Kind::Int,
                "parallel Brama reviews, at least 1".to_string(),
            ),
        ],
    };

    let lifecycle_model = Spec {
        name: "lifecycle-model",
        help: "train and qualify Oko's reviewed lifecycle model on Stado".to_string(),
        description: Some(
            "Upload immutable reviewed train and held-out datasets, train the served model with \
             ster tune sft on the named exclusive Stado GPU target (the reviewed rows as \
             lifecycle-examples, every training setting from --ster-options), audit every held-out \
             decision through Brama -best, and publish the complete candidate only when the lifecycle \
             quality gate passes: at most --audit-max-wrong-share of the decisions semantically \
             wrong, judged by --audit-workers concurrent Brama calls."
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
            required(
                "--ster-options",
                "OPTIONS",
                Kind::Text,
                "the ster tune sft settings the job trains with, e.g. '--rank R --alpha A --epochs E \
                 --learning-rate L --accumulation N --max-sequence T --batch-size B --seed S'; Ster \
                 refuses a run that leaves one of its required settings out"
                    .to_string(),
            ),
            required("--eval-parallel", "N", Kind::Int, "server slots and concurrent requests of the quantized evaluation".to_string()),
            required("--eval-slot-context", "N", Kind::Int, "context tokens each evaluation slot holds".to_string()),
            required("--eval-gpu-layers", "N", Kind::Text, "layers llama-server offloads to the GPU in the evaluation".to_string()),
            required("--audit-workers", "N", Kind::Int, "concurrent Brama calls of the final audit; the route's own concurrency allowance".to_string()),
            required("--audit-max-wrong-share", "F", Kind::Float, "largest share of held-out decisions the audit may call wrong, between none and all".to_string()),
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
             model, require an independent Brama audit, and publish only a qualified private adapter revision. \
             HUMANIZER_HF_REPO must name the destination repository; there is no account default."
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
            required(
                "--limit",
                "LIMIT",
                Kind::Int,
                "most clean authored targets the corpus takes".to_string(),
            ),
            required(
                "--min-targets",
                "N",
                Kind::Int,
                "fewest clean authored targets the corpus must reach; fewer refuses the export".to_string(),
            ),
            required(
                "--max-per-session",
                "N",
                Kind::Int,
                "most targets one session may contribute".to_string(),
            ),
            required(
                "--min-target-chars",
                "N",
                Kind::Int,
                "fewest characters an authored target may have".to_string(),
            ),
            required(
                "--max-target-chars",
                "N",
                Kind::Int,
                "most characters an authored target may have".to_string(),
            ),
            required(
                "--max-target-lines",
                "N",
                Kind::Int,
                "most lines an authored target may have".to_string(),
            ),
            required(
                "--min-target-words",
                "N",
                Kind::Int,
                "fewest words holding a letter an authored target needs".to_string(),
            ),
            required(
                "--min-meaningful-share",
                "F",
                Kind::Float,
                "smallest share of an authored target's characters that are letters, digits or spaces"
                    .to_string(),
            ),
            required(
                "--workers",
                "N",
                Kind::Int,
                "parallel Brama calls in the job's preparation and audit".to_string(),
            ),
            required(
                "--attempts",
                "N",
                Kind::Int,
                "times the job asks one Brama question before its row is given up".to_string(),
            ),
        ]
        .into_iter()
        .chain(super::specs_humanizer::gate_options())
        .chain(super::specs_humanizer::minimum_options())
        .collect(),
    };

    let lifecycle_audit = Spec {
        name: "lifecycle-audit",
        help: "apply the final Brama semantic audit to lifecycle predictions".to_string(),
        description: Some(
            "Judge every held-out student decision independently, reject inferred completion, \
             retain the full verdict record, and fail the lifecycle quality gate when more than \
             --max-wrong-share of them are semantically wrong."
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
            required("--workers", "N", Kind::Int, "concurrent Brama calls; the route's own concurrency allowance".to_string()),
            required("--max-wrong-share", "F", Kind::Float, "largest share of decisions the audit may call wrong, between none and all".to_string()),
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
