//! The humanizer's Ster commands: `humanizer-examples` (a prepared split as
//! the Ster example set it trains on) and `humanizer-evaluate-gguf` (the base
//! and the merged student asked for every test row as served models).

use super::*;

/// The options `humanizer-model` trains and measures its adapter with:
/// the `ster tune sft` settings and how the evaluation serves both models.
pub(crate) fn training_options() -> Vec<Opt> {
    let count = |flag: &'static str, help: &str| required(flag, "N", Kind::Int, help.to_string());
    vec![
        required(
            "--ster-options",
            "OPTIONS",
            Kind::Text,
            "the ster tune sft settings the job trains with, e.g. '--rank R --alpha A --epochs E \
             --learning-rate L --accumulation N --max-sequence T --batch-size B --seed S'; Ster \
             refuses a run that leaves one of its required settings out"
                .to_string(),
        ),
        count("--eval-parallel", "server slots of the evaluation"),
        count("--eval-slot-context", "context tokens each evaluation slot holds"),
        required("--eval-gpu-layers", "N", Kind::Text, "layers llama-server offloads to the GPU in the evaluation".to_string()),
        count("--eval-max-tokens", "the most tokens one evaluated answer may hold"),
        count("--chrf-order", "the character n-gram order the evaluation's chrF counts"),
    ]
}

pub(crate) fn humanizer_served_specs() -> Vec<Spec> {
    let path = |flag: &'static str, help: &str| required(flag, "PATH", Kind::Text, help.to_string());
    let examples = Spec {
        name: "humanizer-examples",
        help: "write a prepared humanizer split as the Ster examples the adapter trains on".to_string(),
        description: Some(
            "Write every row of a split humanizer-prepare wrote as one ster tune sft example: the \
             row's system prompt as its system turn, the generic source as its prompt and the \
             user's own target as its completion. A row without exactly one non-empty system, user \
             and assistant turn is refused by id; a file without a row is refused."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--rows", "a prepared split, e.g. train.jsonl"),
            path("--output", "JSON the supervised examples are written to"),
        ],
    };
    let count = |flag: &'static str, help: &str| required(flag, "N", Kind::Int, help.to_string());
    let evaluate = Spec {
        name: "humanizer-evaluate-gguf",
        help: "ask the base and the merged student humanizer for every test row as served models".to_string(),
        description: Some(
            "Start llama-server on the base GGUF, ask it for every test row (the row's system prompt \
             and source, at most --max-tokens tokens), stop it, then do the same with the student \
             GGUF. Score each answer by chrF of order --chrf-order against the target and against the \
             source and by its length against the target's, and write predictions.jsonl for \
             humanizer-audit and metrics.json with the means and the target chrF gain. A model \
             that does not exist is refused before a server starts; a failed request fails the run \
             naming its row."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--base", "the base model as a full-precision GGUF"),
            path("--student", "the merged student as a full-precision GGUF"),
            path("--dataset", "the prepared test split"),
            path("--predictions", "JSONL every prediction is written to"),
            path("--metrics", "JSON the scores are written to"),
            path("--server", "the llama-server executable"),
            path("--server-log", "where the server's log is written; the student's goes beside it"),
            required("--base-model", "MODEL", Kind::Text, "the base model id metrics.json records".to_string()),
            required("--base-revision", "REVISION", Kind::Text, "the base model revision metrics.json records".to_string()),
            count("--parallel", "server slots"),
            count("--slot-context", "context tokens each slot holds"),
            required("--gpu-layers", "N", Kind::Text, "layers llama-server offloads to the GPU, as llama-server takes it".to_string()),
            count("--max-tokens", "the most tokens one answer may hold"),
            count("--chrf-order", "the character n-gram order chrF counts"),
        ],
    };
    vec![examples, evaluate]
}

pub(crate) fn cmd_humanizer_examples(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    printed(&crate::humanizer::export_examples(&path("--rows"), &path("--output"))?)
}

pub(crate) fn cmd_humanizer_evaluate_gguf(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    let serving = |model: &str, log: std::path::PathBuf| -> Result<crate::serving::Serving> {
        Ok(crate::serving::Serving {
            server: path("--server"),
            model: path(model),
            server_log: log,
            parallel: stated_count(args, "--parallel")?,
            slot_context: stated_count(args, "--slot-context")?,
            gpu_layers: args.text("--gpu-layers").unwrap_or_default().to_string(),
        })
    };
    let log = path("--server-log");
    let run = crate::humanizer::HumanizerEvaluation {
        base: serving("--base", log.clone())?,
        student: serving("--student", log.with_extension("student.log"))?,
        dataset: path("--dataset"),
        predictions: path("--predictions"),
        metrics: path("--metrics"),
        max_tokens: stated_count(args, "--max-tokens")?,
        chrf_order: stated_count(args, "--chrf-order")?,
        base_model: args.text("--base-model").unwrap_or_default().to_string(),
        base_revision: args.text("--base-revision").unwrap_or_default().to_string(),
    };
    printed(&crate::humanizer::evaluate_gguf(&run)?)
}
