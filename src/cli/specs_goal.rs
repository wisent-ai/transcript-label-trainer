//! The goal model's served-chat commands: `goal-examples` (the Ster example
//! set it trains on) and `goal-evaluate-gguf` (its held-out gold rows asked of
//! the quantized model as Jeden serves it).

use super::*;

pub(crate) fn goal_served_specs() -> Vec<Spec> {
    let path = |flag: &'static str, help: &str| required(flag, "PATH", Kind::Text, help.to_string());
    let examples = Spec {
        name: "goal-examples",
        help: "write reviewed goal rows as the Ster examples the served goal model trains on".to_string(),
        description: Some(
            "Write every reviewed row not marked gold as one ster tune sft example, each the chat \
             Jeden serves: the goal system prompt as the system turn, the message as <user>…</user> \
             and the reviewed answer, <goal>…</goal> or <goal/>, as the completion. Gold rows are \
             held out and counted. A file without a row, or whose every row is gold, is refused."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--rows", "JSONL of reviewed goal rows, gold rows marked"),
            path("--output", "JSON the supervised examples are written to"),
        ],
    };
    let evaluate = Spec {
        name: "goal-evaluate-gguf",
        help: "ask the quantized goal model for every held-out gold row the way Jeden serves it".to_string(),
        description: Some(
            "Start llama-server on the quantized model on a loopback port the system assigns, wait \
             until its health answer is ready (reading its log between probes, failing when it exits \
             first), ask it for every gold row's answer with decoding constrained to <goal/> or one \
             <goal>…</goal> line, and write the predictions goal-audit judges and the exact-match \
             share. Each request waits for its answer; a failed request fails the run naming its \
             session."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            path("--model", "the quantized GGUF model"),
            path("--dataset", "JSONL of reviewed goal rows, gold rows marked"),
            path("--predictions", "JSONL every prediction is written to"),
            path("--metrics", "JSON the exact-match share is written to"),
            path("--server", "the llama-server executable"),
            path("--server-log", "where the server's log is written"),
        ],
    };
    vec![examples, evaluate]
}

pub(crate) fn cmd_goal_examples(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    printed(&crate::goal::export_examples(&path("--rows"), &path("--output"))?)
}

pub(crate) fn cmd_goal_evaluate_gguf(args: &Parsed) -> Result<i32> {
    let path = |flag: &str| std::path::PathBuf::from(args.text(flag).unwrap_or_default());
    let run = crate::goal::GoalEvaluation {
        serving: crate::serving::Serving {
            server: path("--server"),
            model: path("--model"),
            server_log: path("--server-log"),
        },
        dataset: path("--dataset"),
        predictions: path("--predictions"),
        metrics: path("--metrics"),
    };
    printed(&crate::goal::evaluate_gguf(&run)?)
}
