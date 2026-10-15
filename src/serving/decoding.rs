//! How a served model picks each token. An evaluation decodes the way the
//! product that serves the model to its users decodes, so what qualifies is
//! what those users get; the caller names that product's way, never this
//! module.

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Decoding {
    /// The sampling Qwen3's model card states for non-thinking mode, the mode
    /// Jeden Desktop runs the goal model in (llama-cli --reasoning off) and
    /// Oko asks the lifecycle model in (enable_thinking false). Both models
    /// are fine-tunes of Qwen3-4B, whose card warns against greedy decoding.
    Qwen3NonThinking,
    /// The server's own sampling: no sampling field is sent. Echo asks the
    /// humanizer without one.
    ServerDefault,
}

impl Decoding {
    /// The request fields that select this decoding on llama-server's chat
    /// route; none when the server's own sampling is meant.
    pub(crate) fn fields(self) -> Vec<(&'static str, Value)> {
        match self {
            Decoding::Qwen3NonThinking => vec![
                ("temperature", json!(0.7)), // https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md "For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0."
                ("top_p", json!(0.8)), // https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md "For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0."
                ("top_k", json!(20)), // https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md "For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0."
                ("min_p", json!(0.0)), // https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md "For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0."
            ],
            Decoding::ServerDefault => Vec::new(),
        }
    }
}
