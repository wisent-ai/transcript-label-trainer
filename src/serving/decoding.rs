//! How a served model picks each token. An evaluation decodes the way the
//! product that serves the model to its users decodes, so what qualifies is
//! what those users get; the caller names that product's way, never this
//! module.

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Decoding {
    /// The most likely token at every step: Jeden Desktop runs the goal model
    /// with llama-cli --temp 0, and Oko asks the lifecycle model at
    /// temperature zero.
    Greedy,
    /// The server's own sampling: no sampling field is sent. Echo asks the
    /// humanizer without one.
    ServerDefault,
}

impl Decoding {
    /// The request field that selects this decoding on llama-server's chat
    /// route, when one has to be sent.
    pub(crate) fn field(self) -> Option<(&'static str, Value)> {
        match self {
            Decoding::Greedy => Some(("temperature", json!(0))), // llama.cpp samples greedily at temperature 0: https://github.com/ggml-org/llama.cpp/issues/9904
            Decoding::ServerDefault => None,
        }
    }
}
