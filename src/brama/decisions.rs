//! Typed choices through Brama's decision endpoint (`POST /v1/decisions`):
//! a question about a text is answered with one of the labels it offers,
//! read from the answer's structure, never searched for in prose.
//!
//! The endpoint answers exactly two aliases: `decision-model`, the route the
//! gateway pays for, and `best-decision-model`, which delegates to `best`,
//! the caller's own subscription. A command naming the chat alias `best`
//! asks the latter; any other chat alias cannot answer a typed choice and is
//! refused naming the two that can.

use super::*;

/// The decision alias that delegates to the caller's `best` subscription.
pub const BEST_DECISION_MODEL: &str = "best-decision-model";
/// The decision alias the gateway's own route answers.
pub const DECISION_MODEL: &str = "decision-model";

/// The decision alias a command's model setting names: `best` asks
/// `best-decision-model`, a decision alias is taken as written, and any other
/// alias is refused, since `POST /v1/decisions` answers nothing else.
pub fn decision_alias(model: &str) -> Result<&'static str> {
    match model {
        BEST_MODEL | BEST_DECISION_MODEL => Ok(BEST_DECISION_MODEL),
        DECISION_MODEL => Ok(DECISION_MODEL),
        other => Err(Error(format!(
            "`{other}` cannot answer a typed choice: Brama's POST /v1/decisions answers only \
             `{DECISION_MODEL}` and `{BEST_DECISION_MODEL}` (`{BEST_MODEL}` asks the latter)"
        ))),
    }
}

impl BramaClient {
    /// Ask `model` (a decision alias) one choice about `state`: `instructions`
    /// say what to decide, and the answer is one of `labels` (label, meaning).
    /// A refusal, an answer without the choice, or a label outside `labels` is
    /// an error naming the question, never a guess.
    pub fn choose(
        &self,
        model: &str,
        state: &str,
        instructions: &str,
        labels: &[(&str, &str)],
    ) -> Result<String> {
        let criteria: serde_json::Map<String, serde_json::Value> = labels
            .iter()
            .map(|(label, meaning)| ((*label).to_string(), serde_json::json!(meaning)))
            .collect();
        let body = serde_json::to_string(&serde_json::json!({
            "model": model,
            "state": state,
            "questions": {
                "choice": { "type": "choice", "instructions": instructions, "criteria": criteria },
            },
        }))?;
        let mut request = self.http.post(format!("{}/v1/decisions", self.url));
        for (name, value) in self.auth_headers(&body)? {
            request = request.header(name, value);
        }
        let response = stado_wait::request_blocking(request.body(body))
            .map_err(|error| Error(format!("Brama unreachable at {}: {error}", self.url)))?;
        let status = response.status();
        let text = response.text().map_err(|error| {
            Error(format!("Brama answered HTTP {status} to a decision but its body could not be read: {error}"))
        })?;
        if status != reqwest::StatusCode::OK {
            bail!(
                "Brama refused the decision `{instructions}` with HTTP {status}: {}",
                text.trim()
            )
        }
        let answer: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            Error(format!(
                "Brama answered the decision with a body that is not JSON ({error}): {}",
                text.trim()
            ))
        })?;
        let label = answer
            .pointer("/answers/choice/choice")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error(format!(
                    "Brama's decision carries no answers.choice.choice: {answer}"
                ))
            })?;
        if !labels.iter().any(|(known, _)| *known == label) {
            bail!(
                "Brama chose `{label}` for `{instructions}`, which is not one of the labels asked"
            );
        }
        Ok(label.to_string())
    }
}
