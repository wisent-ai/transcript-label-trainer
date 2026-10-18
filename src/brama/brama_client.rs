use super::*;

impl BramaClient {
    pub fn new(url: &str, agent_id: String, secret: String, token: Option<String>) -> Result<Self> {
        // No clock on a request: it ends with Brama's answer or a transport
        // error, and reqwest's built-in cut-off is switched off.
        let http = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .map_err(|error| Error(format!("could not build the Brama HTTP client: {error}")))?;
        Ok(BramaClient {
            url: url.trim_end_matches('/').to_string(),
            agent_id,
            secret,
            token,
            http,
        })
    }

    /// The signing identity and the two secrets come from the environment or
    /// from the vault roles the environment declares; nothing names an agent
    /// or an item here. A variable that is set but empty is refused by name.
    pub fn from_env() -> Result<Self> {
        let agent_id = env_value("WISENT_APP_AGENT_ID")?.ok_or_else(|| {
            Error("WISENT_APP_AGENT_ID is required: the agent identity Brama verifies the signature against".to_string())
        })?;
        let secret = match env_value("WISENT_APP_AGENT_AUTH_SECRET")? {
            Some(secret) => secret,
            None => read_role("TLT_BRAMA_AGENT_ROLE", "the agent's signing secret")?,
        };
        let token = match env_value("BRAMA_TOKEN")? {
            Some(token) => token,
            None => read_role("TLT_BRAMA_TOKEN_ROLE", "the Brama bearer")?,
        };
        let url = resolve_url()?;
        Self::new(&url, agent_id, secret, Some(token))
    }

    /// The identity headers jeden signs with: the canonical string is
    /// `"{agent_id}:{timestamp}:{body_hash}"`, keyed by the shared secret. A
    /// clock set before the Unix epoch cannot sign a timestamp Brama accepts
    /// and is refused with the clock's own error.
    pub(crate) fn auth_headers(&self, body: &str) -> Result<Vec<(&'static str, String)>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| Error(format!("this host's clock is before the Unix epoch, so no request can be signed: {error}")))?
            .as_secs()
            .to_string();
        let body_hash = if body.is_empty() {
            String::new()
        } else {
            crate::util::sha256_hex(body)
        };
        let mut mac =
            <Hmac<Sha256> as Mac>::new_from_slice(self.secret.as_bytes()).map_err(|error| {
                Error(format!(
                    "the agent's signing secret cannot key an HMAC: {error}"
                ))
            })?;
        mac.update(format!("{}:{timestamp}:{body_hash}", self.agent_id).as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let mut headers = vec![
            ("content-type", "application/json".to_string()),
            ("x-agent-id", self.agent_id.clone()),
            ("x-agent-timestamp", timestamp),
            ("x-agent-body-sha256", body_hash),
            ("x-agent-signature", signature),
        ];
        if let Some(token) = &self.token {
            headers.push(("authorization", format!("Bearer {token}")));
        }
        Ok(headers)
    }

    /// One non-streaming chat completion; returns the message content. No
    /// output budget and no sampling setting is sent: how long an answer may
    /// be and how the routed model samples are its own, which Brama's route
    /// carries, not values chosen here.
    pub fn chat(&self, model: &str, messages: &[Message]) -> Result<String> {
        match self.answer(model, messages)? {
            ChatAnswer::Content(content) => Ok(content),
            ChatAnswer::DoesNotFit(detail) => bail!("Brama answered HTTP 400: {detail}"),
        }
    }

    /// One chat completion that tells a prompt the routed model cannot take
    /// (Brama's `context_length_exceeded`) apart from every other failure, so
    /// a caller can ask about less instead of guessing a size up front.
    pub(crate) fn answer(&self, model: &str, messages: &[Message]) -> Result<ChatAnswer> {
        let body = serde_json::json!({ "model": model, "messages": messages });
        let payload = match self.completion(&body)? {
            Completion::Answered(payload) => payload,
            Completion::DoesNotFit(detail) => return Ok(ChatAnswer::DoesNotFit(detail)),
        };
        let content = first_message(&payload)
            .and_then(|message| message.get("content"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error(format!("Brama response was not an OpenAI chat completion: {payload}")))?;
        Ok(ChatAnswer::Content(content.trim().to_string()))
    }

    /// The arguments of the one call of the function `name` the model must
    /// make: `parameters` is its JSON Schema, and the request names it as the
    /// required tool, so the answer is that object and is never searched for
    /// inside prose. An answer without exactly that one call, or whose
    /// arguments are not a JSON object, is refused with what Brama sent.
    pub(crate) fn call(
        &self,
        model: &str,
        messages: &[Message],
        name: &str,
        description: &str,
        parameters: serde_json::Value,
    ) -> Result<serde_json::Map<String, serde_json::Value>> {
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": { "name": name, "description": description, "parameters": parameters },
            }],
            "tool_choice": { "type": "function", "function": { "name": name } },
        });
        let payload = match self.completion(&body)? {
            Completion::Answered(payload) => payload,
            Completion::DoesNotFit(detail) => bail!("Brama answered HTTP 400: {detail}"),
        };
        let calls = first_message(&payload)
            .and_then(|message| message.get("tool_calls"))
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| Error(format!("{model} made no call of {name}: {payload}")))?;
        let [call] = calls.as_slice() else {
            bail!("{model} did not answer by exactly one call of {name}: {payload}");
        };
        if call.pointer("/function/name").and_then(serde_json::Value::as_str) != Some(name) {
            bail!("{model} called a function other than {name}: {payload}");
        }
        let arguments = call
            .pointer("/function/arguments")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error(format!("{model}'s call of {name} carries no arguments: {payload}")))?;
        match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(serde_json::Value::Object(object)) => Ok(object),
            Ok(other) => bail!("{model}'s call of {name} has arguments that are not an object: {other}"),
            Err(error) => bail!("{model}'s call of {name} has arguments that are not JSON ({error}): {arguments}"),
        }
    }

    /// POST one chat completion `body`, signed; the JSON Brama answered, or
    /// the prompt-does-not-fit body Brama names `context_length_exceeded`.
    /// Any other refusal is the error, with Brama's status and body.
    fn completion(&self, body: &serde_json::Value) -> Result<Completion> {
        let body = serde_json::to_string(body)?;
        let mut request = self.http.post(format!("{}/v1/chat/completions", self.url));
        for (name, value) in self.auth_headers(&body)? {
            request = request.header(name, value);
        }
        let response = stado_wait::request_blocking(request.body(body))
            .map_err(|error| Error(format!("Brama unreachable at {}: {error}", self.url)))?;
        let status = response.status();
        let text = response.text().map_err(|error| {
            Error(format!(
                "Brama answered HTTP {status} but its body could not be read: {error}"
            ))
        })?;
        if status != reqwest::StatusCode::OK {
            let detail = text.trim();
            let detail = if detail.is_empty() {
                "(empty body)"
            } else {
                detail
            };
            // Brama states a prompt the route cannot take as error.code; a
            // body that is not JSON or names no code is reported whole below.
            if let Ok(payload) = serde_json::from_str::<serde_json::Value>(detail) {
                if payload
                    .pointer("/error/code")
                    .and_then(serde_json::Value::as_str)
                    == Some("context_length_exceeded")
                {
                    return Ok(Completion::DoesNotFit(detail.to_string()));
                }
            }
            bail!("Brama answered HTTP {status}: {detail}")
        }
        let payload = serde_json::from_str(&text).map_err(|error| {
            Error(format!(
                "Brama answered HTTP {status} with a body that is not JSON ({error}): {}",
                text.trim()
            ))
        })?;
        Ok(Completion::Answered(payload))
    }
}

/// What Brama answered one signed chat completion request.
enum Completion {
    /// The completion's JSON body.
    Answered(serde_json::Value),
    /// The prompt does not fit the routed model; Brama's answer body.
    DoesNotFit(String),
}

/// The message of a completion's first choice.
fn first_message(payload: &serde_json::Value) -> Option<&serde_json::Value> {
    payload
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
}

/// What Brama answered one chat completion.
pub(crate) enum ChatAnswer {
    /// The model's message content.
    Content(String),
    /// The prompt does not fit the routed model; Brama's answer body.
    DoesNotFit(String),
}
